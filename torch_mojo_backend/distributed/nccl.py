"""ctypes binding for NCCL (libnccl.so.2) from the nvidia-nccl-cu12 wheel.

Why ctypes and not torch's own NCCL: this backend must work with a CPU-only
torch install (see "Rules about the eager mode" in AGENTS.md), so we cannot
rely on torch.distributed.ProcessGroupNCCL or torch.cuda being functional.
The cu12 wheel's libnccl.so.2 statically links the CUDA runtime and dlopens
libcuda.so.1 by itself, so this binding has zero dependencies beyond the
wheel and a CUDA driver — the same trick mojo_device/cuda_peer.py uses.

Device selection: NCCL infers the target GPU from the CUDA context that is
current on the calling thread (it queries the driver it dlopened). MAX binds
the per-device *primary* context, so memory allocated by MAX is directly
valid for NCCL. `set_current_cuda_device()` below performs the minimal
driver-API dance (cuInit -> cuDevicePrimaryCtxRetain -> cuCtxSetCurrent)
that torch's `cudaSetDevice` would have done, without needing libcudart.

Enum values are pinned to nccl.h from NCCL 2.27+ (verified against the
2.31.2 header); NCCL keeps these ABI-stable across 2.x.
"""

import ctypes
import functools
import os
from pathlib import Path

# nccl.h: ncclResult_t
NCCL_SUCCESS = 0
NCCL_IN_PROGRESS = 7

# nccl.h: ncclRedOp_t
NCCL_SUM = 0
NCCL_PROD = 1
NCCL_MAX = 2
NCCL_MIN = 3
NCCL_AVG = 4

# nccl.h: ncclDataType_t
NCCL_INT8 = 0
NCCL_UINT8 = 1
NCCL_INT32 = 2
NCCL_UINT32 = 3
NCCL_INT64 = 4
NCCL_UINT64 = 5
NCCL_FLOAT16 = 6
NCCL_FLOAT32 = 7
NCCL_FLOAT64 = 8
NCCL_BFLOAT16 = 9

NCCL_UNIQUE_ID_BYTES = 128

_NCCL_LIB_ENV = "TORCH_MOJO_BACKEND_NCCL_LIB"


class NcclUniqueId(ctypes.Structure):
    """nccl.h: typedef struct { char internal[128]; } ncclUniqueId."""

    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


class NcclError(RuntimeError):
    """An NCCL call returned a non-success ncclResult_t."""

    def __init__(self, func_name: str, result: int, detail: str):
        super().__init__(f"{func_name} failed: {detail} (ncclResult_t={result})")
        self.result = result


def _candidate_libnccl_paths() -> list[str]:
    override = os.environ.get(_NCCL_LIB_ENV)
    if override:
        return [override]
    candidates = []
    try:
        import nvidia.nccl

        # nvidia.nccl is a namespace package: no __file__, only __path__.
        for package_dir in nvidia.nccl.__path__:
            candidates.append(str(Path(package_dir) / "lib" / "libnccl.so.2"))
    except ImportError:
        pass
    # System fallbacks, same spirit as MAX's comm/vendor/ccl.mojo search list.
    candidates += ["libnccl.so.2", "libnccl.so"]
    return candidates


@functools.cache
def _libnccl() -> ctypes.CDLL:
    errors = []
    lib = None
    for path in _candidate_libnccl_paths():
        try:
            # RTLD_GLOBAL so a later dlopen of the "libnccl.so.2" soname (for
            # example by MAX's optional vendor-CCL bridge) resolves to this
            # exact library instead of a mismatched system copy.
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as e:
            errors.append(f"{path}: {e}")
    if lib is None:
        raise RuntimeError(
            "could not load libnccl.so.2 — install the nvidia-nccl-cu12 wheel "
            f"or set {_NCCL_LIB_ENV} to the library path. Tried:\n  "
            + "\n  ".join(errors)
        )

    lib.ncclGetErrorString.restype = ctypes.c_char_p
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
    # ncclUniqueId is passed BY VALUE — it must be a ctypes.Structure here: a
    # bare (c_char * 128) array argtype decays to a pointer like a C array,
    # shifting every following argument (NCCL then sees a garbage rank).
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        NcclUniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommUserRank.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    for name, extra in [
        ("ncclAllReduce", [ctypes.c_int, ctypes.c_int]),  # datatype, op
        ("ncclReduceScatter", [ctypes.c_int, ctypes.c_int]),
        ("ncclAllGather", [ctypes.c_int]),  # datatype
    ]:
        fn = getattr(lib, name)
        fn.argtypes = (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            + extra
            + [ctypes.c_void_p, ctypes.c_void_p]  # comm, stream
        )
    lib.ncclBroadcast.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,  # datatype
        ctypes.c_int,  # root
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.ncclReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,  # datatype
        ctypes.c_int,  # op
        ctypes.c_int,  # root
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    for name in ["ncclSend", "ncclRecv"]:
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,  # datatype
            ctypes.c_int,  # peer
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
    lib.ncclGroupStart.argtypes = []
    lib.ncclGroupEnd.argtypes = []
    return lib


def _check(func_name: str, result: int):
    if result != NCCL_SUCCESS:
        detail = _libnccl().ncclGetErrorString(result).decode()
        raise NcclError(func_name, result, detail)


def nccl_version() -> int:
    """The runtime library's version code, e.g. 23102 for 2.31.2."""
    version = ctypes.c_int(0)
    _check("ncclGetVersion", _libnccl().ncclGetVersion(ctypes.byref(version)))
    return version.value


def get_unique_id() -> bytes:
    """Generate the 128-byte communicator id (rank 0 only; share via the store)."""
    uid = NcclUniqueId()
    _check("ncclGetUniqueId", _libnccl().ncclGetUniqueId(ctypes.byref(uid)))
    # Not uid.internal: ctypes truncates c_char-array fields at the first NUL.
    return ctypes.string_at(ctypes.byref(uid), NCCL_UNIQUE_ID_BYTES)


class NcclComm:
    """One NCCL communicator, bound to the CUDA device that was current at init."""

    def __init__(self, handle: int):
        self._handle = handle
        self._aborted = False

    @classmethod
    def init_rank(cls, nranks: int, unique_id: bytes, rank: int) -> "NcclComm":
        """Collective, blocking: every rank of the clique must call concurrently."""
        if len(unique_id) != NCCL_UNIQUE_ID_BYTES:
            raise ValueError(f"unique_id must be {NCCL_UNIQUE_ID_BYTES} bytes")
        uid = NcclUniqueId.from_buffer_copy(unique_id)
        handle = ctypes.c_void_p(0)
        _check(
            "ncclCommInitRank",
            _libnccl().ncclCommInitRank(ctypes.byref(handle), nranks, uid, rank),
        )
        assert handle.value is not None  # a checked init never leaves it null
        return cls(handle.value)

    def destroy(self):
        if self._handle and not self._aborted:
            _libnccl().ncclCommDestroy(ctypes.c_void_p(self._handle))
            self._handle = 0

    def abort(self):
        if self._handle:
            _libnccl().ncclCommAbort(ctypes.c_void_p(self._handle))
            self._aborted = True
            self._handle = 0

    def async_error(self) -> int:
        err = ctypes.c_int(0)
        _check(
            "ncclCommGetAsyncError",
            _libnccl().ncclCommGetAsyncError(
                ctypes.c_void_p(self._handle), ctypes.byref(err)
            ),
        )
        return err.value

    def all_reduce(
        self, send_ptr: int, recv_ptr: int, count: int, dtype: int, op: int, stream: int
    ):
        _check(
            "ncclAllReduce",
            _libnccl().ncclAllReduce(
                send_ptr, recv_ptr, count, dtype, op, self._handle, stream
            ),
        )

    def broadcast(
        self,
        send_ptr: int,
        recv_ptr: int,
        count: int,
        dtype: int,
        root: int,
        stream: int,
    ):
        _check(
            "ncclBroadcast",
            _libnccl().ncclBroadcast(
                send_ptr, recv_ptr, count, dtype, root, self._handle, stream
            ),
        )

    def reduce(
        self,
        send_ptr: int,
        recv_ptr: int,
        count: int,
        dtype: int,
        op: int,
        root: int,
        stream: int,
    ):
        _check(
            "ncclReduce",
            _libnccl().ncclReduce(
                send_ptr, recv_ptr, count, dtype, op, root, self._handle, stream
            ),
        )

    def all_gather(
        self, send_ptr: int, recv_ptr: int, send_count: int, dtype: int, stream: int
    ):
        _check(
            "ncclAllGather",
            _libnccl().ncclAllGather(
                send_ptr, recv_ptr, send_count, dtype, self._handle, stream
            ),
        )

    def reduce_scatter(
        self,
        send_ptr: int,
        recv_ptr: int,
        recv_count: int,
        dtype: int,
        op: int,
        stream: int,
    ):
        _check(
            "ncclReduceScatter",
            _libnccl().ncclReduceScatter(
                send_ptr, recv_ptr, recv_count, dtype, op, self._handle, stream
            ),
        )

    def send(self, ptr: int, count: int, dtype: int, peer: int, stream: int):
        _check(
            "ncclSend",
            _libnccl().ncclSend(ptr, count, dtype, peer, self._handle, stream),
        )

    def recv(self, ptr: int, count: int, dtype: int, peer: int, stream: int):
        _check(
            "ncclRecv",
            _libnccl().ncclRecv(ptr, count, dtype, peer, self._handle, stream),
        )


def group_start():
    _check("ncclGroupStart", _libnccl().ncclGroupStart())


def group_end():
    _check("ncclGroupEnd", _libnccl().ncclGroupEnd())


# --- CUDA driver context selection ------------------------------------------
# NCCL picks its GPU from the thread's current CUDA context. cuda_peer.py
# already dlopens libcuda for pointer queries; here we need three more driver
# calls. cudaSetDevice() in the runtime API is exactly this sequence.

CUDA_SUCCESS = 0


@functools.cache
def _libcuda() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    return lib


def _check_cu(func_name: str, result: int):
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"{func_name} failed (CUresult={result})")


def set_current_cuda_device(ordinal: int):
    """Make `ordinal`'s primary context current on this thread (for NCCL init)."""
    lib = _libcuda()
    _check_cu("cuInit", lib.cuInit(0))
    device = ctypes.c_int(0)
    _check_cu("cuDeviceGet", lib.cuDeviceGet(ctypes.byref(device), ordinal))
    context = ctypes.c_void_p(0)
    _check_cu(
        "cuDevicePrimaryCtxRetain",
        lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
    )
    _check_cu("cuCtxSetCurrent", lib.cuCtxSetCurrent(context))
