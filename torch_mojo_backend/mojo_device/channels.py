"""Side CUDA streams ("channels") for the mojo device.

The eager backend runs all compute, transfers, and stream-ordered frees on
ONE stream per device — the MAX DeviceContext's default stream (see
docs/kernel_call_queue.md). That invariant stays: channels are *additional*
streams for work that should overlap the default stream, such as NCCL
collectives (torch_mojo_backend/distributed) or, later, prefetch copies.

A channel wraps a `max.driver.DeviceStream` (MAX owns the stream; keeping
the Channel alive keeps the stream alive) plus the raw `CUstream` handle,
which MAX documents as usable by native code outside its runtime — for
example, to record CUDA events on it. Cross-stream ordering is done with
raw driver events on the native handles, through this module's own minimal
libcuda binding (the same pattern as mojo_device/cuda_peer.py): the driver
API is thread-safe, so a worker thread can poll or wait a channel's events
without touching MAX's DeviceContext, whose thread-safety is undocumented.

Correctness rules for channel users:

1. Drain the kernel-call queue before ordering a channel against the
   default stream — a queued-but-unlaunched kernel is invisible to any
   stream (`deferred_compile.drain()`).
2. Hold a Python reference to every tensor a channel reads or writes until
   the work is known complete (an end event fired). Frees are
   stream-ordered on the DEFAULT stream, so a buffer freed while a channel
   still uses it is a use-after-free.
3. Anything the default stream must observe from a channel needs an
   explicit fence (event or host wait); nothing is implicit.

CUDA-only for now: channels raise on devices whose API is not "cuda".
"""

import ctypes
import functools
import threading

import max.driver

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_READY = 600
_CU_EVENT_DISABLE_TIMING = 0x2


@functools.cache
def _driver() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cuEventQuery.argtypes = [ctypes.c_void_p]
    lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
    lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
    lib.cuEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cuStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cuStreamQuery.argtypes = [ctypes.c_void_p]
    return lib


_CONTEXT_CURRENT = threading.local()


def ensure_context_current(ordinal: int) -> None:
    """Make `ordinal`'s primary CUDA context current on this thread.

    Driver objects (events, streams) are context-scoped; creating one from a
    fresh thread needs the device's context current first. Cached per thread.
    """
    if getattr(_CONTEXT_CURRENT, "ordinal", None) == ordinal:
        return
    lib = _driver()
    _check("cuInit", lib.cuInit(0))
    device = ctypes.c_int(0)
    _check("cuDeviceGet", lib.cuDeviceGet(ctypes.byref(device), ordinal))
    context = ctypes.c_void_p(0)
    _check(
        "cuDevicePrimaryCtxRetain",
        lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
    )
    _check("cuCtxSetCurrent", lib.cuCtxSetCurrent(context))
    _CONTEXT_CURRENT.ordinal = ordinal


def _check(func_name: str, result: int) -> None:
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"{func_name} failed (CUresult={result})")


class CudaEvent:
    """One driver-API event, created on the caller's current CUDA context."""

    def __init__(self, enable_timing: bool = False) -> None:
        handle = ctypes.c_void_p(0)
        flags = 0 if enable_timing else _CU_EVENT_DISABLE_TIMING
        _check("cuEventCreate", _driver().cuEventCreate(ctypes.byref(handle), flags))
        self._handle = handle
        self.enable_timing = enable_timing

    def elapsed_time_ms(self, end: "CudaEvent") -> float:
        """Milliseconds between this (start) event and `end`; both timed."""
        out = ctypes.c_float(0.0)
        _check(
            "cuEventElapsedTime",
            _driver().cuEventElapsedTime(ctypes.byref(out), self._handle, end._handle),
        )
        return out.value

    def record(self, stream_handle: int) -> None:
        _check("cuEventRecord", _driver().cuEventRecord(self._handle, stream_handle))

    def query(self) -> bool:
        """True once every piece of work recorded before the event finished."""
        result = _driver().cuEventQuery(self._handle)
        if result == CUDA_ERROR_NOT_READY:
            return False
        _check("cuEventQuery", result)
        return True

    def synchronize(self) -> None:
        _check("cuEventSynchronize", _driver().cuEventSynchronize(self._handle))

    def destroy(self) -> None:
        if self._handle is not None:
            # The driver defers the actual release until captured work
            # completes, so destroying right after enqueuing waits is safe.
            _driver().cuEventDestroy_v2(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass  # interpreter shutdown; the context is going away anyway


class Channel:
    """One extra CUDA stream on a mojo device, orderable against the default."""

    def __init__(
        self, device: max.driver.Device, name: str, wrap_default: bool = False
    ) -> None:
        if device.api != "cuda":
            raise NotImplementedError(
                f"channels require a CUDA device (got api={device.api!r})"
            )
        self.device = device
        self.name = name
        self._default_handle = device.default_stream.native_stream_handle
        self.is_default = wrap_default
        if wrap_default:
            self._stream = device.default_stream
            self.handle = self._default_handle
        else:
            self._stream = max.driver.DeviceStream(device)  # MAX owns; keep alive
            self.handle = self._stream.native_stream_handle

    def wait_default_stream(self) -> None:
        """Order this channel after everything enqueued on the default stream.

        Callers must have drained the kernel-call queue first (rule 1 above).
        """
        if self.is_default:
            return
        event = CudaEvent()
        event.record(self._default_handle)
        _check(
            "cuStreamWaitEvent",
            _driver().cuStreamWaitEvent(self.handle, event._handle, 0),
        )
        event.destroy()

    def make_default_stream_wait(self) -> None:
        """Order subsequent default-stream work after this channel's work."""
        if self.is_default:
            return
        event = CudaEvent()
        event.record(self.handle)
        _check(
            "cuStreamWaitEvent",
            _driver().cuStreamWaitEvent(self._default_handle, event._handle, 0),
        )
        event.destroy()

    def wait_event(self, event: CudaEvent) -> None:
        """Order subsequent work on this channel after `event`."""
        _check(
            "cuStreamWaitEvent",
            _driver().cuStreamWaitEvent(self.handle, event._handle, 0),
        )

    def record_event(self, event: CudaEvent | None = None) -> CudaEvent:
        """An event capturing everything enqueued on this channel so far."""
        if event is None:
            event = CudaEvent()
        event.record(self.handle)
        return event

    def query(self) -> bool:
        """True when everything enqueued on this channel has completed."""
        result = _driver().cuStreamQuery(self.handle)
        if result == CUDA_ERROR_NOT_READY:
            return False
        _check("cuStreamQuery", result)
        return True

    def synchronize(self) -> None:
        _check("cuStreamSynchronize", _driver().cuStreamSynchronize(self.handle))


_CHANNELS: dict[tuple[int, str], Channel] = {}
_CHANNELS_LOCK = threading.Lock()


def get_channel(device: max.driver.Device, name: str) -> Channel:
    """The (device, name) channel, created on first use. Thread-safe.

    The reserved name "default" wraps the device's default stream instead of
    creating a new one.
    """
    key = (device.id, name)
    with _CHANNELS_LOCK:
        channel = _CHANNELS.get(key)
        if channel is None:
            channel = Channel(device, name, wrap_default=(name == "default"))
            _CHANNELS[key] = channel
        return channel


def default_channel(device: max.driver.Device) -> Channel:
    """The channel view of the device's default stream."""
    return get_channel(device, "default")
