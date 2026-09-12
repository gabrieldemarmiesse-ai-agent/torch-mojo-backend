"""Device timing for torch.autograd.profiler / torch.profiler on the mojo device.

PyTorch's hook for a PrivateUse1 backend's profiler is C++ only (``ProfilerStubs``).
``shim.cpp`` adapts it to a C function table; it is compiled on first use against the
installed torch with ``torch.utils.cpp_extension`` (a page of C++, one torch header),
cached per torch version next to this file, and loaded with ctypes. This module fills
the table. The current implementation is a DUMMY: events are host timestamps, so the
"device time" it reports is host wall time between the profiler's record calls.
"""

import ctypes
import time
from pathlib import Path

import torch

_NAME = "tmb_profiler_shim"
_lib: ctypes.CDLL | None = None
_hooks: object | None = None  # keeps the callback objects alive
_events: dict[int, int] = {}


class _Hooks(ctypes.Structure):
    _fields_ = [
        ("record", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32))),
        ("elapsed", ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p)),
        ("release", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("mark", ctypes.CFUNCTYPE(None, ctypes.c_char_p)),
        ("range_push", ctypes.CFUNCTYPE(None, ctypes.c_char_p)),
        ("range_pop", ctypes.CFUNCTYPE(None)),
        ("device_count", ctypes.CFUNCTYPE(ctypes.c_int32)),
        ("synchronize", ctypes.CFUNCTYPE(None)),
    ]


def build_directory() -> Path:
    """One cache directory per installed torch: the shim binds to its headers."""
    return Path(__file__).parent / "__shimcache__" / f"torch-{torch.__version__}"


def shim_path() -> Path:
    return build_directory() / f"{_NAME}.so"


def load_shim(verbose: bool = False) -> Path:
    """Compile the shim if this torch has no cached one, then return its path."""
    path = shim_path()
    if not path.is_file():
        from torch.utils.cpp_extension import load  # noqa: PLC0415 -- pulls in the build toolchain; only needed here

        build_directory().mkdir(parents=True, exist_ok=True)
        load(
            name=_NAME,
            sources=[str(Path(__file__).parent / "shim.cpp")],
            is_python_module=False,
            build_directory=str(build_directory()),
            extra_cflags=["-O1"],
            verbose=verbose,
        )
    return path


def _dummy_hooks() -> _Hooks:
    counter = [0]

    def record(device_index: "ctypes._Pointer[ctypes.c_int32]") -> int:
        counter[0] += 1
        handle = counter[0]
        _events[handle] = time.perf_counter_ns()
        device_index[0] = 0
        return handle

    def elapsed(a: int, b: int) -> float:
        return (_events.get(b, 0) - _events.get(a, 0)) / 1000.0

    def release(handle: int):
        _events.pop(handle, None)

    def device_count() -> int:
        return torch.accelerator.device_count()

    def synchronize():
        torch.accelerator.synchronize()

    return _Hooks(
        _Hooks._fields_[0][1](record),
        _Hooks._fields_[1][1](elapsed),
        _Hooks._fields_[2][1](release),
        _Hooks._fields_[3][1](lambda name: None),
        _Hooks._fields_[4][1](lambda name: None),
        _Hooks._fields_[5][1](lambda: None),
        _Hooks._fields_[6][1](device_count),
        _Hooks._fields_[7][1](synchronize),
    )


def register() -> Path:
    """Load the shim and install the (dummy) hooks as the mojo device's profiler."""
    global _lib, _hooks
    if _lib is not None:
        return shim_path()
    path = load_shim()
    _lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _lib.tmb_profiler_register.argtypes = [ctypes.POINTER(_Hooks)]
    _lib.tmb_profiler_register.restype = ctypes.c_int32
    _hooks = _dummy_hooks()
    if _lib.tmb_profiler_register(ctypes.byref(_hooks)) != 0:
        raise RuntimeError("tmb_profiler_register failed")
    return path


__all__ = ["build_directory", "load_shim", "register", "shim_path"]
