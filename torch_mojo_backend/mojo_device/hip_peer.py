"""HIP-runtime facts about mojo tensors on AMD GPUs: which device owns a
pointer, and making a device current on the calling thread.

The AMD counterpart of `cuda_peer.py`, with the same posture: everything is
asked of the runtime that MAX itself drives, so the answers are facts about an
allocation rather than assumptions about how two runtimes enumerate. MAX
dlopens `libamdhip64.so` from the ROCm install (``$ROCM_PATH`` or
``/opt/rocm``); this module reuses THAT copy — found through the process's own
memory map — instead of resolving the soname a second time, which on a box
with several ROCm versions could map a different runtime with its own device
table. RCCL links against the same soname, so the ordinals here, MAX's and
RCCL's are one numbering.

Only the driver-style `hipPointerGetAttribute` (singular) is used: it writes
a plain int, where `hipPointerGetAttributes` fills a struct whose layout
changed in ROCm 6.0 — the trap cuda_peer.py describes for the CUDA runtime
API.
"""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path

# hip/driver_types.h: hipPointer_attribute (CUDA's CUpointer_attribute values).
_ATTRIBUTE_MEMORY_TYPE = 2
_ATTRIBUTE_DEVICE_ORDINAL = 9
# hip/hip_runtime_api.h: hipMemoryType, ROCm >= 6.0 numbering.
_MEMORYTYPE_DEVICE = 2
_MEMORYTYPE_MANAGED = 3
_MEMORYTYPE_UNIFIED = 11

_SONAMES = ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so")
HIP_SUCCESS = 0


def _mapped_runtime_path() -> Path | None:
    """The libamdhip64 already loaded in this process (by MAX), if any."""
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.rstrip("\n").partition(" /")[2]
                if path and Path("/" + path).name.startswith("libamdhip64.so"):
                    return Path("/" + path)
    except OSError:
        pass
    return None


def _candidate_runtime_paths() -> list[str]:
    candidates: list[str] = []
    mapped = _mapped_runtime_path()
    if mapped is not None:
        candidates.append(str(mapped))
    candidates.extend(_SONAMES)
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"):
        if root:
            candidates.extend(str(Path(root) / "lib" / name) for name in _SONAMES)
    return candidates


@functools.cache
def _runtime() -> ctypes.CDLL | None:
    """libamdhip64, or None where this is not a ROCm stack."""
    for path in _candidate_runtime_paths():
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        lib.hipPointerGetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.hipPointerGetAttribute.restype = ctypes.c_int
        lib.hipSetDevice.argtypes = [ctypes.c_int]
        lib.hipSetDevice.restype = ctypes.c_int
        lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipGetDevice.restype = ctypes.c_int
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        return lib
    return None


def available() -> bool:
    return _runtime() is not None


def runtime_dir() -> Path | None:
    """Directory of the HIP runtime in use — where its ROCm's librccl lives too."""
    mapped = _mapped_runtime_path()
    return mapped.parent if mapped is not None else None


def _pointer_attribute(ptr: int, attribute: int) -> int | None:
    lib = _runtime()
    if lib is None or not ptr:
        return None
    value = ctypes.c_int(-1)
    status = lib.hipPointerGetAttribute(
        ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_void_p(ptr)
    )
    return value.value if status == HIP_SUCCESS else None


def device_ordinal(ptr: int) -> int | None:
    """The HIP ordinal owning `ptr`, or None if it is not GPU memory.

    Device, managed and unified allocations all count: on an APU (MI300A)
    the HBM pool is shared with the host, and the runtime classifies a
    `hipMalloc` there the same way it does on a discrete card, but the
    ordinal is the useful fact for every kind the GPU can address natively.
    """
    kind = _pointer_attribute(ptr, _ATTRIBUTE_MEMORY_TYPE)
    if kind not in (_MEMORYTYPE_DEVICE, _MEMORYTYPE_MANAGED, _MEMORYTYPE_UNIFIED):
        return None
    return _pointer_attribute(ptr, _ATTRIBUTE_DEVICE_ORDINAL)


def set_device(ordinal: int):
    """Make `ordinal` this thread's current HIP device (thread-local in HIP).

    `cudaSetDevice`'s counterpart: RCCL resolves the GPU a communicator binds
    to from it at `ncclCommInitRank`.
    """
    lib = _runtime()
    if lib is None:
        raise RuntimeError("libamdhip64 is not loadable; is this a ROCm system?")
    status = lib.hipSetDevice(ordinal)
    if status != HIP_SUCCESS:
        detail = lib.hipGetErrorString(status).decode()
        raise RuntimeError(
            f"hipSetDevice({ordinal}) failed: {detail} (hipError_t={status})"
        )


def warn_if_cuda_torch_on_hip():
    """One-time hint at registration: a CUDA torch wheel on an AMD box is slow.

    The HIP runtime walks every shared object mapped in the process on each
    kernel load (`dl_iterate_phdr` from libhsa-runtime64, hunting embedded
    code objects), and the CUDA torch wheel maps ~3 GB of NVIDIA libraries it
    never uses here. Measured on 4x MI300A, nanoGPT 124M under DDP: the first
    training step took 14.7 s with torch 2.11+cu130 against 1.0 s with
    torch 2.11+cpu, and steady state ran 6% slower as well. The CPU wheel is
    what this backend needs anyway.
    """
    import warnings

    import torch

    from torch_mojo_backend.torch_compile_backend.utils import get_accelerators

    if getattr(torch.version, "cuda", None) is None:
        return
    if not any(device.api == "hip" for device in get_accelerators()):
        return
    warnings.warn(
        "torch-mojo-backend: this torch is a CUDA build "
        f"(torch {torch.__version__}) but the GPUs here are AMD. The HIP "
        "runtime rescans every mapped library at each kernel load, and the "
        "CUDA wheel maps gigabytes of unused NVIDIA libraries, so first-use "
        "kernel loads are ~10x slower. Install the CPU wheel instead: "
        "uv pip install torch --index-url https://download.pytorch.org/whl/cpu",
        RuntimeWarning,
        stacklevel=2,
    )
