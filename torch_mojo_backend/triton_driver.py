"""Run Triton kernels on the mojo device.

Triton compiles and launches through its own CUDA backend (bundled ptxas,
libcuda from the display driver), so it needs no CUDA build of torch; the one
thing it asks torch for is "which device and stream is current", through a
driver object. This driver answers with the mojo device: launches land on the
mojo current stream's vendor handle, so they are ordered with our kernels.
The CUDA device ordinal is the mojo device index (MAX enumerates accelerators
in CUDA order, and CUDA_VISIBLE_DEVICES applies to both).
"""

from __future__ import annotations

import ctypes
import functools
from typing import TYPE_CHECKING

import torch

from torch_mojo_backend.native import device_module

if TYPE_CHECKING:
    from triton.backends.driver import DriverBase

_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


@functools.cache
def _device_capability(device: int) -> tuple[int, int]:
    """(major, minor) straight from the CUDA driver API, like Triton's own
    torch-free path; independent of the torch build and Triton version."""
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit(0)
    handle = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(handle), device) != 0:
        raise RuntimeError(f"cuDeviceGet({device}) failed")
    major, minor = ctypes.c_int(), ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(major), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, handle
    )
    cuda.cuDeviceGetAttribute(
        ctypes.byref(minor), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, handle
    )
    return (major.value, minor.value)


def _cuda_accelerator() -> bool:
    from torch_mojo_backend.torch_compile_backend.utils import (  # noqa: PLC0415 -- imports max.driver; keep it off the import path
        get_accelerators,
    )

    return any(getattr(d, "api", "") == "cuda" for d in get_accelerators())


class _DeviceInterface:
    """What triton.testing (do_bench) asks of the `torch.cuda` module."""

    @staticmethod
    def Event(enable_timing: bool = False) -> torch.Event:  # noqa: N802 -- torch.cuda's spelling
        return torch.Event(device="mojo", enable_timing=enable_timing)

    synchronize = staticmethod(device_module.synchronize)
    current_device = staticmethod(device_module.current_device)

    @staticmethod
    def empty_cache():
        pass


@functools.cache
def driver_class() -> type[DriverBase]:
    """A Triton driver for the mojo device (a CudaDriver whose device and
    stream come from torch.mojo).

    Cached because it is also the class of a registered Triton backend
    (`monkeypatching.register_the_mojo_triton_target`), and Triton compares
    the active driver against it with `isinstance`."""
    from triton.backends.nvidia.driver import (  # noqa: PLC0415 -- triton is optional
        CudaDriver,
        CudaLauncher,
        CudaUtils,
    )

    class MojoTritonDriver(CudaDriver):
        def __init__(self):
            # not the base constructors: they bind torch.cuda
            self.utils = CudaUtils()
            self.launcher_cls = CudaLauncher
            self.get_device_capability = _device_capability
            self.get_current_device = device_module.current_device
            self.set_current_device = device_module.set_device
            self.get_current_stream = self._mojo_stream

        @staticmethod
        def _mojo_stream(device: int) -> int:
            return device_module.stream_native_handle(
                device_module.current_stream(device)
            )

        def get_active_torch_device(self) -> torch.device:
            return torch.device("mojo", self.get_current_device())

        def get_device_interface(self) -> type[_DeviceInterface]:
            return _DeviceInterface

        @staticmethod
        def is_active() -> bool:
            return _cuda_accelerator()

        def get_empty_cache_for_benchmark(self) -> torch.Tensor:
            return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="mojo")

    return MojoTritonDriver


def make_driver() -> DriverBase:
    return driver_class()()


def enable_triton():
    """Make Triton launch on the mojo device (call after register_mojo_devices)."""
    from triton.runtime import driver  # noqa: PLC0415 -- triton is optional

    driver.set_active(make_driver())
