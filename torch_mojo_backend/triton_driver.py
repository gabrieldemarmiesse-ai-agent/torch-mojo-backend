"""Run Triton kernels on the mojo device.

Triton compiles and launches through its own GPU backend (bundled ptxas,
libcuda / libamdhip64 from the display driver), so it needs no CUDA or ROCm
build of torch; the one thing it asks torch for is "which device and stream
is current", through a driver object. The drivers here answer with the mojo
device: launches land on the mojo current stream's vendor handle, so they are
ordered with our kernels. The vendor device ordinal is the mojo device index
(MAX enumerates accelerators in vendor order, and the *_VISIBLE_DEVICES
variables apply to both).

`register_mojo_devices()` arranges for the driver to be installed as soon as
`triton.runtime.driver` is imported (or right away if it already is), unless
TORCH_MOJO_BACKEND_TRITON=0. With a CUDA build of torch that also works, this
driver still wins once installed: every Triton launch then follows the mojo
current stream, not torch.cuda's.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

import torch

from torch_mojo_backend.native import device_module

if TYPE_CHECKING:
    from types import ModuleType

    from triton.backends.driver import DriverBase, GPUDriver

_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
_DRIVER_MODULE = "triton.runtime.driver"


@functools.cache
def accelerator_api() -> str:
    """ "cuda", "hip", "metal" or "cpu": what MAX drives on this machine."""
    from torch_mojo_backend.torch_compile_backend.utils import (  # noqa: PLC0415 -- imports max.driver; keep it off the import path
        get_accelerators,
    )

    for d in get_accelerators():
        api = getattr(d, "api", "")
        if api != "cpu":
            return str(api)
    return "cpu"


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


def _make_stream_context_current(stream: int):
    """A launch needs the context that owns the stream current on this
    thread, and Triton's launcher only sets one when none is; MAX streams
    live in MAX's own contexts (one per device), so the stream itself says
    which (cuStreamGetCtx). HIP streams are not bound to a context."""
    if accelerator_api() != "cuda":
        return
    cuda = ctypes.CDLL("libcuda.so.1")
    ctx = ctypes.c_void_p()
    if cuda.cuStreamGetCtx(ctypes.c_void_p(stream), ctypes.byref(ctx)) != 0:
        raise RuntimeError("cuStreamGetCtx failed for the mojo stream")
    if cuda.cuCtxSetCurrent(ctx) != 0:
        raise RuntimeError("cuCtxSetCurrent failed for the mojo stream's context")


def _current_stream_handle(device: int) -> int:
    """Triton asks this right before every launch, for the current device:
    the place to make the stream's driver context current as well."""
    handle = device_module.stream_native_handle(device_module.current_stream(device))
    _make_stream_context_current(handle)
    return handle


class _DeviceInterface:
    """What triton.testing (do_bench, the autotuner's benchmarker) asks of
    the `torch.cuda` module."""

    @staticmethod
    def Event(enable_timing: bool = False) -> torch.Event:  # noqa: N802 -- torch.cuda's spelling
        return torch.Event(device="mojo", enable_timing=enable_timing)

    synchronize = staticmethod(device_module.synchronize)
    current_device = staticmethod(device_module.current_device)
    set_device = staticmethod(device_module.set_device)

    @staticmethod
    def empty_cache():
        pass


def _set_current_device(device: torch.device | str | int | None):
    device_module.set_device(
        device_module.current_device() if device is None else device
    )


def _bind_mojo(driver: GPUDriver):
    """The callables Triton's GPUDriver base takes from torch.cuda."""
    driver.get_current_device = device_module.current_device
    driver.set_current_device = _set_current_device
    driver.get_current_stream = _current_stream_handle


def _cuda_driver_class() -> type[DriverBase]:
    from triton.backends.nvidia.driver import (  # noqa: PLC0415 -- triton is optional
        CudaDriver,
        CudaLauncher,
        CudaUtils,
    )

    class MojoCudaDriver(CudaDriver):
        def __init__(self):
            # not the base constructors: they bind torch.cuda
            self.utils = CudaUtils()
            self.launcher_cls = CudaLauncher
            self.get_device_capability = _device_capability
            _bind_mojo(self)

        def get_active_torch_device(self) -> torch.device:
            return torch.device("mojo", self.get_current_device())

        def get_device_interface(self) -> type[_DeviceInterface]:
            return _DeviceInterface

        @staticmethod
        def is_active() -> bool:
            return accelerator_api() == "cuda"

        def get_empty_cache_for_benchmark(self) -> torch.Tensor:
            return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="mojo")

    return MojoCudaDriver


def _hip_driver_class() -> type[DriverBase]:
    """The AMD counterpart; its target comes from the HIP driver API
    (`utils.get_device_properties`), so nothing else changes. Untested here:
    written from Triton's AMD driver, no AMD GPU was available."""
    from triton.backends.amd.driver import (  # noqa: PLC0415 -- triton is optional
        HIPDriver,
        HIPLauncher,
        HIPUtils,
    )

    class MojoHipDriver(HIPDriver):
        def __init__(self):
            self.utils = HIPUtils()
            self.launcher_cls = HIPLauncher
            _bind_mojo(self)

        def get_active_torch_device(self) -> torch.device:
            return torch.device("mojo", self.get_current_device())

        def get_device_interface(self) -> type[_DeviceInterface]:
            return _DeviceInterface

        @staticmethod
        def is_active() -> bool:
            return accelerator_api() == "hip"

        def get_empty_cache_for_benchmark(self) -> torch.Tensor:
            return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="mojo")

    return MojoHipDriver


@functools.cache
def driver_class() -> type[DriverBase]:
    """The mojo Triton driver class for the vendor MAX drives. Cached: it is
    also the class of the registered Triton backend
    (`monkeypatching.register_the_mojo_triton_target`), which Triton compares
    the active driver against with `isinstance`."""
    api = accelerator_api()
    if api == "cuda":
        return _cuda_driver_class()
    if api == "hip":
        return _hip_driver_class()
    raise RuntimeError(
        f"Triton has no backend for the mojo device's {api!r} accelerator"
    )


def make_driver() -> DriverBase:
    return driver_class()()


def enable_triton():
    """Make Triton launch on the mojo device (idempotent)."""
    from triton.runtime import driver  # noqa: PLC0415 -- triton is optional

    driver.set_active(make_driver())


class _AfterTritonDriverImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Installs the mojo driver the moment `triton.runtime.driver` finishes
    importing: a finder that hands back the module's real spec with a loader
    wrapper, so importing triton stays as lazy as the user's code makes it."""

    def find_spec(
        self, name: str, path: object, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if name != _DRIVER_MODULE:
            return None
        sys.meta_path.remove(self)  # one shot; the real finders take over below
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        self._loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType):
        self._loader.exec_module(module)
        if accelerator_api() in ("cuda", "hip"):  # never break an unrelated import
            module.driver.set_active(make_driver())


def install_triton_hook():
    """Called by register_mojo_devices(): drive Triton from the mojo device
    once its runtime exists, now or on import."""
    if os.environ.get("TORCH_MOJO_BACKEND_TRITON", "1") == "0":
        return
    if importlib.util.find_spec("triton") is None:
        return
    if _DRIVER_MODULE in sys.modules:
        if accelerator_api() in ("cuda", "hip"):
            enable_triton()
    elif not any(isinstance(f, _AfterTritonDriverImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterTritonDriverImport())
