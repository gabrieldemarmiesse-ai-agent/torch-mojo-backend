from torch_mojo_backend.torch_compile_backend.utils import get_accelerators


def _enable_apple_fast_add():
    # Lazy: aten_fast's first import compiles the eager kernels.
    from torch_mojo_backend.eager_kernels import aten_fast

    aten_fast.enable_apple_fast_add()


def register_apple_optimizations():
    """Install optional integrations that are profitable only on Apple GPUs."""
    if any(device.api == "metal" for device in get_accelerators()):
        _enable_apple_fast_add()
