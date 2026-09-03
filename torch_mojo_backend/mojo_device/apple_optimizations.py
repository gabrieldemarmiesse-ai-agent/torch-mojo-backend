from torch_mojo_backend.monkeypatching import (
    use_apple_fast_add as _enable_apple_fast_add,
)
from torch_mojo_backend.torch_compile_backend.utils import get_accelerators


def register_apple_optimizations():
    """Install optional integrations that are profitable only on Apple GPUs."""
    if any(device.api == "metal" for device in get_accelerators()):
        _enable_apple_fast_add()
