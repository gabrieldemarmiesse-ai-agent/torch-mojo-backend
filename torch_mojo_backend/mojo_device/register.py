"""Registration of the `mojo` device (see docs/native_backend.md)."""

import torch

from torch_mojo_backend import native
from torch_mojo_backend.mojo_device.hip_peer import warn_if_gpu_torch_on_hip
from torch_mojo_backend.monkeypatching import fix_privateuse1_dlpack_device_type
from torch_mojo_backend.native import device_module

_registered = False


def register_mojo_devices():
    """Enable the mojo device globally: build/load the two shims once and
    register the backend, the `torch.mojo` module and the `.mojo()` helpers.
    Idempotent."""
    global _registered
    if _registered:
        return
    # Module._apply otherwise replaces a shared CPU Parameter independently in
    # each child module; swapping preserves tied weights (GPT-2's token
    # embedding and lm_head) as one Parameter and one allocation.
    torch.__future__.set_swap_module_params_on_conversion(True)
    torch.utils.rename_privateuse1_backend("mojo")
    torch._register_device_module("mojo", device_module)
    torch.utils.generate_methods_for_privateuse1_backend()
    fix_privateuse1_dlpack_device_type()
    native.register()
    warn_if_gpu_torch_on_hip()
    _registered = True
