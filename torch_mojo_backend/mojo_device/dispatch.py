"""Dispatch entry for the mojo device.

Every aten op intercepted by ``TorchMojoTensor.__torch_dispatch__`` lands
here and runs synchronously, kernel builds included: a specialization that is
not in ``__mojocache__`` yet is compiled inline at its first call. The op's
PrivateUse1 kernel is called out of ``DIRECT_IMPLS`` rather than through
``func(...)``, which would walk the C++ dispatcher back to that same callable.
"""

import threading
from collections.abc import Callable

import torch

# MAX's DeviceContext is not documented thread-safe: serialize every
# device-touching call (the forward runs on the main thread, the backward on
# autograd's engine thread). Re-entrant because an impl may run another torch
# op on a mojo tensor, which re-enters ``__torch_dispatch__``.
_DEVICE_LOCK = threading.RLock()


# Every op registered for PrivateUse1, keyed by its OpOverload and holding
# the exact callable `torch.library.impl` was handed. Filled at device
# registration (mojo_device/register.py); empty until then.
#
# `__torch_dispatch__` already unboxed this call's arguments, so redispatching
# through `func(*args, **kwargs)` re-enters the C++ dispatcher only to box and
# unbox them again for the very same Python callable -- ~8 us per dispatch,
# and an eager nanoGPT step makes ~760 of them (6 ms of a 37 ms host budget
# at batch 12). A dict hit calls that callable straight.
#
# The fallthrough below is what the table cannot cover: CompositeImplicit ops
# that decompose in C++, and anything with no PrivateUse1 registration.
#
# Consequence for implementations: with no `_DisableTorchDispatch` around
# them, a torch op an impl runs on a mojo tensor re-enters
# `__torch_dispatch__` instead of dropping straight to the backend kernel.
# That is correct, but an impl for op X must never call op X on a mojo
# tensor. Impls that need the backend kernel without the round trip use
# `.redispatch(<keyset>, ...)`, as the foreach and addr fallbacks do.
DIRECT_IMPLS: dict[torch._ops.OpOverload, Callable[..., object]] = {}


def dispatch(
    func: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """Entry point called from TorchMojoTensor.__torch_dispatch__."""
    impl = DIRECT_IMPLS.get(func)
    with _DEVICE_LOCK:
        if impl is not None:
            return impl(*args, **kwargs)
        with torch._C._DisableTorchDispatch():
            return func(*args, **kwargs)
