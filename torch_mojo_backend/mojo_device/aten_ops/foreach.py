"""The `_foreach_*` family.

Every op here has a batched fast path and falls back to ATen's own
sequential implementation, redispatched below this PrivateUse1 registration.
"""

from collections.abc import Callable

import torch

from .support import _COMPOSITE_EXPLICIT_AUTOGRAD, _fast


def inplace_dispatcher(op_name: str, fast_name: str) -> Callable:
    """Dispatcher for a mutable ()-returning foreach op: batched fast path,
    with ATen's exact sequential semantics as the fallback (redispatched
    below this PrivateUse1 registration, like `_foreach_mul_.Tensor`)."""
    packet_name, _, overload_name = op_name.removeprefix("aten::").partition(".")
    aten_op = getattr(getattr(torch.ops.aten, packet_name), overload_name or "default")

    def dispatcher(self, *args, **kwargs):
        aten_fast = _fast()
        result = getattr(aten_fast, fast_name)(self, *args, **kwargs)
        if result is aten_fast.NOT_HANDLED:
            result = aten_op.redispatch(
                _COMPOSITE_EXPLICIT_AUTOGRAD, self, *args, **kwargs
            )
            # This explicit redispatch runs below ADInplaceOrView, so the
            # TensorList version update is manual on both paths (mutable
            # TensorList schemas returning () get no automatic bump).
            torch.autograd.graph.increment_version(self)
            return result
        torch.autograd.graph.increment_version(self)
        return None

    return dispatcher


def mojo_device__foreach_mul__tensor(self, other):
    aten_fast = _fast()
    result = aten_fast.fast_aten__foreach_mul__tensor(self, other)
    if result is aten_fast.NOT_HANDLED:
        result = torch.ops.aten._foreach_mul_.Tensor.redispatch(
            _COMPOSITE_EXPLICIT_AUTOGRAD, self, other
        )
        # This explicit redispatch runs below ADInplaceOrView. A true wrapper
        # subclass therefore needs the same manual TensorList version update
        # as the direct Mojo kernel path.
        torch.autograd.graph.increment_version(self)
        return result

    # Mutable TensorList schemas returning () do not receive an automatic
    # version bump. Match CUDA, including empty and duplicate list entries.
    torch.autograd.graph.increment_version(self)
    return None


def mojo_device__foreach_norm_scalar(self, ord=2, dtype=None):
    aten_fast = _fast()
    result = aten_fast.fast_aten__foreach_norm(self, ord, dtype=dtype)
    if result is aten_fast.NOT_HANDLED:
        result = aten_fast.foreach_norm_sequential_fallback(self, ord, dtype=dtype)
        if result is aten_fast.NOT_HANDLED:
            return torch.ops.aten._foreach_norm.Scalar.redispatch(
                _COMPOSITE_EXPLICIT_AUTOGRAD, self, ord, dtype=dtype
            )
    return result


def mojo_device__foreach_sqrt(self):
    aten_fast = _fast()
    result = aten_fast.fast_aten__foreach_sqrt(self)
    if result is aten_fast.NOT_HANDLED:
        return torch.ops.aten._foreach_sqrt.default.redispatch(
            _COMPOSITE_EXPLICIT_AUTOGRAD, self
        )
    return result
