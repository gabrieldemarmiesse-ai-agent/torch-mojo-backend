"""BLAS-shaped ops with a custom registration."""

import torch

from torch_mojo_backend.mojo_device.aten_ops.support import (
    _COMPOSITE_EXPLICIT_AUTOGRAD,
    _fast,
)


def mojo_device_addr(
    self: torch.Tensor,
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    beta: int | float = 1,
    alpha: int | float = 1,
) -> torch.Tensor:
    """Fused beta*self + alpha*outer(vec1, vec2) in one kernel launch that
    reproduces CPU's own addr_kernel op order and per-op rounding exactly
    (see `fast_aten_addr`'s docstring for why: absent this, ATen's own
    CompositeExplicitAutograd fallback for backends without a native addr
    kernel composes outer/scale/add in a different multiplication order
    than the native CPU/CUDA kernel, which drifts enough to fail OpInfo
    conformance for fp16/bf16). Anything the fast path declines (self not
    broadcastable to exactly (len(vec1), len(vec2)), mixed dtypes, ...)
    redispatches to that same composite fallback, unchanged."""
    aten_fast = _fast()
    result = aten_fast.fast_aten_addr(self, vec1, vec2, beta, alpha)
    if result is aten_fast.NOT_HANDLED:
        return torch.ops.aten.addr.default.redispatch(
            _COMPOSITE_EXPLICIT_AUTOGRAD, self, vec1, vec2, beta=beta, alpha=alpha
        )
    return result
