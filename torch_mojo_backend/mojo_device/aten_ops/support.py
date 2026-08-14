"""Shared plumbing for the eager aten implementations in this package.

Everything the implementation modules have in common lives here — the lazily
imported `aten_fast` handle, the actionable `NotImplementedError`, the
strided/broadcasting copy that every `out=` variant ends with — so no
implementation module ever has to import another one.
"""

from collections.abc import Callable

import torch

from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    TorchMojoTensor,
    _resize_payload,
)

# Unsupported foreach regimes must reach ATen's exact sequential semantics
# without redispatching to this PrivateUse1 registration again.
_COMPOSITE_EXPLICIT_AUTOGRAD = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


_aten_fast_module = None


def _fast():
    """The aten_fast module.

    Imported lazily: the first import triggers the (cached) Mojo kernel
    compilation, which pure torch.compile workloads should never pay for.
    """
    global _aten_fast_module
    if _aten_fast_module is None:
        from torch_mojo_backend.eager_kernels import aten_fast

        _aten_fast_module = aten_fast
    return _aten_fast_module


def max_dtype_to_torch_dtype(dtype):
    from max.experimental.torch import max_dtype_to_torch

    return max_dtype_to_torch(dtype)


def _describe_args(args, kwargs) -> str:
    descs = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, TorchMojoTensor):
            descs.append(f"{tuple(a._shape)}:{a._dtype}")
        elif isinstance(a, torch.Tensor):
            descs.append(f"{tuple(a.shape)}:{a.dtype}:{a.device}")
    return ", ".join(descs) or "none"


def _unsupported(op_name: str, args=(), kwargs=None) -> NotImplementedError:
    return NotImplementedError(
        f"{op_name} is not supported by mojo eager mode for these inputs "
        f"(tensor args: {_describe_args(args, kwargs or {})}). The graph "
        "fallback was removed; add a fast kernel in "
        "torch_mojo_backend/eager_kernels/ or open an issue."
    )


def _eager_impl(fast_name: str, op_name: str) -> Callable:
    """Bind an op to its aten_fast implementation; raise on NOT_HANDLED."""
    fast_fn: Callable | None = None
    not_handled = None

    def dispatcher(*args, **kwargs):
        nonlocal fast_fn, not_handled
        if fast_fn is None:
            aten_fast = _fast()
            fast_fn = getattr(aten_fast, fast_name)
            not_handled = aten_fast.NOT_HANDLED
        result = fast_fn(*args, **kwargs)
        if result is not_handled:
            raise _unsupported(op_name, args, kwargs)
        return result

    return dispatcher


def _not_implemented(op_name: str) -> Callable:
    """Explicit raiser for ops that used to run through the graph fallback
    and have no fast implementation yet. Registered (instead of left
    unregistered) so users get an actionable message, and so the remaining
    surface is greppable."""

    def raiser(*args, **kwargs):
        raise _unsupported(op_name, args, kwargs)

    return raiser


def _copy_into_tensor(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """dst[...] = src[...] with dtype cast + broadcast, any strides."""
    aten_fast = _fast()
    if src._dtype != dst._dtype:
        src = aten_fast._cast_tensor(src, dst._dtype)
    if tuple(src._shape) != tuple(dst._shape):
        expanded = aten_fast.fast_aten_expand(src, dst._shape)
        if expanded is aten_fast.NOT_HANDLED:
            raise _unsupported("aten::copy_ (broadcast)", (dst, src))
        src = expanded
    aten_fast._copy_into(dst, src)


def _out_variant(op_name: str, fast_name: str, *, dtype_policy: str = "safe_cast"):
    """Wrap a functional fast implementation as an out= variant: compute,
    then copy into `out` (strided-safe)."""

    def dispatcher(*args, out: TorchMojoTensor, **kwargs):
        if not isinstance(out, TorchMojoTensor):
            raise RuntimeError(f"{op_name}: expected out to be a mojo tensor")

        # Reject a cross-device destination before launching the functional
        # composition.  Fast implementations already require all tensor
        # operands to share a device, so the first mojo operand identifies
        # the only valid output context.
        input_device = next(
            (arg._device for arg in args if isinstance(arg, TorchMojoTensor)), None
        )
        if input_device is not None and out._device != input_device:
            raise RuntimeError(
                f"{op_name}: expected out and input tensors to be on the same device"
            )

        aten_fast = _fast()
        result = getattr(aten_fast, fast_name)(*args, **kwargs)
        if result is aten_fast.NOT_HANDLED:
            raise _unsupported(op_name, args, kwargs)

        if out._device != result._device:
            raise RuntimeError(
                f"{op_name}: expected out and result tensors to be on the same device"
            )
        result_dtype = max_dtype_to_torch_dtype(result._dtype)
        out_dtype = max_dtype_to_torch_dtype(out._dtype)
        if dtype_policy == "exact":
            valid_dtype = result_dtype == out_dtype
        elif dtype_policy == "bool_or_uint8":
            valid_dtype = out_dtype in (torch.bool, torch.uint8)
        elif dtype_policy == "safe_cast":
            valid_dtype = torch.can_cast(result_dtype, out_dtype)
        else:
            raise AssertionError(f"unknown out dtype policy: {dtype_policy}")
        if not valid_dtype:
            raise RuntimeError(
                f"result type {result_dtype} can't be cast to the desired "
                f"output type {out_dtype}"
            )

        if tuple(result._shape) == tuple(out._shape):
            _copy_into_tensor(out, result)
        else:
            _resize_payload(out, result._shape)
            _copy_into_tensor(out, result)
        return out

    return dispatcher
