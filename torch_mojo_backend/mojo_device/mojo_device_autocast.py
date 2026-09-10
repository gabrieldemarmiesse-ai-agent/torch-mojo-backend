"""AutocastPrivateUse1 policies for eager Mojo tensors.

PyTorch lets a private backend advertise supported AMP dtypes, but it does not
install any operator policies for that backend. These wrappers mirror the CUDA
policies needed by nanoGPT: matmul/attention run in the active lower precision,
normalization and NLL run in FP32, and the reductions CUDA lists under
`fp32_set_opt_dtype` (softmax, log_softmax, sum, prod) produce FP32 when the
caller left `dtype` unspecified. (`F.cross_entropy` is not such a caller: it
passes the input's own dtype to log_softmax, so that stage stays bf16 under
autocast on CUDA and here alike.) Unlisted operations fall through.
"""

import functools
from collections.abc import Callable, Mapping

import torch

_registered = False
_fallback_library: torch.library.Library | None = None
_aten_library: torch.library.Library | None = None

# torch/_C/__init__.pyi's DispatchKey stub lists PrivateUse1, AutogradPrivateUse1
# etc. but omits AutocastPrivateUse1, even though it exists at runtime.
_AUTOCAST_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.AutocastPrivateUse1  # ty: ignore[unresolved-attribute]
)


def _cast_mojo_floating(value: object, dtype: torch.dtype) -> object:
    """Recursively cast eligible Mojo floating tensors, leaving metadata alone."""
    if isinstance(value, torch.Tensor):
        if (
            value.device.type == "mojo"
            and value.is_floating_point()
            and value.dtype != torch.float64
            and value.dtype != dtype
        ):
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_cast_mojo_floating(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_mojo_floating(item, dtype) for item in value]
    if isinstance(value, Mapping):
        return {key: _cast_mojo_floating(item, dtype) for key, item in value.items()}
    return value


def _policy_wrapper(
    op: torch._ops.OpOverload, dtype_getter: Callable[[], torch.dtype]
) -> Callable[..., object]:
    @functools.wraps(op)
    def wrapper(*args: object, **kwargs: object) -> object:
        # Casts themselves must redispatch below AutocastPrivateUse1, otherwise
        # their internal _to_copy calls would re-enter this policy layer.
        with torch._C._ExcludeDispatchKeyGuard(_AUTOCAST_KEYSET):
            dtype = dtype_getter()
            cast_args = _cast_mojo_floating(args, dtype)
            cast_kwargs = _cast_mojo_floating(kwargs, dtype)
            assert isinstance(cast_args, tuple)
            assert isinstance(cast_kwargs, dict)
            return op(*cast_args, **cast_kwargs)

    return wrapper


def _lower_precision_wrapper(op: torch._ops.OpOverload) -> Callable[..., object]:
    return _policy_wrapper(op, lambda: torch.get_autocast_dtype("mojo"))


def _fp32_wrapper(op: torch._ops.OpOverload) -> Callable[..., object]:
    return _policy_wrapper(op, lambda: torch.float32)


def _is_eligible(value: object) -> bool:
    """CUDA's `firstarg_is_eligible`: a floating, non-double tensor on the device."""
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "mojo"
        and value.is_floating_point()
        and value.dtype != torch.float64
    )


def _set_opt_dtype_wrapper(op: torch._ops.OpOverload) -> Callable[..., object]:
    """CUDA's `fp32_set_opt_dtype` policy: run in FP32 unless the caller chose a
    dtype (`aten/src/ATen/autocast_mode.h`). Implemented the way ATen
    implements a `dtype` argument on these ops, by casting the input first,
    which keeps the eager fast paths on their dtype-less signatures."""
    dtype_index = [arg.name for arg in op._schema.arguments].index("dtype")

    @functools.wraps(op)
    def wrapper(*args: object, **kwargs: object) -> object:
        with torch._C._ExcludeDispatchKeyGuard(_AUTOCAST_KEYSET):
            given = (
                args[dtype_index] if len(args) > dtype_index else kwargs.get("dtype")
            )
            first = args[0] if args else kwargs.get("self")
            if (
                given is None
                and isinstance(first, torch.Tensor)
                and _is_eligible(first)
            ):
                upcast = first.to(dtype=torch.float32)
                if args:
                    args = (upcast, *args[1:])
                else:
                    kwargs = {**kwargs, "self": upcast}
            return op(*args, **kwargs)

    return wrapper


def register_autocast_ops():
    """Install fallthrough plus the explicit CUDA-matching GPT policies."""
    global _registered, _fallback_library, _aten_library
    if _registered:
        return

    _fallback_library = torch.library.Library("_", "IMPL", "AutocastPrivateUse1")
    _fallback_library.fallback(torch.library.fallthrough_kernel)
    _aten_library = torch.library.Library("aten", "IMPL", "AutocastPrivateUse1")

    lower_precision_ops = (
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
    )
    fp32_ops = (
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.nll_loss.default,
        torch.ops.aten.nll_loss_forward.default,
    )
    fp32_set_opt_dtype_ops = (
        torch.ops.aten.log_softmax.int,
        torch.ops.aten.softmax.int,
        torch.ops.aten.sum.default,
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.prod.default,
        torch.ops.aten.prod.dim_int,
    )
    for op in lower_precision_ops:
        _aten_library.impl(op._schema.name, _lower_precision_wrapper(op))
    for op in fp32_ops:
        _aten_library.impl(op._schema.name, _fp32_wrapper(op))
    for op in fp32_set_opt_dtype_ops:
        # The default overload is addressed by the bare name; only named
        # overloads take the ".overload" suffix.
        name = op._schema.name
        if op._overloadname != "default":
            name = f"{name}.{op._overloadname}"
        _aten_library.impl(name, _set_opt_dtype_wrapper(op))

    _registered = True


__all__ = ["register_autocast_ops"]
