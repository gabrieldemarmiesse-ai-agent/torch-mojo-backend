"""min along one dim: functional (values, indices) + out= variant."""

from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    TorchMojoTensor,
    _resize_payload,
)

from .support import _copy_into_tensor, _fast, _unsupported


def mojo_device_min_dim(
    input: TorchMojoTensor, dim: int, keepdim: bool = False
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    """Functional torch.min(x, dim): (values, indices). Registered so torch
    doesn't synthesize it from the out= variant (which would allocate the
    outputs on the phantom index-0 device)."""
    aten_fast = _fast()
    result = aten_fast.fast_aten_min_dim(input, dim, keepdim)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::min.dim", (input,))
    return result


def mojo_device_min_dim_min(
    input: TorchMojoTensor,
    dim: int,
    keepdim: bool = False,
    min: TorchMojoTensor | None = None,
    min_indices: TorchMojoTensor | None = None,
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    """Out-variant of torch.min along a dim: writes values into `min` and
    int64 indices into `min_indices` (resizing via payload rebind when the
    pre-allocated shapes don't match, like the other out-variants)."""
    aten_fast = _fast()
    result = aten_fast.fast_aten_min_dim(input, dim, keepdim)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::min.dim_min", (input,))
    values, indices = result
    for dst, src in ((min, values), (min_indices, indices)):
        if dst is None:
            continue
        if tuple(dst._shape) == tuple(src._shape):
            _copy_into_tensor(dst, src)
        else:
            _resize_payload(dst, src._shape)
            _copy_into_tensor(dst, src)
    return (min, min_indices)
