"""In-place ops with custom plumbing."""

from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

from .support import (
    _copy_into_tensor,
    _fast,
    _refuse_unsupported_backward,
    _unsupported,
)


def mojo_device_add_(
    self: TorchMojoTensor, other, alpha: int | float = 1.0
) -> TorchMojoTensor:
    # `alpha` is a Scalar in the schema, not a float: unlike the `float`
    # arguments of, say, native_batch_norm, PyTorch's argument parser does not
    # coerce it, so `x.add_(y, alpha=-2)` arrives here as an int.
    result = _fast().fast_aten_add_(self, other, alpha)
    if result is None:
        raise _unsupported("aten::add_.Tensor", (self, other))
    return result


def mojo_device_fill__scalar(self: TorchMojoTensor, value) -> TorchMojoTensor:
    result = _fast().fast_aten_fill__scalar(self, value)
    if result is None:
        raise _unsupported("aten::fill_.Scalar", (self, value))
    return result


def mojo_device_masked_fill_(
    self: TorchMojoTensor, mask: TorchMojoTensor, value
) -> TorchMojoTensor:
    result = _fast().fast_aten_masked_fill_(self, mask, value)
    if result is None:
        raise _unsupported("aten::masked_fill_", (self, mask, value))
    return result


def mojo_device_mul_(self: TorchMojoTensor, other) -> TorchMojoTensor:
    result = _fast().fast_aten_mul_(self, other)
    if result is None:
        raise _unsupported("aten::mul_.Tensor", (self, other))
    return result


def mojo_device_relu_(self: TorchMojoTensor) -> TorchMojoTensor:
    # `nn.ReLU(inplace=True)` records the same ReluBackward0 as the functional
    # form, and `aten::threshold_backward` has no kernel here, so the refusal
    # has to happen in the forward for the reason
    # `_refuse_unsupported_backward` documents. A leaf would already have been
    # rejected by ADInplaceOrView; a non-leaf reaches this.
    _refuse_unsupported_backward(
        "aten::relu_",
        "aten::threshold_backward",
        (self,),
        "The forward itself is supported: run it under torch.no_grad() or "
        "torch.inference_mode().",
    )
    aten_fast = _fast()
    result = aten_fast.fast_aten_relu(self)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::relu_", (self,))
    _copy_into_tensor(self, result)
    return self


def mojo_device_zero_(self: TorchMojoTensor) -> TorchMojoTensor:
    return mojo_device_fill__scalar(self, 0)
