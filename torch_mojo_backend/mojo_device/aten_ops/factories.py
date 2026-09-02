"""Tensor factories: empty/new_*/full/ones/zeros/scalar_tensor/arange."""

import math
from collections.abc import Sequence

import max.driver as max_driver
import torch
from max.experimental.torch.torch import torch_dtype_to_max

from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    TorchMojoTensor,
    _resize_payload,
    find_equivalent_max_device,
)

from torch_mojo_backend.mojo_device.aten_ops.support import (
    _copy_into_tensor,
    _fast,
    _unsupported,
    max_dtype_to_torch_dtype,
)


def _require_device(device: torch.device | None) -> torch.device:
    """The dispatcher always resolves a factory op's device before calling
    into this PrivateUse1 registration; `None` here would already be a
    crash inside `find_equivalent_max_device`, just with a worse message."""
    assert device is not None, "expected a resolved mojo device from the dispatcher"
    return device


def mojo_device_empty_memory_format(
    size: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> TorchMojoTensor:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMojoTensor._alloc(
        tuple(size),
        torch_dtype_to_max(dtype),
        find_equivalent_max_device(_require_device(device)),
    )


def empty_strided(
    size: Sequence[int],
    stride: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    # The requested strides are ignored: allocation is always contiguous
    # (matching the previous behavior; our metadata is self-consistent).
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMojoTensor._alloc(
        tuple(size),
        torch_dtype_to_max(dtype),
        find_equivalent_max_device(_require_device(device)),
    )


def mojo_device_empty_permuted(
    size: Sequence[int],
    physical_layout: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    # Uninitialized memory: a contiguous allocation of `size` is valid.
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMojoTensor._alloc(
        tuple(size),
        torch_dtype_to_max(dtype),
        find_equivalent_max_device(_require_device(device)),
    )


def mojo_device_empty_like(
    self: TorchMojoTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    mojo_device = self._device if device is None else find_equivalent_max_device(device)
    return TorchMojoTensor._alloc(self._shape, max_dtype, mojo_device)


def _new_factory_device(
    self: TorchMojoTensor, device: torch.device | None
) -> max_driver.Device:
    """Target MAX device for a `new_*` factory. torch passes `self`'s device
    (whose torch-side index is the phantom 0) when the caller doesn't
    override it, so default to `self`'s real MAX device; only an explicit
    CPU request is honored differently."""
    if device is None:
        return self._device
    torch_dev = torch.device(device) if not isinstance(device, torch.device) else device
    if torch_dev.type == "cpu":
        return find_equivalent_max_device(torch_dev)
    return self._device


def mojo_device_new_empty(
    self: TorchMojoTensor,
    size: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    return TorchMojoTensor._alloc(
        tuple(size), max_dtype, _new_factory_device(self, device)
    )


def mojo_device_new_zeros(
    self: TorchMojoTensor,
    size: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    result = _fast().fast_filled(size, 0, max_dtype, _new_factory_device(self, device))
    if result is None:
        raise _unsupported("aten::new_zeros", (self,))
    return result


def mojo_device_new_ones(
    self: TorchMojoTensor,
    size: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    result = _fast().fast_filled(size, 1, max_dtype, _new_factory_device(self, device))
    if result is None:
        raise _unsupported("aten::new_ones", (self,))
    return result


def mojo_device_new_full(
    self: TorchMojoTensor,
    size: Sequence[int],
    fill_value: bool | int | float,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    result = _fast().fast_filled(
        size, fill_value, max_dtype, _new_factory_device(self, device)
    )
    if result is None:
        raise _unsupported("aten::new_full", (self,))
    return result


def _fast_filled_tensor(
    size: Sequence[int],
    value: bool | int | float,
    dtype: torch.dtype,
    device: torch.device | None,
) -> TorchMojoTensor:
    """Filled-tensor factory (alloc + Fill), or raises."""
    try:
        max_dtype = torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        raise _unsupported("aten::full (dtype)", (dtype,)) from None
    result = _fast().fast_filled(
        size, value, max_dtype, find_equivalent_max_device(_require_device(device))
    )
    if result is None:
        raise _unsupported("aten::full", (size, value, dtype))
    return result


def mojo_device_full(
    size: Sequence[int],
    fill_value: bool | int | float,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    if dtype is not None:
        resolved = dtype
    elif isinstance(fill_value, bool):
        resolved = torch.bool
    elif isinstance(fill_value, int):
        resolved = torch.int64
    else:
        resolved = torch.get_default_dtype()
    return _fast_filled_tensor(size, fill_value, resolved, device)


def mojo_device_full_like(
    self: TorchMojoTensor,
    fill_value: bool | int | float,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    mojo_device = self._device if device is None else find_equivalent_max_device(device)
    result = _fast().fast_filled(self._shape, fill_value, max_dtype, mojo_device)
    if result is None:
        raise _unsupported("aten::full_like", (self, fill_value))
    return result


def mojo_device_ones(
    size: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    return _fast_filled_tensor(size, 1, resolved, device)


def mojo_device_ones_like(
    self: TorchMojoTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> TorchMojoTensor:
    return mojo_device_full_like(self, 1, dtype=dtype, device=device)


def mojo_device_zeros(
    size: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    return _fast_filled_tensor(size, 0, resolved, device)


def mojo_device_zeros_like(
    self: TorchMojoTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    memory_format: torch.memory_format | None = None,
) -> TorchMojoTensor:
    return mojo_device_full_like(self, 0, dtype=dtype, device=device)


def mojo_device_scalar_tensor(
    s: bool | int | float,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    resolved = torch.float32 if dtype is None else dtype
    return _fast_filled_tensor((), s, resolved, device)


def _host_arange_tensor(
    start: bool | int | float,
    end: bool | int | float,
    step: bool | int | float,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """torch.arange built on the host (exact torch semantics)."""
    return torch.arange(start, end, step, dtype=dtype)


def _device_arange(
    start: bool | int | float,
    end: bool | int | float,
    step: bool | int | float,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> TorchMojoTensor | None:
    """torch.arange computed by a device kernel, or None to use the host
    path. HF generation loops call torch.arange(..., device=...) every
    step; the host path costs a blocking H2D copy (full queue drain) per
    call, so the common numeric cases run on device instead."""
    for v in (start, end, step):
        # bool is an int subclass; torch treats it as 0/1 here.
        if not isinstance(v, int | float):
            return None
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if abs(v) > _MAX_EXACT_F64_INT:
            # Python inputs cross the kernel boundary as Float64.
            return None
    if step == 0:
        return None  # host path raises torch's own error
    if dtype is None:
        all_int = all(isinstance(v, int) for v in (start, end, step))
        dtype = torch.int64 if all_int else torch.get_default_dtype()
    try:
        max_dtype = torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        return None
    if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
        numel = max(0, -(-(end - start) // step))
    else:
        numel = max(0, math.ceil((float(end) - float(start)) / float(step)))
    return _fast().fast_arange(
        numel,
        start,
        step,
        max_dtype,
        find_equivalent_max_device(_require_device(device)),
    )


# fast_arange receives start/step as Float64; its accumulator follows the
# output dtype and device, matching PyTorch's range kernels.
_MAX_EXACT_F64_INT = 2**53


def mojo_device_arange(
    start: bool | int | float,
    end: bool | int | float | None = None,
    step: bool | int | float = 1,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMojoTensor:
    if end is None:
        start, end = 0, start
    result = _device_arange(start, end, step, dtype, device)
    if result is not None:
        return result
    # Build on the host with exact torch semantics, then one H2D copy.
    cpu = _host_arange_tensor(start, end, step, dtype)
    return TorchMojoTensor._from_cpu(
        cpu, find_equivalent_max_device(_require_device(device))
    )


def mojo_device_arange_start_out(
    start: bool | int | float,
    end: bool | int | float,
    step: bool | int | float = 1,
    *,
    out: TorchMojoTensor,
) -> TorchMojoTensor:
    # torch.arange(start, end, step, device=...) dispatches to the out
    # variant with a pre-allocated `out` of the right size and dtype.
    torch_dtype = max_dtype_to_torch_dtype(out._dtype)
    staged = _device_arange(start, end, step, torch_dtype, out.device)
    if staged is None:
        cpu = _host_arange_tensor(start, end, step, torch_dtype)
        staged = TorchMojoTensor._from_cpu(cpu, out._device)
    if tuple(staged._shape) == tuple(out._shape):
        _copy_into_tensor(out, staged)
    else:
        _resize_payload(out, staged._shape)
        _copy_into_tensor(out, staged)
    return out
