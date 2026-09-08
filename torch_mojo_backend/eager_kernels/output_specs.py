"""Output metadata inferred in Python.

A descriptor whose Mojo entry point writes into a caller-allocated output
infers that output's shape/dtype/device without loading (let alone compiling)
the module; Python allocates it and hands it to the extension call.

This lives in ``eager_kernels`` rather than in ``aten_fast`` because both the
aten layer and ``mojo_device.torch_mojo_tensor`` (strided materialization) go
through it, and the tensor module cannot import the aten layer. Nothing here
imports ``aten_fast``.
"""

from dataclasses import dataclass
from typing import Protocol, cast

import torch
from max.driver import Device
from max.dtype import DType


@dataclass(frozen=True)
class _TensorOutputSpec:
    """Shape/type/device metadata inferred without loading a Mojo module."""

    shape: tuple[int, ...]
    dtype: DType
    device: Device


class _AllocFn(Protocol):
    def __call__(
        self, shape: tuple[int, ...], dtype: DType, device: Device
    ) -> torch.Tensor: ...


def _bootstrap_alloc(
    shape: tuple[int, ...], dtype: DType, device: Device
) -> torch.Tensor:
    """``TorchMojoTensor._alloc``, resolved on first use.

    ``mojo_device`` imports ``eager_kernels``, never the other way round, so
    the allocator cannot be imported at module scope. Rebinding the `_alloc`
    global on the first call keeps that direction intact and leaves the
    steady state at one plain attribute lookup (the same trick as
    ``_ctx_ptr``). Tensors are typed ``torch.Tensor`` here for the same
    import-direction reason. `_alloc` is a plain variable (not `def`-bound)
    so this rebind type-checks against its declared `_AllocFn` type.
    """
    global _alloc
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (  # noqa: PLC0415 -- the cycle this docstring describes
        TorchMojoTensor,
    )

    _alloc = cast(_AllocFn, TorchMojoTensor._alloc)
    return _alloc(shape, dtype, device)


_alloc: _AllocFn = _bootstrap_alloc


def _allocate_output_spec(spec: _TensorOutputSpec) -> torch.Tensor:
    return _alloc(spec.shape, spec.dtype, spec.device)
