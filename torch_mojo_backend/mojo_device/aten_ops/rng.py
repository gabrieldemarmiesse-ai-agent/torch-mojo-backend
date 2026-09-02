"""Random-number ops."""

import torch

from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

from .support import _copy_into_tensor, _unsupported, max_dtype_to_torch_dtype


def mojo_device_normal_(
    self: TorchMojoTensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> TorchMojoTensor:
    if generator is not None:
        raise _unsupported("aten::normal_ (generator)", (self,))
    cpu = torch.empty(self._shape, dtype=max_dtype_to_torch_dtype(self._dtype)).normal_(
        mean, std
    )
    staged = TorchMojoTensor._from_cpu(cpu, self._device)
    _copy_into_tensor(self, staged)
    return self
