"""Data transfer: H2D / D2H / D2D and dtype/device copies."""

from typing import Protocol, cast, runtime_checkable

import max.driver
import torch
from max.experimental.torch.torch import torch_dtype_to_max

from torch_mojo_backend.mojo_device import cuda_peer, torch_mojo_device_module
from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    TorchMojoTensor,
    _copy_strided_into,
    _record_h2d_source,
    find_equivalent_max_device,
)

from torch_mojo_backend.mojo_device.aten_ops.support import (
    _copy_into_tensor,
    _fast,
    max_dtype_to_torch_dtype,
)


@runtime_checkable
class _TensorHolderModule(Protocol):
    """The Mojo `tensor_holder` extension's Python-callable surface this
    module needs -- it's a compiled-on-first-use native module (no stubs
    possible), reached via `eager_kernels`' PEP 562 `__getattr__`."""

    def copy_d2d(
        self, ctx_ptr: int, dst_ptr: int, src_ptr: int, nbytes: int
    ) -> None: ...
    def copy_from_host(
        self, ctx_ptr: int, dev_ptr: int, host_ptr: int, nbytes: int
    ) -> object: ...


def _upload_on_device(
    src: torch.Tensor, dest_ptr: int, dest_device: max.driver.Device
) -> bool:
    """Copy a foreign CUDA tensor straight into `dest_ptr`, or return False.

    Both allocations are device memory in one process, so when they sit on the
    same physical GPU the bytes never need to visit the host -- 0.75 ms rather
    than 386 ms for 537 MB on an H100 PCIe.  `cuda_peer` decides sameness from
    the pointers rather than from device ordinals, so this stays correct on a
    multi-GPU box where the two runtimes may not enumerate alike.

    Three synchronizations are load-bearing and none is optional:

    * the queue drain, because a pending queued write to the destination would
      otherwise land AFTER this copy and silently overwrite it;
    * the CUDA synchronize, because this runtime knows nothing about torch's
      stream and would otherwise read bytes CUDA has not written yet;
    * the mojo synchronize, because torch's caching allocator may hand the
      source block to another tensor the moment this call returns, while the
      copy is still in flight.
    """
    if src.device.type != "cuda" or not src.is_contiguous():
        return False
    # The mojo side must be NVIDIA too: `api` is "cuda" only there, and the
    # whole route speaks the CUDA driver API.  Anything else falls through to
    # the host bounce, which is correct everywhere and merely slower.
    if getattr(dest_device, "api", None) != "cuda":
        return False
    nbytes = src.numel() * src.element_size()
    if nbytes == 0:
        return True
    if not cuda_peer.same_physical_device(src.data_ptr(), dest_ptr):
        return False
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.mojo_device import deferred_compile

    deferred_compile.drain()
    torch.cuda.synchronize()
    holder = cast(_TensorHolderModule, eager_kernels.tensor_holder)
    holder.copy_d2d(
        eager_kernels._ctx_ptr(dest_device), dest_ptr, src.data_ptr(), nbytes
    )
    torch_mojo_device_module.synchronize()
    return True


def mojo_device__copy_from(
    self: torch.Tensor, dest: torch.Tensor, non_blocking: bool = False
) -> torch.Tensor:
    src_is_mojo = isinstance(self, TorchMojoTensor)
    dest_is_mojo = isinstance(dest, TorchMojoTensor)

    if src_is_mojo and dest_is_mojo:
        if self._device != dest._device:
            # Cross mojo-device: bounce through the host.
            bounced = TorchMojoTensor._from_cpu(
                self._to_cpu_tensor(), dest._device, non_blocking=non_blocking
            )
            _copy_into_tensor(dest, bounced)
            return dest
        _copy_into_tensor(dest, self)
        return dest

    if src_is_mojo and not dest_is_mojo:
        # D2H; dest is a CPU torch tensor (copy_ handles cast/layout).
        dest.copy_(self._to_cpu_tensor())
        return dest

    if not src_is_mojo and dest_is_mojo:
        # H2D. Resolve dtype/broadcast on the host, then one upload.
        cpu = self.detach()
        torch_dtype = max_dtype_to_torch_dtype(dest._dtype)
        if cpu.dtype != torch_dtype:
            cpu = cpu.to(torch_dtype)
        if cpu.device.type != "cpu":
            # Same trap as _to_copy: the source only has to be non-mojo, and
            # copy_from_host below reads data_ptr() as host memory.  Try the
            # on-device route first; otherwise bounce through the host, after
            # the cast and before the broadcast materializes anything.
            if tuple(cpu.shape) == tuple(dest._shape) and dest._is_contiguous:
                if _upload_on_device(cpu.contiguous(), dest._ptr, dest._device):
                    return dest
            cpu = cpu.cpu()
        if tuple(cpu.shape) != tuple(dest._shape):
            cpu = cpu.broadcast_to(dest._shape)
        cpu = cpu.contiguous()
        if dest._is_contiguous:
            if dest._numel > 0:
                from torch_mojo_backend import eager_kernels

                holder = cast(_TensorHolderModule, eager_kernels.tensor_holder)
                transfer_owner = holder.copy_from_host(
                    eager_kernels._ctx_ptr(dest._device),
                    dest._ptr,
                    cpu.data_ptr(),
                    dest._numel * dest._itemsize,
                )
                _record_h2d_source(dest._device, transfer_owner, non_blocking)
        else:
            staged = TorchMojoTensor._from_cpu(
                cpu, dest._device, non_blocking=non_blocking
            )
            _copy_strided_into(dest, staged)
        return dest

    raise RuntimeError(
        f"invalid _copy_from configuration: {type(self)} -> {type(dest)}"
    )


def mojo_device__to_copy(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> torch.Tensor:
    aten_fast = _fast()
    if not isinstance(tensor, TorchMojoTensor):
        # Any non-mojo tensor moving onto mojo_device (optionally casting
        # first).  "Not mojo" is not "on the host": with a GPU torch installed
        # this branch also takes `x.cuda().to("mojo")`, and _from_cpu passes
        # data_ptr() to alloc_from_host, which reads it as HOST memory -- so a
        # CUDA pointer segfaulted the process instead of raising.  Cast before
        # the bounce: on a downcast that shrinks the transfer.
        assert device is not None, "moving onto mojo_device always names a device"
        t = tensor.detach()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if t.device.type != "cpu":
            max_device = find_equivalent_max_device(device)
            staged = TorchMojoTensor._alloc(
                tuple(t.shape), torch_dtype_to_max(t.dtype), max_device
            )
            if _upload_on_device(t.contiguous(), staged._ptr, staged._device):
                return staged
            t = t.cpu()
        return TorchMojoTensor._from_cpu(
            t, find_equivalent_max_device(device), non_blocking=non_blocking
        )

    result = tensor
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        if max_dtype != result._dtype:
            if (
                result._dtype in aten_fast._CAST_DTYPES
                and max_dtype in aten_fast._CAST_DTYPES
            ):
                result = aten_fast._cast_tensor(result, max_dtype)
            else:
                # Exotic dtype pair: cast on the host.
                cpu = result._to_cpu_tensor().to(dtype)
                if device is not None and device.type == "cpu":
                    return cpu
                target = (
                    find_equivalent_max_device(device)
                    if device is not None
                    else result._device
                )
                return TorchMojoTensor._from_cpu(cpu, target, non_blocking=non_blocking)
    if device is not None and device.type == "cpu":
        return result._to_cpu_tensor(non_blocking=non_blocking)
    if device is not None:
        target = find_equivalent_max_device(device)
        if target != result._device:
            return TorchMojoTensor._from_cpu(
                result._to_cpu_tensor(), target, non_blocking=non_blocking
            )
    if result is tensor:
        # _to_copy always returns a fresh tensor.
        result = tensor._materialize_contiguous()
    return result
