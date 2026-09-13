"""Every monkeypatch this project applies, in one place.

A monkeypatch here means replacing or mutating, at runtime, an attribute of
a module or class we do not own -- mostly PyTorch internals that have no
extension point yet. Each patch is one function with a docstring saying what
upstream lacks, so that the patch can be turned into an upstream PR and
deleted from here. Nothing else in the package may patch a third-party
module; keep new patches in this file
(``tests/test_monkeypatching_is_centralized.py`` enforces that for
``torch``-rooted assignments).

Every patch installer is called by ``register.register_mojo_devices``, never
at import time. Official registration APIs (``torch.library.impl``, the
PrivateUse1 backend module, ``torch.__future__`` toggles) are not
monkeypatches and stay in ``mojo_device/register.py``.
"""

from functools import wraps

import torch
import torch.distributed.distributed_c10d as c10d


def fix_privateuse1_dlpack_device_type():
    """`Tensor.__dlpack_device__` doesn't recognize a *renamed* PrivateUse1
    backend.

    ``torch/_tensor.py``'s ``Tensor.__dlpack_device__`` maps a PrivateUse1
    tensor to DLPack's ``kDLExtDev`` by comparing ``self.device.type``
    against the string literal ``"privateuse1"`` -- so after
    ``torch.utils.rename_privateuse1_backend("mojo")`` it never matches, and
    every mojo tensor's ``__dlpack_device__()`` raises ``ValueError("Unknown
    device type mojo for Dlpack")``. Two other call sites in that very same
    file (the ``__cuda_array_interface__`` gate) correctly compare against
    ``torch._C._get_privateuse1_backend_name()`` instead of the literal;
    this one method just didn't get the memo.

    ``Tensor.__dlpack__`` itself (the capsule export) is unaffected --
    ATen's C++ DLConvertor keys off the ``DeviceType`` enum, not the
    Python-visible name -- so only the device-query half needs patching.
    ``torch_compile_backend/compiler.py``'s ``fast_from_dlpack`` routes
    around this bug for its own zero-copy exchange (it never calls
    ``__dlpack_device__``), but plain ``torch.utils.dlpack`` /
    ``max.driver.Buffer.from_dlpack(t)`` usage elsewhere (user code,
    ``test_compile_mojo_device.py``) goes through the single-arg DLPack
    protocol, which calls ``__dlpack_device__()`` first and needs this fix.
    """
    original = torch.Tensor.__dlpack_device__
    if getattr(original, "_torch_mojo_backend", False):
        return

    from torch.utils.dlpack import DLDeviceType  # noqa: PLC0415 -- mirrors the private import inside the method being patched

    @wraps(original)
    def __dlpack_device__(self: torch.Tensor) -> tuple[int, int]:
        if self.device.type == torch._C._get_privateuse1_backend_name():
            index = self.device.index if self.device.index is not None else 0
            return (DLDeviceType.kDLExtDev, index)
        return original(self)

    __dlpack_device__._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.Tensor.__dlpack_device__ = (  # ty: ignore[invalid-assignment]
        __dlpack_device__
    )


def fix_batch_isend_irecv_for_python_process_groups():
    """`batch_isend_irecv` never coalesces for a Python `ProcessGroup`
    subclass, and a bidirectional exchange then deadlocks.

    ``torch/distributed/distributed_c10d.py``'s ``batch_isend_irecv`` gates
    its NCCL-style coalescing on ``type(group) is ProcessGroup``. The mojo
    backend IS a Python subclass of ``ProcessGroup`` (it has to be: the
    C++ ``Backend`` cannot be implemented in Python), so the check is False
    however capable the backend is, and every operation in the list goes out
    as its own NCCL group. A two-rank exchange then deadlocks on the device:
    each rank's comm stream holds ``[send(->peer), recv(<-peer)]``, and the
    send cannot retire until the peer posts its recv, which sits behind that
    peer's own send. The very next line asks the backend whether it
    ``supports_coalescing``, which is the real question; ``isinstance`` is
    what the type check means. Everything else is torch's own code path,
    called unchanged.
    """
    original = c10d.batch_isend_irecv
    if getattr(original, "_torch_mojo_backend", False):
        return

    @wraps(original)
    def batch_isend_irecv(p2p_op_list: list[c10d.P2POp]) -> list[c10d.Work]:
        c10d._check_p2p_op_list(p2p_op_list)
        group = p2p_op_list[0].group or c10d._get_default_group()
        device = p2p_op_list[0].tensor.device
        coalesces = (
            type(group) is not torch.distributed.ProcessGroup
            and isinstance(group, torch.distributed.ProcessGroup)
            and group._get_backend(device).supports_coalescing
        )
        if not coalesces:
            return original(p2p_op_list)
        with c10d._coalescing_manager(group, device, async_ops=True) as manager:
            for op in p2p_op_list:
                peer = "group_dst" if op.op is c10d.isend else "group_src"
                op.op(op.tensor, group=op.group, tag=op.tag, **{peer: op.group_peer})
        return manager.works

    batch_isend_irecv._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    c10d.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]
    torch.distributed.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]
