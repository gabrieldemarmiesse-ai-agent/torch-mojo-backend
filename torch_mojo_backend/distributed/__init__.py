"""Distributed training (DDP) support for the mojo eager device.

Usage with torchrun::

    import torch_mojo_backend.distributed as mojo_dist

    mojo_dist.use_local_rank_gpu()      # before any mojo tensor exists
    torch_mojo_backend.register_mojo_devices()
    torch.distributed.init_process_group(backend="mojo")
    model = DDP(model.to("mojo"), broadcast_buffers=False)

``register_mojo_devices()`` registers the "mojo" c10d backend automatically;
``use_local_rank_gpu()`` pins each torchrun worker to its own GPU.
"""

import torch.distributed

__all__ = ["register_distributed_backend", "use_local_rank_gpu"]


def use_local_rank_gpu() -> None:
    from torch_mojo_backend.distributed.process_group import use_local_rank_gpu as impl

    impl()


def register_distributed_backend() -> None:
    """Register the "mojo" backend name with torch.distributed (idempotent)."""
    if not torch.distributed.is_available():
        return
    if "mojo" in torch.distributed.Backend.backend_list:
        return
    from torch_mojo_backend.distributed.process_group import create_mojo_process_group

    torch.distributed.Backend.register_backend(
        "mojo", create_mojo_process_group, devices=["mojo"]
    )
