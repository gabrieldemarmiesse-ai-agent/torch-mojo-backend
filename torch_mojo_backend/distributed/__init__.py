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


def use_local_rank_gpu():
    from torch_mojo_backend.distributed.process_group import (  # noqa: PLC0415 -- process_group needs torch._C._distributed_c10d, absent from torch builds without distributed
        use_local_rank_gpu as impl,
    )

    impl()


def register_distributed_backend():
    """Register the "mojo" backend name with torch.distributed (idempotent)."""
    if not torch.distributed.is_available():
        return
    if "mojo" in torch.distributed.Backend.backend_list:
        return
    from torch_mojo_backend.distributed.process_group import (  # noqa: PLC0415 -- same as above; the is_available() guard right before is what makes it optional
        create_mojo_process_group,
    )

    torch.distributed.Backend.register_backend(
        "mojo", create_mojo_process_group, devices=["mojo"]
    )
