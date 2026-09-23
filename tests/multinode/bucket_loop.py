"""DDP's traffic shape without DDP, for bisecting a multi-node slowdown.

A list of gradient buckets, allreduced in order, once per "step", with one
synchronize per step -- plus optional concurrent compute and optional
rank-dependent skew, which are the two things DDP adds that a size sweep
does not. It exists because `demo_scripts/nanogpt_ddp.py` at 8 ranks over two
nodes ran 37x slower than the same job under RCCL while every isolated
collective benchmark said the transport was fine; this reproduces the traffic
without the training loop, so the two can be compared.

Measured on 2 nodes x 4 MI300A over cxi, 216 MiB of buckets per step: 6.0
ms/step with no compute, 8.8 with one matmul per bucket, 36.8 with a
rank-skewed three. nanoGPT's step, with the same bytes, takes 2800 ms -- so
the collectives are not what costs it.

BUCKETS  comma-separated bucket sizes in MiB
STEPS    steps to time
ALLOC    once (default) | each
MM       matmuls of size MMDIM enqueued on the DEFAULT stream before each
         bucket's allreduce -- what DDP's backward is doing while the
         collective runs. 0 disables.
"""

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402

register_mojo_devices()
dist.init_process_group(backend="mojo")
rank = dist.get_rank()
buckets = [
    float(s) for s in os.environ.get("BUCKETS", "1,25,25,25,25,25,90").split(",")
]
steps = int(os.environ.get("STEPS", "10"))
alloc = os.environ.get("ALLOC", "once")
mm = int(os.environ.get("MM", "0"))
skew = int(os.environ.get("SKEW", "0"))
# Tensors allocated and dropped BETWEEN buckets, i.e. while the previous
# bucket's collective is still outstanding. `ALLOC=each` does not test this:
# the loop synchronizes once per step, so its frees all land on an idle
# device. DDP's backward frees intermediates continuously with collectives in
# flight, which is the thing worth reproducing.
free_n = int(os.environ.get("FREE", "0"))
free_mib = float(os.environ.get("FREE_MIB", "8"))
mmdim = int(os.environ.get("MMDIM", "4096"))
dev = torch.device("mojo")
dt = torch.float32


def make():
    return [torch.ones(int(m * 2**20) // 4, dtype=dt, device=dev) for m in buckets]


bufs = make() if alloc == "once" else None
w = torch.randn(
    mmdim if mm else 1, mmdim if mm else 1, dtype=torch.bfloat16, device=dev
)
for t in bufs if bufs is not None else make():
    dist.all_reduce(t)
if mm:
    for _ in range(mm):
        w = w @ w
torch.accelerator.synchronize()
dist.barrier()
torch.accelerator.synchronize()

per_step = []
for s in range(steps):
    bs = bufs if bufs is not None else make()
    t0 = time.perf_counter()
    for i, t in enumerate(bs):
        # Rank-dependent extra compute: DDP's ranks reach each bucket at
        # different times because their backward does.
        extra = skew * ((rank + i) % 4)
        for _ in range(mm + extra):
            w = w @ w
        dist.all_reduce(t)
        for _ in range(free_n):
            tmp = torch.empty(int(free_mib * 2**20) // 4, dtype=dt, device=dev)
            del tmp
    torch.accelerator.synchronize()
    per_step.append((time.perf_counter() - t0) * 1e3)
total_mib = sum(buckets)
med = sorted(per_step)[len(per_step) // 2]
print(
    f"[rank {rank}] {len(buckets)} buckets, {total_mib:g} MiB/step, alloc={alloc}, "
    f"mm={mm}+skew{skew}x{mmdim} free={free_n}x{free_mib:g}MiB: median {med:.1f} ms/step (first {per_step[0]:.1f}, "
    f"last {per_step[-1]:.1f})",
    flush=True,
)
dist.destroy_process_group()
sys.stdout.flush()
if os.environ.get("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM"):
    os._exit(0)
