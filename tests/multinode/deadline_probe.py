"""One rank stops answering; the others must fail loudly, not hang.

Rank 0 sleeps past `MOJOCCL_IB_TIMEOUT_S` before its second allreduce. Every
other rank's second allreduce hits the device deadline inside the fused
inter-node kernel, latches the fault, and the THIRD allreduce (the reporting
call is one collective behind) raises `RuntimeError` with the
`DEVICE DEADLINE` line. Exit code 0 from a rank means it never saw the
deadline, which is the failure this probe exists to catch::

    MOJOCCL_IB_TIMEOUT_S=3 torchrun --nnodes=2 --nproc-per-node=8 ... \\
        tests/multinode/deadline_probe.py

Expected: ranks other than 0 print `probe: deadline raised as expected` and
exit 0; rank 0 exits 0 too (its own calls complete or fail after the
others') -- torchrun's exit code is 0 only when every rank saw the deadline
or was the one asleep.
"""

import os
import sys
import time

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import datetime  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402


def main():
    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    rank = dist.get_rank()
    timeout_s = float(os.environ.get("MOJOCCL_IB_TIMEOUT_S", "60"))
    device = torch.device("mojo", torch.accelerator.current_device_index())
    x = torch.ones(8 << 20, dtype=torch.float32, device=device)
    dist.all_reduce(x)
    torch.accelerator.synchronize()
    if rank == 0:
        time.sleep(timeout_s * 3 + 5)
    ok = False
    try:
        dist.all_reduce(x)
        torch.accelerator.synchronize()
        dist.all_reduce(x)
        torch.accelerator.synchronize()
    except RuntimeError as e:
        ok = "REMOTE" in str(e).upper() or "deadline" in str(e).lower()
        print(
            f"[rank {rank}] probe: deadline raised as expected: {str(e)[:120]}",
            flush=True,
        )
    if rank == 0:
        print("[rank 0] probe: slept through the deadline", flush=True)
        sys.exit(0)
    if not ok:
        print(f"[rank {rank}] probe: FAIL, no deadline error", flush=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
