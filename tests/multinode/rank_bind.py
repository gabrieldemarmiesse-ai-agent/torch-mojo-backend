"""Bind this torchrun rank to a compact slice of the task's CPU mask, then exec the demo."""

import os
import sys

lr = int(os.environ["LOCAL_RANK"])
lw = int(os.environ["LOCAL_WORLD_SIZE"])
cpus = sorted(os.sched_getaffinity(0))
per = len(cpus) // lw
mine = cpus[lr * per : (lr + 1) * per]
os.sched_setaffinity(0, mine)
print(
    f"rank_bind: local_rank {lr} -> cpus {mine[0]}-{mine[-1]} ({len(mine)})", flush=True
)
os.execv(sys.executable, [sys.executable, "demo_scripts/nanogpt_ddp.py", *sys.argv[1:]])
