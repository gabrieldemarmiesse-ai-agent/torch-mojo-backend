"""Bind this torchrun rank to the CPUs local to its GPU's NUMA node, then exec the demo.

Rank r uses GPU r (CUDA_VISIBLE_DEVICES is still the full list here: the demo slices it
later). The GPU's PCI bus id gives its NUMA-local CPU list from sysfs; the ranks that
share that node split its physical cores evenly, each rank keeping both hardware
threads of its cores. Falls back to a compact slice of the task's mask if sysfs or
nvidia-smi is unavailable (e.g. AMD).
"""

import os
import subprocess
import sys


def _cpulist(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def _numa_bind(lr: int, lw: int, allowed: list[int]) -> list[int] | None:
    try:
        bus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except Exception:
        return None
    gpu_cpus: dict[int, tuple[int, ...]] = {}
    for line in bus.strip().splitlines():
        idx, bdf = (x.strip() for x in line.split(","))
        dev = (
            "0000:" + bdf.split(":", 1)[1].lower()
            if bdf.count(":") == 2
            else bdf.lower()
        )
        try:
            local = _cpulist(open(f"/sys/bus/pci/devices/{dev}/local_cpulist").read())
        except OSError:
            return None
        gpu_cpus[int(idx)] = tuple(sorted(set(local) & set(allowed)))
    if lr not in gpu_cpus or not gpu_cpus[lr]:
        return None
    mine = gpu_cpus[lr]
    siblings = sorted(
        {
            tuple(
                sorted(
                    _cpulist(
                        open(
                            f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
                        ).read()
                    )
                )
            )
            for c in mine
        }
    )
    cores = [
        s for s in siblings if s[0] in mine
    ]  # one entry per physical core, all its threads
    peers = sorted(r for r in range(lw) if gpu_cpus.get(r) == mine)
    if not cores or lr not in peers:
        return None
    per = len(cores) // len(peers)
    k = peers.index(lr)
    chunk = cores[k * per : (k + 1) * per] if per else cores
    return sorted({c for core in chunk for c in core if c in allowed})


lr = int(os.environ["LOCAL_RANK"])
lw = int(os.environ["LOCAL_WORLD_SIZE"])
allowed = sorted(os.sched_getaffinity(0))
cpus = _numa_bind(lr, lw, allowed)
how = "numa"
if not cpus:
    per = len(allowed) // lw
    cpus = allowed[lr * per : (lr + 1) * per]
    how = "compact"
os.sched_setaffinity(0, cpus)
print(
    f"rank_bind[{how}]: local_rank {lr} -> {len(cpus)} cpus {cpus[0]}-{cpus[-1]} ({','.join(str(c) for c in cpus[:4])}...)",
    flush=True,
)
os.execv(sys.executable, [sys.executable, "demo_scripts/nanogpt_ddp.py", *sys.argv[1:]])
