"""Bind this torchrun rank to the CPUs local to its GPU's NUMA node, then exec the demo.

Rank r uses GPU r (CUDA_VISIBLE_DEVICES is still the full list here: the demo slices it
later). The GPU's PCI bus id gives its NUMA-local CPU list from sysfs; the ranks that
share that node split its physical cores evenly, each rank keeping both hardware
threads of its cores. The bus ids come from nvidia-smi on NVIDIA and rocm-smi on AMD
(MI300A: one APU per NUMA node, so the binding is the node's own cores); falls back to
a compact slice of the task's mask when neither tool nor sysfs answers.
"""

import os
import re
import shutil
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


def _gpu_bus_ids() -> dict[int, str] | None:
    """GPU index -> PCI address as sysfs spells it (dddd:bb:dd.f), in the order the
    runtime enumerates devices (nvidia-smi / rocm-smi order)."""
    ids: dict[int, str] = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        for line in out.strip().splitlines():
            idx, bdf = (x.strip() for x in line.split(","))
            ids[int(idx)] = _sysfs_bdf(bdf)
        return ids or None
    except Exception:
        pass
    rocm_smi = shutil.which("rocm-smi") or os.path.join(
        os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "rocm-smi"
    )
    try:
        out = subprocess.run(
            [rocm_smi, "--showbus"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except Exception:
        return None
    for m in re.finditer(
        r"GPU\[(\d+)\]\s*:\s*PCI Bus:\s*([0-9a-fA-F]+:[0-9a-fA-F]+:[0-9a-fA-F]+\.[0-9a-fA-F])",
        out,
    ):
        ids[int(m.group(1))] = _sysfs_bdf(m.group(2))
    return ids or None


def _sysfs_bdf(bdf: str) -> str:
    """'00000000:0A:00.0' (nvidia-smi) or '0001:02:00.0' (rocm-smi) -> '0000:0a:00.0'.
    The domain is kept: MI300A puts each APU in its own PCI domain."""
    domain, rest = bdf.split(":", 1)
    return f"{domain[-4:].zfill(4)}:{rest}".lower()


def _numa_bind(lr: int, lw: int, allowed: list[int]) -> list[int] | None:
    bus = _gpu_bus_ids()
    if bus is None:
        return None
    gpu_cpus: dict[int, tuple[int, ...]] = {}
    for idx, dev in bus.items():
        try:
            local = _cpulist(open(f"/sys/bus/pci/devices/{dev}/local_cpulist").read())
        except OSError:
            return None
        gpu_cpus[idx] = tuple(sorted(set(local) & set(allowed)))
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
