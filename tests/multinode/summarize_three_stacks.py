"""Tables from e2e_honest logs. Every run is verified before it counts: the library that ran
(loader trace line for mojo stacks), the network transport (NCCL NET/IB lines for NCCL
stacks; mojoccl has no TCP data path, its transport is verified by a separate job), and
the NUMA binding. Warm-up runs are discarded. Mean +- 95% CI over rounds (t, n-1 dof)."""

import glob
import re
import statistics
import sys

job = sys.argv[1]
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
NAMES = {
    "stock": "stock CUDA torch 2.11 + NCCL",
    "mojo_nccl": "mojo backend + NCCL",
    "mojo_mojoccl": "mojo backend + mojoccl (all Mojo)",
}
runs, problems = {}, []
for f in sorted(
    glob.glob(f"/home/gabriel/ddp_work/logs/e2e_three_stacks_{job}_bs*_*_*.log")
):
    m = re.search(rf"{job}_bs(\d+)_(\d+)_(\w+)\.log", f)
    assert m is not None, f
    bs, i, cfg = int(m.group(1)), int(m.group(2)), m.group(3)
    if cfg == "warmup":
        continue
    text = open(f).read()
    tps, el = {}, {}
    for s in re.finditer(
        r"step\s+(\d+) \| loss ([\d.]+) \|\s+([\d.]+)k tok/s \|\s+([\d.]+)s", text
    ):
        k = int(s.group(1))
        tps[k] = float(s.group(3))
        el[k] = float(s.group(4))
    tag = f"bs{bs} run {i} {cfg}"
    if 30 not in tps:
        problems.append(f"{tag}: incomplete")
        continue
    lib = re.findall(r"collectives via (\S+) \(([^)]*)\)", text)
    ib = len(re.findall(r"NET/IB", text))
    sock = len(re.findall(r"NET/Socket", text))
    numa = len(re.findall(r"rank_bind\[numa\]", text))
    compact = len(re.findall(r"rank_bind\[compact\]", text))
    if cfg == "stock":
        if lib:
            problems.append(f"{tag}: the package was importable in the stock run")
        if ib == 0:
            problems.append(f"{tag}: no NET/IB line from NCCL")
    elif cfg == "mojo_nccl":
        if not lib or "libnccl" not in lib[0][0]:
            problems.append(f"{tag}: library line says {lib[:1]}")
        if ib == 0:
            problems.append(f"{tag}: no NET/IB line from NCCL")
    elif cfg == "mojo_mojoccl":
        if not lib or "mojoccl" not in lib[0][1]:
            problems.append(f"{tag}: library line says {lib[:1]}")
    if sock:
        problems.append(f"{tag}: NCCL reports NET/Socket ({sock} lines)")
    if numa < 16 or compact:
        problems.append(f"{tag}: binding numa={numa} compact={compact}")
    runs.setdefault((bs, cfg), []).append(
        (
            statistics.mean(tps[k] for k in range(20, 31)),
            el[1],
            [tps[k] for k in range(20, 31)],
        )
    )


def ci(xs):
    return (
        T[len(xs)] * statistics.stdev(xs) / len(xs) ** 0.5
        if len(xs) > 1
        else float("nan")
    )


print(
    "VERIFICATION:",
    "all runs verified" if not problems else "\n  " + "\n  ".join(problems),
)
for bs in sorted({b for b, _ in runs}, reverse=True):
    ref = [x[0] for x in runs[(bs, "stock")]]
    rm, rc = statistics.mean(ref), ci(ref)
    n = len(ref)
    print(
        f"\nBatch {bs}x1024 per rank, 16 ranks on 2x8 H100, nanoGPT-124M, bf16 autocast, 30 steps, {n} interleaved rounds after a discarded warm-up; +- is a 95% CI over rounds:\n"
    )
    print(
        "| stack | mean tokens/s, steps 20-30 | ratio vs stock CUDA torch + NCCL | step 1 (init) |"
    )
    print("|---|---|---|---|")
    for cfg in ("stock", "mojo_nccl", "mojo_mojoccl"):
        r = runs.get((bs, cfg), [])
        t = [x[0] for x in r]
        s1 = [x[1] for x in r]
        if not t:
            print(f"| {NAMES[cfg]} | no runs | | |")
            continue
        tm, tc = statistics.mean(t), ci(t)
        ratio = tm / rm
        rci = ratio * ((tc / tm) ** 2 + (rc / rm) ** 2) ** 0.5
        print(
            f"| {NAMES[cfg]} | {tm:.0f}k +- {tc:.0f}k | {ratio:.3f} +- {rci:.3f} | {statistics.mean(s1):.1f} s |"
        )
    per_step = {
        cfg: [v for x in runs.get((bs, cfg), []) for v in x[2]] for cfg in NAMES
    }
    print(
        "median per-step tokens/s over all rounds: "
        + ", ".join(
            f"{NAMES[c]} {statistics.median(v):.0f}k" for c, v in per_step.items() if v
        )
    )
