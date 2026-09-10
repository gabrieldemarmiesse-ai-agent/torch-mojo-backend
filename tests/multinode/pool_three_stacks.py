"""Pool the e2e_three_stacks logs of several jobs (node pairs).

usage: pool_three_stacks.py LOGDIR JOB...

Per batch size: mean of the per-round "steps 20-30" tok/s over all rounds of all pairs (+- 95% CI, t with n-1 dof), ratio vs stock with propagated CI, per-pair ratios, and pause statistics over
all 30-step runs (steps 3..30 slower than 2x that run's median: count, share of the runs' time)."""

import glob
import re
import statistics as st
import sys
from math import sqrt

LOGDIR = sys.argv[1]
jobs = sys.argv[2:]
T = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    10: 2.262,
    15: 2.145,
    20: 2.093,
    25: 2.064,
}
STEP = re.compile(r"step\s+(\d+) \| loss ([\d.]+) \|\s+([\d.]+)k tok/s \|\s+([\d.]+)s")


def ci(xs):
    n = len(xs)
    t = T.get(n, 1.96 if n > 25 else T[max(k for k in T if k <= n)])
    return t * st.stdev(xs) / sqrt(n) if n > 1 else float("nan")


runs = {}  # (bs, cfg) -> list of (job, mean20_30, per-step tok/s dict)
for job in jobs:
    for f in sorted(glob.glob(f"{LOGDIR}/e2e_three_stacks_{job}_bs*_*_*.log")):
        m = re.search(rf"{job}_bs(\d+)_(\d+)_(\w+)\.log", f)
        bs, cfg = int(m[1]), m[3]
        if cfg == "warmup":
            continue
        tps = {
            int(s[1]): float(s[3])
            for s in STEP.finditer(open(f, errors="replace").read())
        }
        if 30 not in tps:
            print(f"INCOMPLETE {f}")
            continue
        runs.setdefault((bs, cfg), []).append(
            (job, st.mean(tps[k] for k in range(20, 31)), tps)
        )
for bs in sorted({b for b, _ in runs}, reverse=True):
    ref = [r[1] for r in runs[(bs, "stock")]]
    rm, rc = st.mean(ref), ci(ref)
    print(f"\nbatch {bs}: {len(ref)} rounds per stack over {len(jobs)} node pairs")
    for cfg in ("stock", "mojo_nccl", "mojo_mojoccl"):
        r = runs[(bs, cfg)]
        t = [x[1] for x in r]
        tm, tc = st.mean(t), ci(t)
        ratio = tm / rm
        rci = ratio * sqrt((tc / tm) ** 2 + (rc / rm) ** 2)
        per_pair = []
        for job in jobs:
            a = [x[1] for x in r if x[0] == job]
            b = [x[1] for x in runs[(bs, "stock")] if x[0] == job]
            if a and b:
                per_pair.append(f"{st.mean(a) / st.mean(b):.3f}")
        steps = [(k, v) for x in r for k, v in x[2].items() if k >= 3]
        times = [262144 * bs / 32 / (v * 1e3) for _, v in steps]
        meds = {}
        for x in r:
            med = st.median(x[2][k] for k in range(3, 31))
            meds[id(x)] = med
        slow = [(x[0], k) for x in r for k in range(3, 31) if x[2][k] < meds[id(x)] / 2]
        slow_time = sum(
            262144 * bs / 32 / (x[2][k] * 1e3) - 262144 * bs / 32 / (meds[id(x)] * 1e3)
            for x in r
            for k in range(3, 31)
            if x[2][k] < meds[id(x)] / 2
        )
        print(
            f"  {cfg:13} {tm:6.0f}k +- {tc:3.0f}k  ratio {ratio:.3f} +- {rci:.3f}  per pair [{' '.join(per_pair)}]  pauses(>2x median): {len(slow)} of {len(times)} steps, {100 * slow_time / sum(times):.2f}% of time, at steps {sorted(set(k for _, k in slow))[:12]}"
        )
