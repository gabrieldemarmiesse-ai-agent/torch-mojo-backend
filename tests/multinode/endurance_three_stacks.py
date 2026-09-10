"""Endurance run analysis. usage: endurance_three_stacks.py LOGDIR JOB

Per-step times come from the tok/s lines of e2e_endurance_<job>_<cfg>.log.
Prints tokens/time throughput, drift (first vs last fifth), and pause statistics (steps slower than 2x the median):
count, share of wall time, and which step indices."""

import re
import sys
import glob
import statistics as st

TOK = 8 * 32 * 1024
STEP = re.compile(r"(?:\d+: )?step\s+(\d+) \| loss (\S+) \|\s+([\d.]+)k tok/s")
LOGDIR = sys.argv[1]
job = sys.argv[2]
for f in sorted(glob.glob(f"{LOGDIR}/e2e_endurance_{job}_*.log")):
    cfg = f.rsplit("_", 1)[1][:-4]
    t = {}
    loss = {}
    for line in open(f, errors="replace"):
        m = STEP.search(line)
        if m and int(m[1]) not in t:
            t[int(m[1])] = TOK / (float(m[3]) * 1e3)
            loss[int(m[1])] = float(m[2])
    n = max(t) if t else 0
    if n < 100:
        print(f"{cfg}: incomplete ({n} steps)")
        continue
    ts = [t[s] for s in range(3, n + 1) if s in t]
    med = st.median(ts)
    fifth = len(ts) // 5
    first, last = ts[:fifth], ts[-fifth:]
    pauses = [s for s in range(3, n + 1) if s in t and t[s] > 2 * med]
    pause_time = sum(t[s] - med for s in pauses)
    print(
        f"{cfg}: steps 3..{n}: {TOK * len(ts) / sum(ts) / 1e3:.0f}k tok/s (median step {med * 1e3:.1f} ms) | first fifth {TOK * len(first) / sum(first) / 1e3:.0f}k -> last fifth {TOK * len(last) / sum(last) / 1e3:.0f}k ({100 * (TOK * len(last) / sum(last) / (TOK * len(first) / sum(first)) - 1):+.1f}%) | pauses >2x median: {len(pauses)} ({100 * pause_time / sum(ts):.2f}% of time) at {pauses[:20]}{'...' if len(pauses) > 20 else ''} | loss@{n} {loss[n]}"
    )
