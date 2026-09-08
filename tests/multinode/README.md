# Two-node mojoccl validation and bench

`run_two_node_checks.sbatch` is a 2-node x 8-GPU (16-rank) SLURM job that
exercises the mojo distributed backend (`torch_mojo_backend/distributed/`)
both with real NCCL (`TORCH_MOJO_BACKEND_CCL` unset, called "vendor" below)
and with `mojoccl` (`TORCH_MOJO_BACKEND_CCL=mojo`, the in-repo Mojo
re-implementation of NCCL's C ABI — see the "Mojo collectives" section of
`docs/distributed.md`). Background reading: `AGENTS.md`, `docs/distributed.md`,
`docs/mojo_collectives_feasibility.md` (especially §2 traffic profile, §5.1
and §5.5 for the single- and two-node NCCL reference numbers, and §7/§8 for
the multi-node design this job is meant to validate once it lands).

## What it runs

One `sbatch` job, three phases, always in this order:

**A. `tests/ddp_worker.py`, 16 ranks.** For each `ccl` in `vendor`, `mojo`:
run modes `collectives`, `ddp_parity`, `stress` (one `torchrun
--nnodes=2 --nproc-per-node=8` launch per mode). `stress` is a no-op PASS
under vendor NCCL (it self-skips — see `run_stress` in `ddp_worker.py`); it
is mojoccl-specific regression coverage ported from the kernel harness.

**B. `ar_bench_gpt2.py` allreduce bench, ABBA order:** `vendor, mojo, mojo,
vendor` — the AGENTS.md-documented ordering that cancels a thermal/clock
ramp to first order. Prints one `RESULT ccl=... dtype=... size_mib=...
median_us=... min_us=... busbw_gbs=...` line per (ccl, dtype, size) at 1,
9, 27, 168, 512 MiB, fp32 and bf16. Only the vendor legs run with
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` (algo/protocol choices — the
mojo legs don't go through NCCL's tuner, and mojo's `TORCH_MOJO_BACKEND_CCL`
path talks to `libmojoccl.so`, not NCCL, so the flag can't tell us anything
about it).

**C. `demo_scripts/nanogpt_ddp.py --device mojo`, 40 steps, ABBA order,**
same shape as B: `vendor, mojo, mojo, vendor`. Fixed args:
`--nanogpt-path /home/gabriel/ddp_work/nanoGPT --data-dir
/home/gabriel/ddp_work/nanoGPT/data/shakespeare --log-interval 1
--eval-iters 10 --seed 1337 --max-iters 40 --eval-interval 40` (an eval,
hence a val-loss line, is printed once at step 40).

Every leg is a single `torchrun` launch spanning both nodes
(`--rdzv-backend=c10d`, `MASTER_ADDR` from `scontrol show hostnames`), run
through `srun bash -c "... flock /tmp/gpu_lock_0.lock uv run --no-sync
torchrun ..."` — `srun` puts one such shell on each node (`--ntasks-per-node=1`
from the `#SBATCH` header), so the flock is local to that node's `/tmp` and
guards that node's 8 GPUs against any other job touching them concurrently.
`PYTHONPATH` is pinned to this worktree
(`/home/gabriel/ddp_work/mojo_coll_mn`) so `ar_bench_gpt2.py`, which lives
outside the repo under `/home/gabriel/ddp_work/mojo_collectives/`, resolves
`torch_mojo_backend` from here rather than whatever else might be on the
path.

## RUN_MOJO

`RUN_MOJO=0` (env var, e.g. `sbatch --export=ALL,RUN_MOJO=0
run_two_node_checks.sbatch`) skips every mojo-ccl leg: phase A's `ccl=mojo`
iteration is skipped outright, and phases B and C run their vendor leg(s)
only (no ABBA — there is nothing to interleave against). Default
(`RUN_MOJO=1` or unset) attempts every leg.

As of this writing `torch_mojo_backend/distributed/mojoccl/` is intra-node
only (its rendezvous is a `/dev/shm` directory, `MAX_WORLD = 8`) — multi-node
transport is being implemented on this same branch
(`mojo-collectives-multinode`) in parallel. **Use `RUN_MOJO=0` until that
lands**: without it, every mojo leg at 16 ranks will fail or hang up to
`ddp_worker.py`'s 300 s `init_process_group` timeout, burning the job's time
budget for no signal. Once the transport lands, drop `RUN_MOJO=0` (or set
`RUN_MOJO=1`) to exercise it end to end.

## Output

`-o /home/gabriel/ddp_work/logs/mojoccl_2node_%j.log` is the **job log** —
node names, per-node SM clock, branch/commit, then, per leg, a `===
<phase> ccl=... ... ===` header, a filtered view of that leg's output
(`[rank N] OK/FAIL ...` lines for ddp_worker, `RESULT ...` lines for the
bench, `step ...`/`val loss ...` lines for training), and a `=== <phase>
ccl=... ... exit: N ===` trailer with that leg's `torchrun` exit code. This
is the file `summarize.py` reads.

Each leg's **full, unfiltered** output (including every per-step line and,
for the vendor bench legs, the full `NCCL_DEBUG=INFO` tuning log) is also
saved to its own file under `/home/gabriel/ddp_work/logs/`, e.g.
`mojoccl_2node_<jobid>_worker_vendor_collectives.log`,
`mojoccl_2node_<jobid>_arbench_vendor_leg1.log`,
`mojoccl_2node_<jobid>_nanogpt_mojo_leg2.log`.

## summarize.py

```bash
uv run --no-sync python tests/multinode/summarize.py \
    /home/gabriel/ddp_work/logs/mojoccl_2node_<jobid>.log
```

Stdlib only (no torch import — runs fine on the login node). Parses the job
log's `RESULT`/`=== ... ===`/`step ...` lines (regexes at the top of the
file) into one markdown report with three tables: allreduce bench (per
dtype/size, vendor and mojo median device time and the mojo/vendor ratio —
median-of-medians if a ccl has more than one ABBA leg at that size), the six
`ddp_worker` pass/fail results, and the training runs' step-40 loss, val
loss and tok/s with pass/fail. Pass `--out FILE` to write the report to a
file instead of stdout; multiple log paths may be given (e.g. to merge a
job log with one of the per-leg logs) and are concatenated before parsing.

## Vendor-only NCCL reference

The first run of this job should be `RUN_MOJO=0`, to establish the current
NCCL reference numbers at 16 ranks before the mojoccl transport exists —
see `/home/gabriel/ddp_work/mojo_collectives/mn/NCCL_REFERENCE_16.md` for
node names, SM clock, NCCL algo/protocol choices and the resulting numbers
from that run.
