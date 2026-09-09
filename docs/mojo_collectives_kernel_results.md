# Mojo intra-node collectives — measured results

Deliverable: `collectives_kernels.mojo` (this directory). Harness: `harness.mojo`
+ `validate.sbatch` (correctness + the tables below, NCCL legs interleaved),
`ab.sbatch` (the NCCL A/B alone), `sweep.sbatch` / `tune.sbatch` /
`diag.sbatch` (the tuning sweeps), `bench.sbatch`, `smoke.sbatch`,
`split.sbatch` + `split_report.py` (the split allreduce of §10).
`./build.sh` builds the per-dtype harnesses and the assembly dumps;
`./build.sh sweeps` also builds the `-D`-tuned variants the sweep scripts
expect. Raw logs: `/home/gabriel/ddp_work/logs/ccl_*.log`.

All numbers: 8xH100 SXM (NVSwitch), one process per GPU, regions exchanged with
`cuIpcGetMemHandle`/`cuIpcOpenMemHandle`, **staged** (user tensors in
MAX-allocated memory; every copy the design needs is inside the measurement),
20 back-to-back launches x 5 reps, median, max over ranks. Device time = wall
over the 20 launches / 20, cross-checked against an in-kernel `globaltimer`
span recorded by stamp kernels on the same stream (they agree to <0.5%
everywhere; the CSV prints both). SM clock 1980 MHz (boost, unpinned — the
SLURM `--comment=gpu_clock` pin does not take on this cluster; checked per job).

Nodes: `cl02s01dgx18`, `cl02s01dgx04`, `cl02s01dgx16`, `cl02s04dgx01`. Node to
node spread on the same code is ~2%, so cross-job comparisons below are only
made within a job.

## 1. Headline: same-node A/B against NCCL (job 233776, shipped constants)

Legs interleaved within each round (NCCL NVLS, mojo fp32, mojo bf16, NCCL ring),
3 rounds, same node, same job. NCCL = stock torch 2.11.0+cu128 / NCCL 2.28.9,
device time from CUDA events, max over ranks — the same statistic the Mojo
harness prints.

fp32, world 8 (us):

| bytes | **mojo** | NCCL default (NVLS) | NCCL ring (`NCCL_NVLS_ENABLE=0`) | mojo / NVLS | target |
|---|---|---|---|---|---|
| 4 B | **9.4** | 18.4 | 17.5 | 0.51x | <= 25 ✔ |
| 256 KiB | **15.3** | 19.9 | 19.5 | 0.77x | — |
| 512 KiB | **21.4** | 19.6 | 19.6 | 1.09x | — |
| 1 MiB | **23.6** | 24.7 | 25.0 | 0.96x | — |
| 9 MiB | **64.6** | 85.8 | 76.4 | 0.75x | <= 91 ✔ |
| **27 MiB** (DDP bucket) | **163.8** | 174.9 | 181.8 | **0.94x** | <= 184 ✔ |
| 168 MiB (tail bucket) | **958** | 750 | 899 | 1.28x | "as close as unicast gets" |
| 512 MiB | **2860** | 2146 | 2636 | 1.33x | — |

bf16, world 8: 9.4 / 15.4 / 21.4 / 23.6 / 64.8 / **163.1** / 959 / 2856 us —
within 0.5% of fp32 at every size (target: 2%).

world 4 (fp32): 8.4 / 10.9 / 14.0 / 18.3 / 56.5 / 135.7 / 814 / 2437 us.
world 2 (fp32): 8.0 / 8.9 / 10.3 / 15.4 / 40.9 / 101.0 / 623 / 1856 us.

**The three stated targets are met with the staging included**, and the two
sizes that matter for GPT-2 DDP (9 and 27 MiB) beat NCCL's best algorithm on
the same node by 25% and 6%. The feasibility study's *direct* kernel (operating
on library-owned buffers, no user tensors involved) was 66 / 165 us at 9 /
27 MiB: **the staging now costs nothing measurable**, against +25% (82 /
210 us) for the naive copy-in/copy-out staging the study measured. The one
size where NCCL wins below 1 MiB is 512 KiB (21.4 vs 19.6), at the top of the
one-shot regime.

## 2. Why the staging became free

The user's tensors are MAX-allocated and cannot be exported by IPC, so peers can
only read the library's region. Instead of copying the input in and the result
out, the kernel makes the *transfers themselves* do the staging:

* **push** — each rank reads its own input straight from user memory and writes
  shard *s* into peer *s*'s slot. That write is the copy-in and it is the NVLink
  transfer the reduce-scatter needed anyway.
* **reduce** — each rank sums the `world` contributions to its own shard (its own
  from user memory, the peers' from its region) and writes the result both to
  its region and to its slice of the user output.
* **pull** — each rank reads the peers' reduced shards straight into the user
  output.

NVLink traffic per GPU is `2*(world-1)/world*bytes` — the unicast minimum, the
same as a direct reduce-scatter + all-gather — and no local byte is copied that
the direct kernel would not also copy. Three cross-GPU syncs: a start barrier
(the arena-reuse invariant, see §6) plus one per data dependency.

Messages at or below 512 KiB take a **one-shot** path instead — every rank
pushes its whole input to every peer and reduces locally, `(world-1)x` the
NVLink bytes but one data sync instead of two. Measured crossover (one-shot vs
two-shot, us): 128 KiB 9.3/19.0, 256 KiB 12.3/19.3, 512 KiB 18.1/19.8, 1 MiB
30.0/20.5 (job 233731, before the start barrier added ~2.5 us to both).

## 3. What tops out where: the unicast ceiling at 168 MiB

At 168 MiB the kernel is 1.07x NCCL ring and 1.28x NCCL NVLS. The decomposition
below says why, and it is measured, not modelled.

`allgather` with 21 MiB per rank (= 168 MiB output) moves exactly the same
154 MB of NVLink reads per GPU as the allreduce's pull phase, and nothing else
except a local 22 MB stage copy: **502-508 us**, i.e. **~305 GB/s per GPU per
direction**. The allreduce at 168 MiB is 958-990 us for a push (154 MB out)
plus a pull (154 MB in), i.e. **~315 GB/s per direction** — the same rate. So:

* both NVLink phases already run at the rate a pure peer-copy kernel achieves;
* the local reduce costs **~30 us of the ~980** (980 - 2x~480), not the ~110 us
  a naive HBM model predicts — the memory system absorbs it;
* therefore **pipelining the reduce behind the transfers can win at most ~3%**,
  and was not implemented (see §6, negative results).

The ceiling for this design is the fabric rate under all-to-all traffic:
**~310 GB/s per direction per GPU, 69% of NVLink4's 450 GB/s**. NCCL ring gets
343 GB/s (899 us) with neighbour-only traffic — ~10% better fabric efficiency
that a direct (all-to-all) reduce-scatter/all-gather does not reach by tuning
(every knob was swept, §5); closing it needs a different schedule, an
N-1-step ring, which is a different kernel and would give up the latency win
at 9-27 MiB that matters more for this workload.

**NCCL NVLS's 750 us is out of reach for any unicast kernel**, and not by a
small margin of tuning: `multimem.ld_reduce` has the NVSwitch do the reduction,
so each GPU moves `2*bytes/world` across the fabric instead of
`2*bytes*(world-1)/world` — 7x fewer bytes at world 8. To match it we would need
`cuMulticastCreate` + `cuMemExportToShareableHandle` + fd passing over an
AF_UNIX socket (`SCM_RIGHTS`), and — the part that reaches outside this file —
the region would have to be allocated with `cuMemCreate`/`cuMemMap` instead of
`cuMemAlloc` so it can be bound to the multicast object. MAX's
`DeviceMulticastBuffer` is single-process and cannot be reused across processes.
That is the exact gap; it was not attempted here because the region allocator
belongs to the plumbing layer. **It has since been closed exactly along those
lines** — `nvls_kernels.mojo` and `vmm.mojo`, dispatched above 48 MiB, taking
168 MiB from 990 to 790 us and 512 MiB from 2988 to 2226. Nothing in this file
changed: the six exported names below are untouched and the unicast kernels
still carry everything below the crossover. See the "NVLS" subsection of
docs/distributed.md.

Consequences for GPT-2 DDP: buckets 0-11 (9 and 27 MiB) are faster than NCCL and
overlapped with backward anyway; the exposed 168 MiB tail bucket costs
+0.21 ms/step versus NCCL NVLS, on a ~45 ms step.

## 4. Broadcast and allgather

| collective | size | first design (us) | **shipped design (us)** |
|---|---|---|---|
| broadcast | 27 MiB | 552 (root stages, all read) | **154-157** |
| broadcast | 168 MiB | 3374 | **927-1035** |
| broadcast | 256 MiB | — | **1436** |
| allgather | 3.4 MiB/rank (27 MiB out) | — | **96-98** |
| allgather | 21 MiB/rank (168 MiB out) | — | **502-508** |
| allgather | 32 MiB/rank (256 MiB out) | — | **778** |

(ranges are across nodes, jobs 233776/233789/233794; world 3 also measured:
broadcast 27 MiB 118 us, allgather 9 MiB/rank 77 us.)

The first broadcast put `(world−1)·nbytes` on the root's single outbound link
(552 us at 27 MiB where the fabric can do ~150). The shipped one is **scatter +
all-gather**: the root scatters shard *p* into rank *p*'s stage (nbytes out of
the root, spread over the peers), then every non-root rank gathers the `world`
shards. 3.6× faster, and both phases are permutations, so no link carries more
than `nbytes`. It is out-of-place capable (ncclBroadcast semantics): only the
root reads `send_ptr`, everyone writes `recv_ptr`, and when the two differ on
the root it copies locally instead of gathering its own message back over
NVLink. Allgather is a local stage + a peer gather — already the unicast
minimum — and honours a `stride_bytes` that differs from `nbytes_per_rank` so
the ABI layer can chunk one rank's contribution across calls. DDP's two
~250 MiB parameter broadcasts cost ~1.4 ms each instead of ~5 ms.

## 5. Tuning sweeps

Grid (blocks of 256 threads), fp32 world 8, µs:

| blocks | 4 B | 1 MiB | 9 MiB | 27 MiB | 168 MiB | 512 MiB |
|---|---|---|---|---|---|---|
| 64 | — | — | — | — | 1000 | 2955 |
| 96 | 6.8 | 20.6 | 63.2 | 177.2 | 991 | 2982 |
| 128 | 6.7 | 20.6 | 65.7 | 166.9 | **977** | **2826** |
| 132 | 6.9 | 20.7 | 67.4 | 167.5 | 993 | 2934 |
| 160 | — | — | — | — | 1104 | 3458 |
| **216** | 6.8 | 20.5 | **61.1** | 167.4 | 1002 | 3030 |
| 264 | 6.8 | 20.6 | 66.6 | **164.7** | 951 | 3051 |
| 432 | 6.9 | 20.7 | 64.0 | 169.8 | 1075 | 3331 |
| 864 | 6.7 | 20.5 | 63.5 | 175.0 | 1000 | 3168 |

Two regimes, hence the two constants in the file (`_AR_MAX_BLOCKS` = 216 below
64 MiB, `_AR_BIG_BLOCKS` = 128 at and above it; both fitted on H100 and marked
as such):

* **small/medium**: more threads win because the transfer is latency-bound —
  216 blocks is the flat optimum at 9 MiB (61.1 vs 65.7 at 128).
* **large**: a grid that fits in **one wave** of the 132 SMs wins, because the
  barrier is per block index: with more blocks than SMs the second wave runs the
  *whole* collective after the first, on fewer SMs. 128 blocks is 7% faster than
  216 at 512 MiB (2826 vs 3030). 160 blocks (1.2 waves, a long tail wave) is the
  worst point measured — 3458 µs, 22% worse than 128.

Unroll (16-byte vectors in flight per thread in the copy loops), fp32 world 8:

| unroll | 9 MiB | 27 MiB | 168 MiB | 512 MiB |
|---|---|---|---|---|
| 1 | 62.2 | 168.1 | 1083 | 3222 |
| 2 | 61.3 | 163.9 | 1025 | 3162 |
| **4** | 61.5 | 167.1 | **994** | **3078** |
| 8 | 61.6 | 164.5 | 1026 | 3369 |

Memory-level parallelism matters only in the bandwidth-bound regime (+9% from
U=1 to U=4 at 168 MiB) and is flat below 27 MiB. U=8 regresses (register
pressure). `_UNROLL = 4`.

One-shot/two-shot threshold (job 233731, three interleaved rounds, measured
before the start barrier added ~2.5 us to every leg). Each column is the same
size run by two builds differing only in `-D ccl_oneshot_max`, so the pair
isolates the path, not the size:

| bytes | 64 KiB | 128 KiB | 256 KiB | 512 KiB | 1 MiB | 2 MiB | 4 MiB |
|---|---|---|---|---|---|---|---|
| one-shot | 7.5 | **9.3** | **12.2** | **18.1** | 30.0 | 51.9 | 97.0 |
| two-shot | 7.5 | 19.0 | 19.3 | 19.9 | **20.5** | **24.2** | **34.5** |

The crossover sits just above 512 KiB, so `_ONESHOT_MAX_BYTES = 512 KiB`. Below
256 KiB the one-shot path is 2x faster; above 1 MiB it degrades as
`(world-1)x` the bytes should.

## 6. Negative results (do not re-explore)

* **Pipelining the local reduce behind the transfers** (chunk the shard, give
  each chunk its own pair of flags, and run `reduce(c+1)` concurrently with
  `pull(c)` — half the block's warps on each): designed, costed, **not
  implemented**, because the measurement in §3 bounds the win. The reduce is
  worth ~30 µs of the 988 at 168 MiB, and K chunks recover at most
  `(K−1)/K · 30 µs` while adding `2K` syncs at ~2–3 µs each: K=4 nets ≈ +7 µs,
  i.e. nothing. The design is sound and written down here so nobody re-derives
  it: the matching invariant is per *block*, not per thread (block *b* must
  produce exactly the indices block *b* of its peers consumes, and with a
  grid-stride loop that set is fixed by the loop shape), so threads inside a
  block may be split by role as long as every phase keeps the same
  block→index mapping. Worth revisiting only if the reduce ever gets more
  expensive (a wider dtype conversion, a fused op).
* **A stack array of peer pointers in the reduce loop** (`InlineArray[Pointer,
  MAX_WORLD]`, the natural way to write it, and what MAX's `_multi_gpu_barrier`
  warns about as MOCO-1431): the array is demoted to **local memory**, so the
  PTX shows `ld.local.b64` per pointer per iteration and every payload load
  becomes a generic-address `ld.v4.b32` instead of `ld.global.v4.b32`. Forming
  the address arithmetically inside the unrolled loop fixes it: 18
  `ld.global.v4.b32`, zero local traffic.
* **More blocks**: 432 and 864 are worse everywhere (see the table); the
  intuition "more blocks fill NVLink better" is wrong once the grid exceeds one
  or two waves, because the per-block barrier serialises waves.
* **Broadcast by staging the whole message on the root**: 3.6× slower than
  scatter + all-gather (552 vs 154 us at 27 MiB). Do not "simplify" it back.
* **A byte copier that falls back to single bytes when a pointer is
  misaligned**: `_copy_bytes` chooses 16-byte vectors or a fallback per call,
  and for a byte collective the writer and the reader are different ranks
  looking at different pointer pairs (the root's `send` vs a peer's `recv`), so
  they can disagree. A byte-wise fallback would then give writer and reader
  different index -> block mappings, and the barrier is per block index: block 3
  would read a chunk block 0 wrote, with nothing ordering them. Both paths now
  walk the same 16-byte chunks in the same grid-stride order; only the
  instructions inside a chunk differ. Found by inspection, not by a failing
  test — it needs `send` and `recv` to have different 16-byte alignment.
* **Dropping the start barrier** (two syncs per allreduce instead of three,
  ordering the arena reuse out of the phase structure alone): measured 2.5–3 µs
  cheaper at every size, and **wrong**. That argument only holds for a run of
  identically shaped collectives. DDP interleaves a 4-byte one-shot allreduce
  and byte collectives with 27 MiB two-shot allreduces, whose staging layouts
  overlap; with only the data syncs, rank R's push for generation g+1 is
  concurrent with a slower rank's reads for generation g whenever the two
  generations lay the arena out differently. The start barrier reduces the
  whole question to one invariant (§ "Buffer-reuse invariant" in the source),
  and `harness.mojo mix` — 800 interleaved generations of exactly that pattern,
  both allreduces in place so a single corrupted generation survives to the end
  — is the regression test.

## 6b. All-gather and broadcast

Same ABBA protocol, fp32, 4 ranks, per-rank contribution for the all-gather
and message size for the broadcast. Busbw is `(world-1)*n/t` and `n/t`
respectively, the same convention `ar_bench.py` prints.

| op | size | RCCL µs | mojo µs | ratio | mojo busbw | before this work |
|---|---|---|---|---|---|---|
| all_gather | 4 MiB | 101.8 | 134.7 | 1.32 | 93 GB/s | |
| all_gather | 16 MiB | 274.7 | 319.8 | 1.16 | 157 GB/s | |
| all_gather | 64 MiB | 947.8 | 1146.1 | 1.21 | **176 GB/s** | 80 GB/s |
| broadcast | 4 MiB | 53.2 | 89.8 | 1.69 | 47 GB/s | |
| broadcast | 16 MiB | 121.1 | 217.9 | 1.80 | 77 GB/s | |
| broadcast | 64 MiB | 342.9 | 680.0 | 1.98 | **99 GB/s** | 78 GB/s |

The all-gather is now within ~20% and the remaining gap is the same one the
allreduce has at the large sizes. The broadcast is not, and the reason is its
schedule rather than its direction: scatter-then-push-all-gather puts
`n + (world-1)/world*n` = 1.75 × message of outbound traffic on the root,
where a ring or a tree puts `n`. Converting the gather half from a pull to a
push took it from 78 to 99 GB/s; getting to RCCL's 195 needs the root to stop
being the only sender of the first phase, which is a different algorithm and
was out of scope here.

## 7. Correctness, and one unresolved flake

`tests/ddp_worker.py` through the process group, `TORCH_MOJO_BACKEND_CCL=mojo`,
`perf-work/suite.sh`, on the shipped tree:

| mode | 4 ranks | 2 ranks |
|---|---|---|
| `collectives` | 20 OK / 0 FAIL | 11 OK / **4 FAIL** (see below) |
| `ddp_parity` | 5 / 0 | 3 / 0 |
| `lazy_fence` | 11 / 0 | 6 / 0 |
| `stress` (`MOJOCCL_REGION_MB=4`, forces chunking) | 282 / 0 | 161 / 0 |
| `abort` | 16 / 0 | 10 / 0 |

`stress` with the default 256 MiB region also passes (300 OK / 0 FAIL). It
exits nonzero for reasons that have nothing to do with these kernels: without
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM` four ranks reserving ~124 GB each
are OOM-killed on this APU, and with it the process segfaults in HIP's atexit
handler unless the script ends in `os._exit(0)` — both already in
`docs/distributed.md`.

**The flake.** `allreduce.int64` — a *one-element* int64 allreduce, which
takes the one-shot path with a single 8-byte store and a single 8-byte load,
one block, one active thread — fails intermittently **at 2 ranks**. Measured
today: 2 failures in 14 runs of `ddp_worker.py collectives` at 2 ranks; every
4-rank run of every mode passed, and 8 consecutive 2-rank repeats passed
before one failed. It is the smallest and most latency-sensitive collective in
the suite, and the first failing check in the file, so a run either fails
immediately or not at all.

What is known and what is not:

* the one-shot kernel itself is **unchanged** by this work; the only thing
  under it that changed is `_sync`'s acquire (relaxed spin plus one acquire
  fence after the wait, instead of an acquire load per iteration), so that is
  the first suspect;
* the same rare failure was also seen with the *intermediate* barrier (the one
  that moved the release writeback after the block barrier), alongside
  reproducible broadcast failures that the shipped barrier fixed;
* **whether the pre-MI300A tree also flakes here was not established.** The
  A/B was attempted (`perf-work/flake.sh`, 8 runs per tree) and the
  before-tree half is void: those runs die in `ncclCommInitRank` with
  `ncclResult_t=3` in this harness, which is a harness problem, not a result.

Next step for whoever picks this up: run `perf-work/flake.sh` a few dozen
times on the shipped tree and on the same tree with the acquire reverted to
`Atomic.load[ACQUIRE]` in the spin — a couple of hundred runs will separate a
pre-existing race from one introduced here, and the acquire is the only knob
between them. Do not read the current numbers as clearing it.

