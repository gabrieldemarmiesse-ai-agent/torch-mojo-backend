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
belongs to the plumbing layer.

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

## 7. Correctness

`harness.mojo verify` (job 233776, 94 passing cases) checks every result against
a host reference recomputed from the same deterministic splitmix64 fill (no
RNG):

* dtypes float32, bfloat16, float16, int32, int64 — world 8; plus float32 world
  4, 2 and **1**, int64 world 3, bfloat16 world 5, int32 world 7 (odd worlds
  take the runtime-`world` kernel instantiation; world 1 degenerates to
  `out = scale * in` and is exercised, not special-cased away).
* sizes 1, 2, 3, 1003, one page, 65537 and `(cap−1)/element_size` elements — i.e.
  ragged, non-multiples of the 16-byte vector width, and the 4-byte case.
* every size run twice: out-of-place and **in-place** (`in_ptr == out_ptr`).
* float32 is checked **exactly** (the fill is k/128 with |k| ≤ 128, so the sum of
  8 is exact in fp32 and the ×1/8 scale is a power of two); bf16/fp16 to half a
  bf16 ulp; **int64 against an Int64 reference with samples of magnitude 2⁵⁶**,
  which no fp64 reference could verify.
* the region's error word is read back after every case (it stays 0).

`harness.mojo mix` is the interleaving regression test: 200 rounds x (4-byte
one-shot allreduce, 27 MiB two-shot allreduce, 2000-byte broadcast, 8-byte
allgather) = **800 generations** whose staging layouts overlap, both allreduces
in place with `scale = 1/world` so the value is idempotent once every rank holds
the mean and a single corrupted generation anywhere in the run is still visible
in the final buffer. Passes for float32 and bfloat16 at world 8, float32 at world 4 and bfloat16 at
world 5 (jobs 233776, 233789, 233794). This is the test that the start barrier
exists for.

Broadcast and allgather results are checked byte-for-byte against the same hash
(sampled every 997 bytes) in the `ops` suite.

## 8. Build

Both targets build from the one file, no vendor `#ifdef` in the device code:

```
uv run --no-sync mojo build harness.mojo -I . --target-accelerator sm_90a  -D dtype=float32 -o harness_float32
uv run --no-sync mojo build harness.mojo -I . --target-accelerator gfx942  -D dtype=float32 -o harness_f32_gfx942
```

`collectives_kernels.mojo` compiles warning-free on both. The emitted device
code is what NCCL/RCCL rely on (`asm/` holds the dumps, produced by
`dump_asm.mojo` with no GPU present):

| | sm_90a | gfx942 |
|---|---|---|
| flag publish | `st.release.sys.global.b64` | `global_store … sc0 sc1` + `buffer_wbl2 sc0 sc1` |
| flag wait | `ld.acquire.sys.global.b64` | `global_load … sc0 sc1` + `buffer_inv sc0 sc1` |
| payload | `ld/st.global.v4.b32` (18/12 in the allreduce) | `global_load/store_dwordx4` (18/12) |
| block barrier | `bar.sync` | `s_barrier` |
| deadline | `%globaltimer` | `s_memrealtime` |

Blocks are 256 threads (RCCL's gfx942 maximum) and every layout is wave-64
safe. The file has exactly **one** comptime vendor gate, in `_sync`: gfx942
emits its workgroup barrier as `s_waitcnt lgkmcnt(0); s_barrier`, which does
*not* wait on vector memory, so another wave's payload stores can still be in
flight when one thread publishes the flag. RCCL solves this by putting
`vmcnt(0)` inside its block barrier; the portable spelling is a release fence
in every thread, which lowers to `s_waitcnt vmcnt(0)` + `buffer_wbl2 sc0 sc1`.
NVIDIA needs nothing there (`bar.sync` is a CTA-scope fence and the release
store is cumulative over it — what NCCL's `postPeer` relies on), and the gate
is comptime, so the sm_90a instruction mix is byte-identical with and without
it (checked by diffing the dumps).

AMD numbers are unmeasured (no MI300A here). Two things the AMD host side will
need, from RCCL's source: the region must be allocated **uncached**
(`hipExtMallocWithFlags(hipDeviceMallocUncached)`) — RCCL's precondition for
polled flags on MI300 — and the tuning constants above are H100 fits and are
marked as such in the source.

## 9. What the ABI layer needs to know

(§10 adds three more names for the multi-node path; everything below applies to
them unchanged.)

Six exported names, matching the contract exactly: `signal_bytes`,
`error_offset`, `region_init(ctx, region)` (no stream — it blocks),
`allreduce[dtype](ctx, stream, ...)`, `broadcast(...)`,
`allgather(..., stride_bytes=-1)`. `MAX_WORLD = 8`. Everything is enqueued on
the stream passed in and returns right after the enqueue; `ctx` is only the
compile/cache handle (`DeviceFunction`s are cached process-globally per
`ctx.id()`, so the per-call cost is the enqueue, not a ~180 us
`compile_function`).

Preconditions the file enforces by raising, all of them cheap host-side checks:

* `numel * size_of[dtype]() <= cap_bytes`, `nbytes <= cap_bytes`,
  `nbytes_per_rank <= cap_bytes` — the caller chunks anything larger (it does).
* **`in_ptr` and `out_ptr` of `allreduce` must be 16-byte aligned.** The payload
  loops use 16-byte vectors, which fault on a misaligned address. Every
  allocator-returned pointer satisfies this and so does every chunk offset the
  ABI layer forms (they are multiples of `cap_bytes`); a mid-tensor *view* may
  not, and such a tensor must be staged into an aligned buffer by the caller
  rather than have the kernel guess. `broadcast`/`allgather` need no such rule —
  the byte copier checks alignment at run time and falls back to a scalar loop.
* `world` in `1..8` (world 1 works and degenerates to `out = scale * in`),
  `rank < world`, `generation >= 1`, `cap_bytes` a positive multiple of 4096.
* `scale` is applied on the final write for floating-point dtypes and ignored
  for integers (the caller passes 1.0 there, which is what NCCL's `ncclAvg`
  does anyway).

Two notes on the surrounding layer:

* **The error word is device memory.** `error_offset()` is a byte offset into
  the region, which is `cuMemAlloc`/`hipMalloc` memory: it cannot be read by
  dereferencing a host pointer (that faults or reads garbage). Copy it back with
  `cuMemcpyDtoH` / a one-thread kernel, as `harness.mojo`'s `check_error` does.
  A nonzero value is `code * 1_000_000 + phase` and means some block gave up
  waiting for a peer (60 s deadline, measured with the GPU's own timer, never
  compared across GPUs); the collective's result is undefined from that
  generation on.
* `generation` must be strictly increasing per communicator **across all
  collective kinds** — the flag values are `generation * 8 + phase` and the
  start barrier's whole job is to order one generation's writes after the
  previous generation's reads. A repeated or decreasing generation silently
  passes barriers it should not.

## 10. Split allreduce for the multi-node path (job 233937)

The hierarchical allreduce of §7 of the feasibility study needs the intra-node
allreduce cut in half so the vendor library can allreduce one shard across
nodes in between. Three names were added; the six existing exports are
untouched (their device code is **byte-identical** before and after -- checked
by diffing `asm/{ar2,ar1,bcast,ag}_{sm_90a,gfx942}.asm` against a build of the
previous file).

```
shard_range(numel, world, rank, elem_bytes) -> (offset_elems, count_elems)
reduce_scatter_stage[dtype](ctx, stream, rank, world, regions, in_ptr,
                            numel, cap_bytes, generation)
allgather_finish[dtype](ctx, stream, rank, world, regions, out_ptr,
                        numel, cap_bytes, scale, generation)
```

`rank`/`world`/`regions` are the **node-local** group. The sequence per bucket:

```
reduce_scatter_stage(g)        push + local reduce; my shard, SUM over the
                               node, unscaled, lands in MY stage_out
<vendor allreduce, in place>   region + signal_bytes() + cap_bytes
                               + offset*elem_bytes, count elements, SAME stream
allgather_finish(g+1)          start barrier, then pull every rank's shard
                               (mine included) into out_ptr, times `scale`
```

Preconditions are `allreduce`'s (16-byte aligned buffer, `numel*elem_bytes <=
cap_bytes`, strictly increasing generation); the pair costs **two**
generations. `out_ptr` may be the same buffer as `in_ptr`.

### 10.1 Shard placement

`shard_range` is a different partition from the fused kernel's and
deliberately so: equal shards of `per` elements with `per` rounded up to the
16-byte vector width, the last non-empty shard short, ranks past the end
empty. Every offset is therefore 16-byte aligned, every rank derives the same
table from `(numel, world, elem_bytes)` alone, and the ABI layer can state the
inter-node collective's arguments in one line. Imbalance against the fused
kernel's balanced split is at most one 16-byte vector per rank.

The shard sits in stage_out **at its own element offset**, i.e. stage_out is an
image of the whole buffer of which only my shard is live. The push slots go at
the base of stage_in and are *compacted* to `world-1` (rank s never writes its
own slot): `world` uncompacted slots can be up to `16*world` bytes larger than
stage_in when `numel*elem_bytes == cap_bytes`, which would spill onto rank 0's
shard in stage_out. Checked exhaustively over every dtype width, world and cap.

### 10.2 The three ordering questions, and why no new protocol was needed

1. **A foreign library rewrites my stage_out shard between the two calls.**
   `allgather_finish`'s start barrier is the fence: a rank publishes its
   generation g+1 flags only from inside that kernel, which its stream starts
   only after its inter-node op completed. Seeing peer p's flag therefore
   implies p's shard is final.
2. **Peer p's shard was written by a previous kernel**, not by the thread that
   publishes the flag. Stream order puts that kernel's writes happen-before the
   release store, and the release/acquire pair is system-scoped and cumulative,
   so the acquiring reader sees them (on gfx942 `_sync`'s AMD-only release
   fence supplies the `buffer_wbl2 sc0 sc1` the workgroup barrier omits). The
   consequence is worth stating: block-index matching, which the fused kernel
   needs *within* one launch, is **not** needed across this boundary, so the
   two halves may be launched with different grids -- and they are.
3. **Arena reuse after the pulls.** Nothing new: the next collective of any
   kind opens with a start barrier and a rank reaches it only after its own
   `allgather_finish` retired, so no generation g+2 write can race a generation
   g+1 pull. The split pair just spends two generations. The existing
   buffer-reuse invariant covers it as written -- verified, not assumed, by the
   extended `mix` below.

### 10.3 Split vs fused, 8xH100 SXM, 1980 MHz (job 233937)

`harness.mojo split` runs, per size, five legs back to back in one process:
the fused `allreduce`; each half alone (both are legal standalone collectives);
the stand-in inter-node step alone, so it can be subtracted; and the whole
pair. Device time from `%globaltimer` stamps on the stream, 20 back-to-back
launches x 5 reps, median, max over ranks, three interleaved rounds.

fp32, world 8 (us):

| bytes | fused | rs stage | stub | ag finish | **pair** | pair-fused | minus stub |
|---|---|---|---|---|---|---|---|
| 4 B | 9.5 | 8.8 | 2.1 | 7.1 | **17.6** | +8.1 | +5.9 |
| 9 MiB | 64.7 | 36.3 | 3.1 | 35.5 | **74.8** | +10.1 | +7.1 |
| **27 MiB** | 162.7 | 89.6 | 3.7 | 93.6 | **186.3** | +23.7 | +19.9 |
| 168 MiB | 935.4 | 507.2 | 9.8 | 492.4 | **1003.9** | +68.5 | +58.8 |
| 512 MiB | 2777.6 | 1513.2 | 60.7 | 1457.9 | **3025.8** | +248.2 | +187.5 |

bf16, world 8 (us):

| bytes | fused | rs stage | stub | ag finish | **pair** | pair-fused | minus stub |
|---|---|---|---|---|---|---|---|
| 4 B | 9.6 | 8.9 | 2.3 | 7.0 | **17.9** | +8.3 | +6.1 |
| 9 MiB | 64.7 | 36.3 | 3.3 | 35.5 | **74.9** | +10.2 | +6.9 |
| **27 MiB** | 162.4 | 89.7 | 4.2 | 93.6 | **187.9** | +25.4 | +21.3 |
| 168 MiB | 935.9 | 503.1 | 11.6 | 491.5 | **1012.2** | +76.3 | +64.7 |
| 512 MiB | 2778.8 | 1501.4 | 101.2 | 1458.8 | **3057.7** | +278.9 | +177.7 |

The stand-in is a one-pass read-modify-write of the shard on the same stream
(it adds a rank-independent constant, so the verify can tell "the inter-node
step ran" from "it was skipped"); a real `ncclAllReduce` over `nNodes` costs
much more, and these columns exist so it can be substituted rather than
guessed at.

**Net of the stand-in the pair costs +6 to +8 us up to 9 MiB and +6-7% at
168-512 MiB**, i.e. the promised "fused plus one launch" at small and medium
sizes, growing to a percentage at the bandwidth-bound end. Where it goes:

* one extra kernel launch and one extra 8-way start barrier (~6 us, and that
  is the whole story at 4 B and 9 MiB);
* `allgather_finish` pulls **`world`** shards, not `world-1`: my own shard has
  to come back out of my stage_out because the inter-node step rewrote it,
  where the fused kernel wrote its own shard straight to the user output during
  the reduce. That is `bytes/world` of extra local read at every size, and it
  is the term that grows.

Neither half is slow in itself: `ag finish` at 27 MiB (93.6 us) matches the
standalone `allgather` collective on the same per-rank size (96-98 us, §4), and
`rs stage` (89.6) is its mirror image. The split simply cannot amortise the
second launch the way one kernel does. Whether that matters is a question for
the ABI layer: at the 9 MiB and 27 MiB DDP buckets it is +7 and +20 us against
an inter-node leg of 60-90 us.

### 10.4 Correctness

`harness.mojo split` verifies against a host reference recomputed from the same
deterministic splitmix64 fill, with the stand-in's constant folded into the
expectation -- so a pair that silently skipped the inter-node step fails, and
does not merely look like a rounding difference. 116 passing cases:

* dtypes float32, bfloat16, float16, int32, int64 at world 8; float32 at world
  4, 2 and **1**; bfloat16 world 5, int64 world 3, int32 world 7 (odd worlds
  take the runtime-`world` kernel instantiation).
* eight sizes per configuration, chosen ragged: 1, 2, 3, 6, 1003, 65537,
  7079424 (27 MiB) and 16777215 elements at fp32, and the corresponding counts
  at the other widths -- i.e. non-multiples of the 16-byte vector width, sizes
  smaller than `world` (so most shards are empty), and `cap-4` bytes.
* each size also re-checked after the timing legs, whose last leg is the pair
  with a zero stand-in -- a plain allreduce, checked as one.
* float32 is checked exactly. bf16/fp16 get a wider band than the fused path
  and have to: the split stores the node-local sum in the *wire* dtype (the
  vendor library reduces that buffer, so it cannot stay in fp32) and rounds
  again on the scaled pull, where the fused kernel rounds once. That is a
  property of the hierarchical algorithm, not of this implementation -- NCCL's
  own hierarchical paths do the same.
* the region's error word is read back after every case (stays 0).

`harness.mojo mix` now rotates **eight** generations per round instead of four:
4-byte one-shot allreduce, 27 MiB two-shot allreduce, 2000-byte broadcast,
8-byte allgather, then a 4-byte split pair and a 27 MiB split pair, each with
the stand-in kernel running between its halves. The split pairs run in place
with a zero stand-in so they stay idempotent like the fused calls and a single
corrupted generation anywhere in the run still survives to the final check.
**200 rounds = 1600 generations**, passing for float32 and bfloat16 at world 8,
float32 at world 4 and bfloat16 at world 5. This is the test that the
buffer-reuse invariant of §10.2(3) actually holds with a foreign kernel writing
the arena mid-collective.

### 10.5 Build

Both new kernels build warning-free for both targets from the same source
(`./build.sh`, which now also dumps `asm/rs_*.asm` and `asm/agf_*.asm`):

| | sm_90a | gfx942 |
|---|---|---|
| `_rs_stage_kernel` payload | 13 `ld.global.v4.b32` / 6 `st.global.v4.b32` | 13 `global_load_dwordx4` / 6 `global_store_dwordx4` |
| `_ag_finish_kernel` payload | 10 / 10 | 10 / 10 |
| flags | 3 `st.release.sys.global` + 4 `ld.acquire.sys.global` | `buffer_wbl2 sc0 sc1` / `buffer_inv sc0 sc1` |
| local/scratch traffic | none | none |

Zero `ld.local` / `scratch_` in either kernel on either target, i.e. the
MOCO-1431 pointer-array trap of §6 was avoided here too (slot addresses are
formed arithmetically inside the unrolled loop).

Reproduce: `sbatch split.sbatch`, then
`python3 split_report.py /home/gabriel/ddp_work/logs/split_<jobid>`.
Raw logs: `/home/gabriel/ddp_work/logs/ccl_split_233937.log` and
`/home/gabriel/ddp_work/logs/split_233937/*.txt` (every rank's CSV).
