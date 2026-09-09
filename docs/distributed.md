# Distributed training (DDP) on the mojo device

The mojo eager device supports `torch.nn.parallel.DistributedDataParallel`
through a c10d backend named `"mojo"`, registered automatically by
`register_mojo_devices()`. Collectives on mojo tensors run over the NCCL C
API — **NCCL** on NVIDIA, **RCCL** on AMD — driven with ctypes from
`torch_mojo_backend/distributed/nccl.py`, no CUDA/ROCm torch build and no
libcudart needed, in keeping with the project's "CPU-only torch install, we
bring the GPU stack" motto:

- NVIDIA: `libnccl.so.2` comes from the `nvidia-nccl-cu12` wheel (a
  dependency of this package).
- AMD: `librccl.so.1` comes from the ROCm install MAX itself already loads
  its HIP runtime from — the one at `$ROCM_PATH` or `/opt/rocm`, or a
  path in `TORCH_MOJO_BACKEND_RCCL_LIB`. Nothing extra to install: every
  ROCm ships RCCL, and taking it from the same install as the HIP runtime
  is what keeps one runtime (one device numbering) in the process. That
  assumes the CPU torch wheel: a ROCm torch wheel loads its own bundled HIP
  runtime next to MAX's, RCCL and the pointer-ownership query would bind to
  one while MAX's buffers belong to the other, and `register_mojo_devices()`
  warns about it (untested; use the CPU wheel).

Collectives on CPU tensors (object collectives, `barrier()`) are served by a
private gloo backend inside the same process group.

## Usage

One process per GPU, launched by torchrun:

```python
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()  # FIRST: pin this rank's GPU before CUDA/MAX initialize

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_mojo_backend import register_mojo_devices

register_mojo_devices()
dist.init_process_group(backend="mojo")

model = MyModel().to("mojo")
model = DDP(model, broadcast_buffers=False)
# training loop as usual; move batches to "mojo" yourself
```

```bash
# single node
uv run torchrun --standalone --nproc-per-node=8 train.py
# multi node (see demo_scripts/nanogpt_ddp.py for a SLURM recipe)
uv run torchrun --nnodes=$NNODES --nproc-per-node=8 \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29500 train.py
```

`use_local_rank_gpu()` gives every rank exactly one visible GPU by slicing
the launcher's whole-allocation list by `LOCAL_RANK` — SLURM-style
`CUDA_VISIBLE_DEVICES=0,...,7` on NVIDIA, `ROCR_VISIBLE_DEVICES=0,...,3` (or
`HIP_VISIBLE_DEVICES`) on AMD — so `"mojo"` is always the right device and
each process binds one CUDA context / HIP device. A list already narrowed to
one entry (one srun task per GPU) is left alone. AMD has two levels, and only
one is narrowed: when `ROCR_VISIBLE_DEVICES` is present it takes the rank's
entry, and a `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` list next to it
(SLURM's gres plugin exports both by default; the HIP runtime reads either as
an index into the HSA-visible set) is rewritten to `0`. Call it before
anything touches the GPU runtime or enumerates MAX devices.

## What works, what to avoid

- `DDP(model)` with the defaults; keep `device_ids=None` (the default for a
  non-CUDA module) and move inputs to the device yourself.
- `broadcast_buffers=False` is recommended when buffers never change (e.g.
  causal masks) — it removes a per-step broadcast.
- **`find_unused_parameters=True` and `static_graph=True` are unsupported**:
  that path needs a pinned-memory allocator PyTorch does not let a
  Python-level PrivateUse1 backend register (`reducer.cpp`
  `all_reduce_local_used_map`).
- `dist.all_reduce/broadcast/all_gather(_into_tensor)/reduce_scatter_tensor/
  send/recv/barrier` and the object collectives all work; `ReduceOp`
  SUM/PROD/MIN/MAX/AVG map to NCCL/RCCL (PREMUL_SUM does not).
- Per-rank randomness: seed the device RNG per rank
  (`torch.mojo.manual_seed_all(seed + rank)`); weight init runs on the CPU
  RNG (`torch.manual_seed`) and DDP broadcasts rank 0's weights anyway.

## Design notes

- **A comm stream overlaps compute.** Collectives run on a dedicated side
  device stream per device (`mojo_device/device_streams.py`): it waits for
  the default stream so every producer kernel comes first, then the
  collective is enqueued. Every tensor a collective touches is fenced with
  `device_streams.record_use` (the backend's `record_stream` analog): its
  eventual stream-ordered free is ordered after the collective on the
  device, because MAX does not fence frees across streams by itself
  (measured; see the memory note in `mojo_device/device_streams.py`).
  `TORCH_MOJO_BACKEND_COMM_STREAM=0` pins collectives to the default stream
  instead (simplest ordering, zero overlap) — also the automatic path for
  collectives needing default-stream copies after the NCCL call. One
  contract carried over from stock torch: `wait()` an async collective
  before reading its result —
  including before exporting it through DLPack.
- **Work objects** wrap already-completed `torch.futures.Future`s (no
  `devices=` — the PrivateUse1 device guard is a stub, and a device-typed
  future would do out-of-bounds bookkeeping for index ≥ 1), so `wait()` is
  a host no-op in both paths.
- **The default stream is ordered after the comm stream lazily**, at the
  first op that touches a buffer a collective read or wrote
  (`mojo_device/comm_fence.py`): the collective records those buffers as
  pending, and a hook in front of every eager op makes the default stream
  wait on the comm stream when it sees one. Host reads no op mediates —
  `torch.mojo.synchronize()`, a default-stream `synchronize()`/`query()`,
  DLPack export, the D2H copy — fence the same way. This is what lets the
  host run ahead: DDP's reducer never blocks on a bucket's future, so it
  keeps enqueuing while allreduces fly, and the fence lands in
  `finalize_backward` where it first reads a reduced bucket — after every
  backward kernel is already enqueued, so overlap is unchanged. Blocking
  the host on those futures instead cost ~2 ms of a 96 ms step, spent
  launching the bucket→grad copies, `clip_grad_norm_` and the optimizer
  against an idle GPU: nanoGPT 124M on 32 H100s (4 nodes, bf16, batch
  32×1024 per rank) went 10.89 → 11.10 M tok/s when it stopped doing so
  (paired A/B/B/A runs, medians of the 10-step windows), closing most of
  the gap to stock CUDA torch's 11.16. Stock `ProcessGroupNCCL` gets there with a
  device-typed future whose `wait()` makes the current stream wait; that
  needs a C++ DeviceGuardImpl for PrivateUse1 which torch does not provide
  and this backend cannot ship.
- **The Python PG replaces the whole process group** (torch ≥ 2.10 behavior),
  so torch cannot compose `cpu:gloo` alongside it; the internal gloo handles
  CPU tensors instead, and `_device_types` stays empty, which routes object
  collectives to CPU — exactly what the internal gloo serves.
- **Comm setup**: rank 0 calls `ncclGetUniqueId` and publishes the 128
  raw bytes through the c10d store (the same rendezvous torchrun already
  provides); every rank then calls `ncclCommInitRank`. Both libraries bind
  the communicator to per-thread runtime state — the current CUDA context
  on NVIDIA (a 4-call libcuda sequence, `nccl.set_current_cuda_device`),
  the current HIP device on AMD (`hipSetDevice`, `mojo_device/hip_peer.py`)
  — and DDP invokes the PG from the autograd thread, so this is re-asserted
  per thread. Which physical GPU a mojo tensor lives on is read off its
  pointer (`cuda_peer`/`hip_peer.device_ordinal`), never assumed from an
  ordinal. The library is picked once per process from the device api of
  the first mojo tensor a collective sees (`Device.api` is `"cuda"` or
  `"hip"`).
- Errors raised inside collectives print a full traceback to stderr before
  propagating (`_loud`): an exception escaping into the autograd engine on
  this backend can otherwise kill the process with no Python traceback.

## Cluster notes (SLURM, IB — NVIDIA)

- ptxas needs no configuration: the package defaults
  `MODULAR_NVPTX_COMPILER_PATH` to the CUDA 12.8 ptxas of the
  `nvidia-cuda-nvcc-cu12` wheel it depends on, whose cubins load on r570+
  drivers (`torch_mojo_backend/_ptxas.py`). Export the variable yourself
  only to use another ptxas.
- `NCCL_DEBUG=WARN` (or `INFO` during bring-up) is the first knob for
  diagnosing init hangs; on multi-homed nodes set `NCCL_SOCKET_IFNAME` if
  NCCL's interface auto-detection picks a dead interface.
- First-run kernel builds: the JIT compile pool sizes itself per process
  from whole-node resources, so 8 cold ranks can oversubscribe a node.
  The kernel caches (`__mojocache__`, `~/.modular`) are shared over NFS, so
  a one-off single-process warmup run (or just letting step 1 be slow once)
  populates them for every node.

## Cluster notes (SLURM, Slingshot — AMD MI300A)

Verified on CINES's Adastra (4 × MI300A per node, ROCm 6.4.3, RCCL 2.22.3).

- **Install the CPU wheel of torch on AMD machines**
  (`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`).
  The default PyPI wheel is the CUDA build, and it maps ~3 GB of NVIDIA
  libraries the process never uses; the HIP runtime walks every mapped
  shared object at each kernel load, so every first-use kernel load pays
  for them. Measured with nanoGPT 124M under DDP on 4 MI300A: the first
  training step took 14.7–18.5 s with torch 2.11+cu130 and 1.0 s with
  torch 2.11+cpu, and steady state was 6% faster too (680k vs 640k tok/s).
  `register_mojo_devices()` warns when it sees a CUDA torch build next to
  HIP devices.
- SLURM's GPU binding sets `ROCR_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`
  (`0,1,2,3` for one task with `--gpus-per-task=4`); `use_local_rank_gpu()`
  slices it. Launch one `torchrun` per node with `--nproc-per-node` equal to
  the node's GPU count, exactly as on NVIDIA.
- MAX dlopens `libamdhip64.so`/`libhsa-runtime64.so` from `$ROCM_PATH` or
  `/opt/rocm`; RCCL is taken from the same place. If the site's ROCm is not
  where MAX looks, set `ROCM_PATH` (or `module load rocm`) in the sbatch
  script so every rank gets it. A `GLIBCXX_3.4.30 not found` from
  `max._core` means the system libstdc++ is older than GCC 12: put a newer
  one on `LD_LIBRARY_PATH` (on Cray systems `/opt/cray/pe/gcc-libs`).
- Multi-node over Slingshot needs the site's libfabric RCCL plugin
  (`module load aws-ofi-rccl`, which puts `librccl-net.so` on the path);
  without it RCCL falls back to TCP sockets. `NCCL_DEBUG=INFO` shows
  `NET/OFI Selected Provider is cxi` when it took.
- RCCL's knob for MI300 is `NCCL_MIN_NCHANNELS`; the site recommends 42 for
  up to 4 APUs and 32 beyond. The single-node numbers above were taken
  with the defaults.
- Put the checkout, its `.venv` and `__mojocache__` on the fast parallel
  filesystem (scratch on Adastra, not work): a first-use kernel load makes
  the HIP runtime walk every mapped shared object, and with the venv on a
  slow filesystem each one costs tens of seconds per rank.
- **Memory on an APU.** MAX's default device allocator reserves ~115 GB
  (≈90% of the MI300A's 128 GB) per process at its first allocation, and on
  an APU that is the host's RAM: four ranks leave ~15 GB of a 512 GB node
  for everything else. Two consequences were measured: four ranks compiling
  kernels at once (and once a plain 4-rank run, on a node with less free
  memory) were OOM-killed, and the reservation evicts the page cache
  between import and the first training step, so the first step re-reads
  every kernel extension and every mapped library from Lustre — 12 to 18 s
  in about half the runs, ~1 s in the others. Modular's on-demand allocator
  fixes both: with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` a process
  holds 4 GB instead of 115, the first step is a steady 1.5 s and the
  steady state is unchanged. Its one defect with MAX 26.5 + ROCm 6.4.3 is a
  segfault at interpreter exit, inside ROCr's `Runtime::~Runtime` tearing
  down the VMM mappings from HIP's atexit handler (a pure-MAX script
  reproduces it; dropping every device reference first does not help), so
  a script that selects it must end with `os._exit(0)` once its own cleanup
  (`destroy_process_group`, checkpoint) is done — `demo_scripts/nanogpt_ddp.py`
  does. Without the knob, warm the kernel cache with a single process first
  (`--nproc-per-node=1`, a couple of steps) and expect the bimodal first
  step. Capping the HIP heap instead (`GPU_MAX_HEAP_SIZE=30`) is not an
  option: MAX's allocator becomes ~40x slower.

### Measured: nanoGPT 124M, bf16 autocast, batch 12×1024 per rank, 20 steps

Wall time of the training loop (after model, DDP and optimizer
construction), the first step included; three interleaved runs each, on
Adastra MI300A nodes, ours with the CPU torch wheel and
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`, stock torch 2.9.1+rocm6.4:

| | ours | stock ROCm torch |
|---|---|---|
| 1 node, 4 ranks: 20 steps | 3.1 / 3.2 / 3.3 s | 10.3 / 10.1 / 10.8 s |
| of which step 1 | ~1.5 s | ~8.7 s |
| steady state | 687–698k tok/s | 660k tok/s |
| whole process (imports to exit) | 12–13 s | 42–56 s |
| 2 nodes × 2 ranks over Slingshot: 30 steps | 4.1 / 4.6 s | 10.7 / 10.4 s |
| of which step 1 | 1.5 / 1.4 s | 8.1 / 7.6 s |
| steady state (steps 3–30) | 80 / 84 ms per step, an occasional 100–180 ms step | 79 ms per step, within 1 ms |

The two stacks print identical losses at every logged step. The 2-node
runs used the default GPU-Direct transport for both; per-step times come
from the tokens/s the demo prints for each step (its elapsed column has
0.1 s resolution).

- **Multi-node status.** On one node pair (a1003/a1004) our ranks failed in
  `ncclCommInitRank` with an RCCL internal error from the libfabric
  plugin's Connect step, after topology setup; the stock legs on the same
  pair worked. On the next pair (a1016/a1019) every configuration passed
  the full collective and DDP-parity checks across nodes — the default
  GPU-Direct path, `NCCL_NET_GDR_LEVEL=0` and `NCCL_NET=Socket` alike — so
  the failure was not reproduced and looks node-pair specific (that pair
  also held a job stuck in COMPLETING). If it recurs, `NCCL_NET_GDR_LEVEL=0`
  and `NCCL_NET=Socket` are the fallbacks, in that order. The 2-node
  numbers above are with 2 ranks per node because of the memory paragraph
  above (4 ranks per node work on the single node once the kernel cache is
  warm, and with the VMM allocator without caveat).

```bash
#!/bin/bash
#SBATCH --account=<account> --constraint=MI300 --nodes=2 --exclusive --time=1:00:00
module purge
module load aws-ofi-rccl   # multi-node only
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/cray/pe/gcc-libs:/opt/rocm/lib:${LD_LIBRARY_PATH}
export NCCL_DEBUG=WARN
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1   # APU: see the memory paragraph
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
srun --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=96 -- \
    uv run torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29500" \
    --rdzv-id="$SLURM_JOB_ID" demo_scripts/nanogpt_ddp.py ...
```

## Mojo collectives (experimental): `TORCH_MOJO_BACKEND_CCL=mojo`

An in-repo replacement for NCCL/RCCL's intra-node collectives, written in Mojo
and exposed through **NCCL's own C ABI**: `torch_mojo_backend/distributed/mojoccl/`
builds `libmojoccl.so` on first use (into the eager kernels' `__mojocache__`,
same lock/atomic-rename machinery) and `nccl.py` dlopens it instead of
`libnccl.so.2` when `TORCH_MOJO_BACKEND_CCL=mojo`. `process_group.py` is
unchanged; NCCL/RCCL stays the default. Design and measurements:
`docs/mojo_collectives_feasibility.md` (study) and
`docs/mojo_collectives_kernel_results.md` (kernels).

Scope, deliberately narrow — it is an experiment showing Mojo can write
NCCL-class collectives, not a general library:

- 2–8 ranks per node (`MAX_WORLD = 8`), up to 16 nodes, one process per GPU
  under torchrun; the inter-node hop is Mojo over libibverbs — see
  "Multi-node" below;
- `ncclAllReduce` (float32/float16/bfloat16/int32/int64, SUM and AVG),
  `ncclBroadcast` and `ncclAllGather` (every dtype, byte-granular);
  `ncclReduce`, `ncclReduceScatter`, `ncclSend`, `ncclRecv` return
  `ncclInvalidUsage`, so DDP works and anything needing them does not;
- the rendezvous is a TCP socket that `ncclGetUniqueId` opens on rank 0;
  the 128-byte `ncclUniqueId` carries its address, port and a random magic
  (NCCL's shape), and `ncclCommInitRank` runs three relayed all-gathers over
  it (host identity, IPC handle plus IB connection data, barrier);
- every rank owns one shared staging region (`MOJOCCL_REGION_MB`, default
  256 MiB, multiple of 4 KiB; larger requests are chunked). MAX's own
  allocations cannot be shared across processes (§5.6 of the study), which
  is why the kernels stage through this region; the push / local-reduce /
  pull design makes the staging free. The region is either a `cuMemAlloc`
  block shared with legacy IPC or, where NVSwitch multicast is available,
  VMM memory bound to a multicast object — see "NVLS" below;
- a rank that stops responding makes its peers time out after 60 s inside the
  kernel and `ncclCommGetAsyncError` reports it; there is no abort path.

Measured on 8×H100 SXM through the process group (wall over 20 launches,
median of 5, interleaved legs; NCCL 2.31.2 NVLS for comparison), **unicast
kernels only** — i.e. `MOJOCCL_NVLS=0`, which is what everything below the
48 MiB crossover runs anyway: 1 MiB 29 vs 31 µs, 9 MiB 70 vs 92, 27 MiB (the
DDP bucket) 164 vs 181, 168 MiB 988 vs 756, 512 MiB 2.99 vs 2.15 ms. The last
two rows are where the multicast path takes over — next subsection. AMD: the
same source cross-compiles for gfx942 (the one vendor gate is a release fence
on AMD's `s_barrier`), but it has not run on an MI300A yet.

### NVLS: the large sizes go through the switch

The 168 and 512 MiB rows above are the **unicast** ceiling (~310 GB/s per
direction over NVSwitch): a push/reduce/pull allreduce moves
`2(world-1)/world × bytes` per GPU each way, and no schedule beats that
while every byte travels point to point. NCCL closes it with NVSwitch
multicast, and so does this library now.

A single-node communicator whose devices all report
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` builds its region as VMM memory
bound to a per-node multicast object, and routes allreduces of
`MOJOCCL_NVLS_MIN_MB` (48 MiB) or more through `multimem.ld_reduce` /
`multimem.st`: rank r pulls its 1/world slice through the switch, which sums
the `world` contributions and returns one value, and pushes the result back
into all `world` regions in one instruction. NVLink traffic falls to `bytes`
per GPU each way — 1.75× less at world 8 — paid for with ~1.5× the HBM
traffic, because the user's tensors are MAX-allocated and cannot be bound to
a multicast object, so every byte is staged in and out. That trade is why
the path has a size floor: 48 MiB is the measured crossover, sharp (4% the
wrong side at 40 MiB, 4% the right side at 48) and the same for fp32 and
bf16. int32/int64 stay unicast.

**Bring-up** (`vmm.mojo`, NCCL's `src/transport/nvls.cc` sequence). The
node's local rank 0 calls `cuMulticastCreate` and exports the object as a
POSIX file descriptor; the fd travels to its node-mates over an AF_UNIX
`SOCK_DGRAM` socket as an `SCM_RIGHTS` control message, which is what NCCL
does and, unlike `pidfd_getfd`, works whatever
`/proc/sys/kernel/yama/ptrace_scope` says. Every rank then
`cuMulticastAddDevice`s its own device, **barrier**, `cuMemCreate`s its
physical memory and `cuMulticastBindMem`s it at multicast offset 0 — so one
multicast address covers the node's eight distinct allocations — and maps it
**twice**: a multicast VA that only `multimem.*` may touch, and a plain VA
that the unicast kernels, the staging copies and the flag spin use. Peers are
imported with `cuMemImportFromShareableHandle` over the same sockets, because
legacy `cuIpcOpenMemHandle` cannot open VMM memory. Then **barrier**: no rank
issues a multimem instruction before every rank has mapped. The whole
sequence costs 150–230 ms once per communicator, dominated by
`cuMulticastBindMem` and the mappings.

The socket name is derived from the unique id's magic and the local rank
(`/tmp`, or `MOJOCCL_SOCKET_DIR`), so the rendezvous carries no extra round.
`tests/multinode/selftest/fd_exchange.mojo` runs that transport over ordinary
file descriptors with no GPU and no multicast hardware — the msghdr /
cmsghdr / sockaddr_un structs are laid out by hand over `UInt64` words
(`std.ffi` has no C-struct ABI) and a wrong offset does not fail loudly.

**Memory.** A multicast object's size must be a multiple of what
`cuMulticastGetGranularity` reports, and on H100 that is **512 MiB** for
`RECOMMENDED` against **2 MiB** for `MINIMUM`. NCCL uses RECOMMENDED; this
uses MINIMUM, so that `MOJOCCL_REGION_MB` keeps meaning what it says — at
RECOMMENDED the default region (`128 KiB + 2 × 256 MiB`) rounds up to a 1 GiB
allocation per rank and even a deliberately tiny test region costs 512 MiB,
while at MINIMUM the same region is 514 MiB. `MOJOCCL_NVLS_GRANULARITY=rec`
asks for NCCL's choice back.

The NVLS kernel stages **one** buffer, not two, so it uses the whole `2 × cap`
arena rather than a half: a 512 MiB allreduce is one launch on the default
256 MiB region. That is safe for the same reason a broadcast may use the whole
arena — the start barrier below.

**Fallback is a decision, not a recovery.** The capability travels in the
first bootstrap round, before anything is allocated: every rank contributes
"my device reports multicast and `MOJOCCL_NVLS` is not 0", rank 0 additionally
does a real `cuMulticastCreate` of a granularity-sized object and releases it
(88 µs, and it is where a broken fabric-manager setup shows up), and every
rank ANDs the whole column. One "no" — AMD, pre-Hopper, no NVSwitch, a rank
with `MOJOCCL_NVLS=0` — and the whole communicator builds the `cuMemAlloc`
region with legacy IPC exactly as before, with no half-built state to unwind.
A failure *after* that point is reported rather than papered over, and the
message names `MOJOCCL_NVLS=0`.

**The kernel** (`nvls_kernels.mojo`) is the prototype's split-grid schedule:
the low 25% of the blocks only drive the switch and the rest only drive HBM,
the message is cut into `clamp(bytes/4, 21 MiB, 86 MiB)` chunks, and chunk
c's reduction runs at the same time as chunk c+1's copy-in and chunk c-1's
copy-out. Two things about it are worth knowing before touching it:

* **The barrier is a full barrier**, not the block-index-matched one every
  other kernel in this library uses. There, block b only ever consumes bytes
  block b of a peer produced; here the reduce phase reads a contiguous slice
  that *every* block of every peer helped stage. The index-matched version
  passes every small case and fails from n = 65537 up. It is two levels: a
  device-scope arrival counter, then one
  `multimem.red.release.sys.global.add.u64` from the last block to arrive,
  which posts that GPU's arrival to all eight counters in one instruction; the
  wait is on the plain mapping, so it costs no fabric traffic.
* **A full barrier needs the whole grid resident** or it deadlocks. The grid
  is therefore `min(216, 2 × SM count)` blocks of 256 threads with
  `nvvm.minctasm=2`, and 216 was fitted on H100's 132 SMs.
* **It opens with a start barrier**, before a byte of staging is written, for
  the same reason every other kernel in `collectives_kernels.mojo` does: the
  arena is shared scratch, and a broadcast or an all-gather retiring just
  before this kernel has its peers still *reading* the bytes the copy-in is
  about to overwrite. One barrier per call, not per chunk.

Everything above is behind a compile-time sm_90+ gate (`multimem` exists
nowhere else and RCCL has no equivalent) and a runtime capability check; the
gfx942 and sm_90a cross-compiles both stay clean.

**Checking it.** `tests/ddp_worker.py stress` covers the path at 8 ranks and
256 MiB (every dtype × ragged size × SUM/AVG × in-place/out-of-place);
`tests/nvls_check.py` is the same shape of check at world 2 and 4, which
`ddp_worker` never runs, over sizes that straddle both the dispatch threshold
and the staging arena:

```bash
TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=2 tests/nvls_check.py
TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=8 tests/ddp_worker.py stress
# the numbers below: one leg per configuration, palindromic order
TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=8 ar_bench_gpt2.py
MOJOCCL_NVLS=0 TORCH_MOJO_BACKEND_CCL=mojo torchrun --nproc-per-node=8 ar_bench_gpt2.py
```

**Results**, 8×H100 SXM on one node (job 234314, `cl02s01dgx05`),
`ar_bench_gpt2.py` through the process group, medians in µs. Six legs in
palindromic order — NCCL, unicast, NVLS, NVLS, unicast, NCCL — so a clock or
thermal ramp cancels to first order; each column is the mean of its two legs:

| dtype | MiB | unicast | **NVLS** | NCCL | NVLS/unicast | NVLS/NCCL |
|---|---|---|---|---|---|---|
| fp32 | 9 | 70 | 70 | 94 | 1.00 | 0.74 |
| fp32 | 27 (DDP bucket) | 165 | 164 | 182 | **1.00** | 0.91 |
| fp32 | 168 (tail bucket) | 988 | **799** | 756 | **0.81** | 1.06 |
| fp32 | 512 | 2991 | **2218** | 2154 | **0.74** | 1.03 |
| bf16 | 9 | 70 | 70 | 91 | 1.00 | 0.77 |
| bf16 | 27 | 165 | 165 | 179 | **1.00** | 0.92 |
| bf16 | 168 | 994 | **798** | 745 | **0.80** | 1.07 |
| bf16 | 512 | 2988 | **2213** | 2126 | **0.74** | 1.04 |

Everything at or below 27 MiB is byte-identical code and reads identical,
which is the point of the crossover: the dispatch buys the tail bucket 19%
and the 512 MiB bucket 26% and costs the DDP bucket nothing. Against NCCL the
tail bucket goes from 1.31× to 1.06× and 512 MiB from 1.39× to 1.03×. 1 MiB
is below this bench's noise floor (per-leg medians 27–59 µs on every
configuration, NVLS or not) and is left out of the table.

A seventh leg measured `MOJOCCL_NVLS_GRANULARITY=rec` — NCCL's 512 MiB
multicast objects instead of the default 2 MiB ones — at 806/2213 fp32 and
798/2213 bf16 for the two large sizes: **the same within noise**, and the
2 MiB objects allocate 514 MiB per rank against 1 GiB. That granularity had
never been measured before (the prototype only ever used RECOMMENDED).

### Multi-node

The inter-node hop is Mojo too: no vendor collective library anywhere.
An allreduce on a communicator spanning N nodes runs, per chunk: the
intra-node reduce-scatter (`reduce_scatter_stage`) leaves every rank its
shard of the node-reduced bucket in its own `stage_out`; each rank
RDMA-writes that shard to the counterpart rank (same `local_rank`) on every
other node and receives theirs into the `network` area of its region; a
small kernel sums the N−1 inbox shards into the shard; the intra-node
all-gather (`allgather_finish`) then pulls the globally reduced shards into
the user output, scaled for AVG. Broadcast and all-gather use the same RDMA
path with a simpler schedule (root's node fans out to its counterparts, then
intra-node; node blocks exchanged, then placed by global rank) and stay
unpipelined — they run at DDP init, not in the step. Single-node
communicators keep the fused intra-node path and never touch IB.

**The three phases overlap.** The bucket is cut into K chunks and issued on
the one comm stream, with at most `PIPE_ARENAS` chunks alive:

```
RS(0) rel(0)  RS(1) rel(1)  RS(2) rel(2)  RS(3) rel(3)
              wait(0) add(0) AG(0)  RS(4) rel(4)
              wait(1) add(1) AG(1)  RS(5) rel(5)  …
```

so the proxy exchanges chunk k while the GPU reduce-scatters later chunks
and all-gathers earlier ones. `K = sqrt(bytes / (local_world × 640 KB))`,
capped at 16 by choice and raised from below by geometry when a chunk would
not fit — the square root is of the 16 µs an extra chunk costs (two more
launches and two more 8-way start barriers) against the 40–45 GB/s the RDMA
runs at. It gives K = 1 up to ~10 MiB, 2 at the 27 MiB DDP bucket, 5 at
168 MiB and 10 at 512 MiB, and keeps a chunk's shard above 1 MiB without a
second clause.

Two things make concurrent chunks safe, and neither is stream order across
ranks. **The staging arena is replicated.** A multi-node region is
`PIPE_ARENAS` complete arenas — each its own signal area and its own
`[stage_in | stage_out]` of `cap/PIPE_ARENAS` — followed by one cap-sized
network area; chunk k uses arena `k % PIPE_ARENAS`, so concurrent chunks
cannot collide in the push slots or in the shard, and that arena's own start
barrier is what orders chunk k+`PIPE_ARENAS` behind chunk k's pulls, the
invariant one arena already had. `collectives_kernels.mojo` is untouched:
the split kernels take a shifted base and a smaller cap, nothing more. The
staging total is `2 × cap` either way, so the region is the size it always
was. **The inbox is reused only against a credit** — see the transport
below. `PIPE_ARENAS` (4) and the derived `INBOX_SLOTS` (5) are source
constants in `mojoccl.mojo`, not environment variables: they are part of the
wire layout and every rank has to agree on them.

The chunk cap is now one arena, and the inbox slot group, rather than the
whole region: 64 MiB at 2 nodes and 36 MiB at 8 with the default 256 MiB
region, so the large sizes are chunked by geometry as well as by choice.
`tests/multinode/selftest/geometry_test.mojo` sweeps that arithmetic over
regions of 1 MiB–1 GiB, `local_world` 1–8 and 2–16 nodes.

**Transport** (`torch_mojo_backend/distributed/mojoccl/{ibverbs,internode,
internode_kernels,bootstrap}.mojo`): libibverbs is dlopened; setup calls are
symbols, the data path (`ibv_post_send`/`post_recv`/`poll_cq`) is reached
through the `ibv_context_ops` table at the header's offsets, as NCCL's
`ibvwrap` does. One RC queue pair per remote node, attributes borrowed from
NCCL's `net_ib/connect.cc` (cited in the source); one `ibv_reg_mr` of the
whole region (`nvidia_peermem`; dmabuf is not implemented), with relaxed
ordering through `ibv_reg_mr_iova2`; each shard is one
`IBV_WR_RDMA_WRITE_WITH_IMM`, empty recv work requests exist only so the
immediate produces a completion, and a self-QP `IBV_WR_RDMA_READ` flush
orders the payload in GPU memory behind the completion that landed in host
memory. Every exchange is all-to-all — a rank whose shard is empty (7 of 8
ranks on DDP's 4-byte AVG allreduce) still posts a 16-byte placeholder — so
that an arrival tally of N−1 is what completes one.

**Flow control is explicit credits.** The second half of the network area is
carved once into `INBOX_SLOTS` fixed groups and exchange `e` lands in
`e % INBOX_SLOTS`; a peer may write that group again only once every
receiver has released it, published as a cumulative "consumed through e"
counter in a 4-byte `RDMA_WRITE_WITH_IMM` whose immediate carries a credit
bit — NCCL's head/tail pair in miniature
(`nccl:src/transport/net.cc`). This replaces a double-buffer-by-parity
argument that derived reuse safety from stream order ("peer B cannot post
e+2 before receiving my e+1 data, which I send only after my own add kernel
for e"), which holds for exactly one exchange in flight and breaks under the
pipelined schedule. The credit needs no kernel of its own: it rides on this
rank's next request as `credit_upto`, the number of consumer kernels already
enqueued ahead of that request, and when the proxy observes the request that
kernel has run, so every kernel enqueued before it has completed.
`INBOX_SLOTS = PIPE_ARENAS + 1` is what keeps the rank furthest behind from
ever waiting on a credit.

The GPU/network hand-off is a progress thread per rank driven through a
pinned, device-mapped mailbox: a one-thread kernel releases the exchange
into `MB_REQUEST`, the thread posts it, and a second one-thread kernel spins
on `MB_DONE` until the thread has seen the N−1 arrivals and flushed. Several
exchanges live between the two, so the thread is one non-blocking step
function (`ib_drive`, shared with the `MOJOCCL_IB_PROXY=0` callback and the
GPU-free self-tests) that retires exchanges in sequence order — RC ordering
is per queue pair, so arrivals are not ordered across peers — and pipelines
the flush read the same way. The thread spins only while something is
outstanding; idle, it yields and then sleeps in `MOJOCCL_IB_PROXY_IDLE_US`
steps, because a thread spinning between exchanges competed with the
host-bound Python dispatch thread and cost ~20% of end-to-end training
throughput at 16 ranks. A `cuLaunchHostFunc`/`hipLaunchHostFunc` stream
callback does the same job behind `MOJOCCL_IB_PROXY=0` and costs about
480 µs of fixed driver latency per exchange on this cluster (2.6× slower at
the DDP bucket), which is why the thread is the default. HCA choice: the
longest common `/sys/devices` prefix between the GPU's and the HCA's PCI
paths, ties by `local_rank`; only ACTIVE InfiniBand ports (the RoCE ports
are skipped). Addressing is LID-only, so one IB subnet.

| variable | default | controls |
|---|---|---|
| `MOJOCCL_SOCKET_IFNAME` | first UP non-loopback IPv4 interface with a default route (`bond0` here) | interface whose address rank 0 publishes in the unique id; one name, no lists |
| `MOJOCCL_BOOTSTRAP_TIMEOUT_S` | 120 | deadline for every bootstrap socket wait |
| `MOJOCCL_IB_HCA` | affinity choice | exact HCA name to use instead (`mlx5_4`) |
| `MOJOCCL_IB_TIMEOUT_S` | 60 | how long a rank waits for its peers' shards before latching an error |
| `MOJOCCL_IB_PROXY` | 1 | `0`: stream host callback instead of the progress thread |
| `MOJOCCL_IB_PROXY_IDLE_US` | 20 | sleep quantum of the idle progress thread (it spins only during an exchange) |
| `MOJOCCL_IB_PROXY_CPU` | unset | pin the progress thread to this CPU |
| `MOJOCCL_IB_RELAXED_ORDERING` | 1 | `0`: plain `ibv_reg_mr` |
| `MOJOCCL_IB_TRACE` | 0 | `1`: one line per rank at destroy — HCA, port, peers, slot groups, exchanges, credit stalls, mean µs posting / in flight / flushing |
| `MOJOCCL_REGION_MB` | 256 | staging size; single node `[signal \| stage_in cap \| stage_out cap]`, multi-node `PIPE_ARENAS` arenas of `cap/PIPE_ARENAS` halves plus a cap-sized network area. Must match on every rank — `ncclCommInitRank` checks it. On an NVLS region the whole thing is rounded up to the multicast granularity (512 MiB on H100) and the halves grow into the rounding |
| `MOJOCCL_NVLS` | 1 | `0`: no multicast region and no NVLS kernel, on every rank of the communicator (it is ANDed across ranks) |
| `MOJOCCL_NVLS_MIN_MB` | 48 | single-node allreduces at or above this go through the switch; below it the unicast kernels keep the traffic. The default is the measured crossover |
| `MOJOCCL_NVLS_GRANULARITY` | `min` | `rec`: size the multicast object with `CU_MULTICAST_GRANULARITY_RECOMMENDED` (NCCL's choice, 512 MiB objects on H100) instead of `MINIMUM` (2 MiB) |
| `MOJOCCL_SOCKET_DIR` | `/tmp` | where the node-local AF_UNIX sockets that carry the VMM/multicast file descriptors are bound |

Limits and failure modes: 8 ranks per node, 16 nodes; more than one node
with no ACTIVE InfiniBand port fails `ncclCommInitRank` with "no ACTIVE
InfiniBand port found" (so Slingshot on Adastra is not covered; a
libfabric/cxi transport would be a second backend); a peer that stops
responding is reported through `ncclCommGetAsyncError` after
`MOJOCCL_IB_TIMEOUT_S`; a stale unique id (tag `MOJOCCL2`) is rejected with
a clear message.

**Requirements.** rdma-core/libibverbs on the nodes (here MLNX OFED 24.10),
GPUDirect RDMA through `nvidia_peermem`, active InfiniBand ports reachable
between every pair of nodes, a routable interface for the TCP bootstrap.

**Running the two-node job.** `tests/multinode/run_two_node_checks.sbatch`
is a 16-rank (2 nodes × 8 GPU) SLURM job: `tests/ddp_worker.py`
(`collectives`/`ddp_parity`/`stress`) under NCCL and under mojoccl, the
allreduce device-time bench (`ar_bench_gpt2.py`) in ABBA order, and a
40-step nanoGPT DDP run under both. `RUN_MOJO=0` keeps only the NCCL legs.
`tests/multinode/summarize.py <job log>` turns a log into the tables below.
`tests/multinode/selftest/` holds five GPU-free self-tests — the bootstrap,
the RDMA transport, the pipelined transport with its credit protocol, the
region geometry, and the `SCM_RIGHTS` fd transport the NVLS bring-up uses
(the last two need no IB either). The first three run on a host with IB HCAs
and no GPU, such as the login node, with `MOJOCCL_IB_PROXY=0`; they caught
six bugs before any GPU time was spent.

**Results**, 16 ranks on 2×8 H100, `ar_bench_gpt2.py` through the process
group, medians in µs. The pipeline against the commit before it, **in one
job on one node pair**, legs in ABBA order (before, after, after, before) so
a clock or fabric drift cancels to first order (job 234237,
`cl02s01dgx05` + `cl02s02dgx23`); K is the chunk count the rule above picks:

| MiB | K | fp32 before | fp32 after | | bf16 before | bf16 after | |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 107 | 107 | 1.00 | 100 | 113 | (noise, see below) |
| 9 | 1 | 187 | 187 | 1.00 | 186 | 187 | 1.00 |
| 27 (DDP bucket) | 2 | 342 | 282 | **0.82** | 345 | 274 | **0.80** |
| 168 (tail bucket) | 5 | 1606 | 1192 | **0.74** | 1592 | 1186 | **0.75** |
| 512 | 10 | 4762 | 3541 | **0.74** | 4739 | 3532 | **0.75** |

1 MiB is at the noise floor of this bench (per-leg medians 90–116 µs either
side, minima 90.3 vs 91.8 fp32 and 90.5 vs 91.4 bf16); K is 1 there and at
9 MiB, so those two sizes run the code they always did. NCCL 2.31.2 on the
same node pair (job 234235) reads 175 / 267 / 941 / 2359 µs fp32 and 168 /
263 / 950 / 2392 bf16 at 9 / 27 / 168 / 512 MiB, so the pipeline takes the
DDP bucket from 1.28× NCCL to 1.06× and the tail bucket from 1.71× to
1.27×. `MOJOCCL_IB_TRACE=1` over the whole bench: 0.15 µs posting, 2.8 µs
flushing, and 160–215 µs per exchange between release and retirement (which
now includes the time later chunks spend queued behind earlier ones);
4% of exchanges waited on a credit.

**Absolute numbers here are node-pair-specific and the table above is the
only fair comparison.** An earlier run of the same bench on a different pair
(job 234072) read 60 / 119 / 278 / 1532 / 4635 µs for the code in the
"before" column; two pairs measured since put that same commit at 107 / 187 /
342 / 1606 / 4762. NCCL is nearly identical on all of them (267 ± 2 µs at
the bucket), which is the tell: it pipelines its network hop, so its numbers
barely move with the fabric's per-message latency, and the unpipelined
hierarchical allreduce's numbers moved a lot. Fitting an exchange latency to
the 9 MiB point (where K is 1, and the shard's 1.1 MiB is 28 µs of wire at
the measured 40–45 GB/s) gives ~19 µs on the fast pair and ~87 µs on the
others. Hiding that latency is exactly what the pipeline does, and it is why
the gain is larger here than the exposed-transfer arithmetic alone predicts.

Splitting harder does not help: at `PIPE_SPLIT_UNIT = 320_000` (K of 3 / 8 /
14 instead of 2 / 5 / 10) the 27 MiB bucket is unchanged and 168 and 512 MiB
regress to 1291 and 3648 µs (job 234242, same ABBA design). The extra
exchange's latency is not fully hidden, so past the point where the network
is covered an extra chunk only buys launches.

`collectives`, `ddp_parity` and `stress` pass at 16 ranks under both
libraries (job 234229, `cl02s01dgx06` + `cl02s04dgx02`), including `stress`'s
200 rounds of interleaved 4-byte / 27 MiB / broadcast / allgather
collectives and its 256 MiB messages, which the pipeline cuts into 7 chunks.
The `MOJOCCL_IB_PROXY=0` fallback passes `collectives` too: it cannot
overlap (the callback blocks the stream) but the schedule and the credit
protocol are correct on it. nanoGPT-124M DDP at 16 ranks reaches the same
losses; end to end its median step throughput was 4173k tok/s against NCCL's
4230k in the same job, with a second mojoccl leg at 3704k — the run-to-run
noise on these shared nodes is still as large as the gap.

What is left at the large sizes is the intra-node half, not the network: the
split reduce-scatter/all-gather pair is ~1004 µs at 168 MiB on one node
against NCCL's 751 µs NVLS multicast, and the pipeline's 1192 µs is 1.19×
that floor.

**The multi-node path deliberately stays unicast**, even on a cluster where
the single-node one takes the multicast route. Three reasons, in order of
weight:

1. *The chunking fights it.* NVLS wins only from 48 MiB up, and the pipeline
   cuts a 168 MiB bucket into K = 5 chunks of 33.6 MiB — below the crossover.
   Forcing chunks above it means K = 3, and by the model the chunk rule is
   built on (total ≈ node-local + network/K, which reproduces the measured
   1192 µs at K = 5 from a node-local 1004 µs) that trades ~125 µs of extra
   exposed network for the ~45 µs NVLS saves node-locally: 1272 µs against
   1192. At 8 nodes the geometry caps a chunk at 36 MiB, so the question does
   not even arise. Only 512 MiB at 2 nodes (K = 10, chunks of 51.2 MiB, just
   over the crossover) would gain, and only ~4%.
2. *It would need RDMA out of VMM memory.* The inter-node write reads
   straight out of an arena's `stage_out`, so the staging has to be inside
   the registered MR — and `ibv_reg_mr` on a `cuMemMap`'d VA returned NULL on
   this cluster, with no dmabuf fallback
   (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0 on every device, so
   `ibv_reg_dmabuf_mr` is unavailable). That probe deserves one more careful
   look (it used `ibv_get_device_list()[0]` rather than the affine HCA and
   never read `errno`) before anything leans on it.
3. Consequently a multi-node communicator would pay 150–230 ms of multicast
   bring-up for nothing, so it does not build a multicast region at all:
   multi-node allocation is byte for byte what it was, which the numbers
   confirm. Re-running this bench at 16 ranks after the NVLS work landed
   (job 234315, `cl02s01dgx24` + `cl02s02dgx23`, the same ABBA design) reads
   285 / 1185 / 3545 µs fp32 and 283 / 1184 / 3534 bf16 at 27 / 168 / 512 MiB
   against the 282 / 1192 / 3541 and 274 / 1186 / 3532 recorded above — within
   1% at every size, and `collectives`, `ddp_parity` and `stress` all pass at
   16 ranks.

The unmeasured half of point 1 is the NVLS *split* kernels themselves — a
multicast reduce-scatter and a multicast all-gather were never written, so
the ~45 µs above is estimated from the prototype's phase timings (copies
275 µs and switch 707 µs at 168 MiB) and not measured. If the chunk cap ever
rises above the crossover for the shapes that matter, this is the experiment
to run.
