# Distributed training (DDP) on the mojo device

The mojo eager device supports `torch.nn.parallel.DistributedDataParallel`
through a c10d backend named `"mojo"`, registered automatically by
`register_mojo_devices()`. Collectives on mojo tensors run over **NCCL**,
loaded directly from the `nvidia-nccl-cu12` wheel with ctypes — no CUDA
torch build and no libcudart needed, in keeping with the project's
"CPU-only torch install, we bring the GPU stack" motto. Collectives on CPU
tensors (object collectives, `barrier()`) are served by a private gloo
backend inside the same process group.

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

`use_local_rank_gpu()` gives every rank exactly one visible GPU (slicing a
SLURM-style `CUDA_VISIBLE_DEVICES=0,...,7` list by `LOCAL_RANK`), so `"mojo"`
is always the right device and each process binds one CUDA context. Call it
before anything touches CUDA or enumerates MAX devices.

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
  SUM/PROD/MIN/MAX/AVG map to NCCL (PREMUL_SUM does not).
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
  collectives needing default-stream copies after the NCCL call. Both paths
  drain the host-side kernel-call queue first so producers are actually on
  a stream (`docs/kernel_call_queue.md`). One contract carried over from
  stock torch: `wait()` an async collective before reading its result —
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
- **NCCL comm setup**: rank 0 calls `ncclGetUniqueId` and publishes the 128
  raw bytes through the c10d store (the same rendezvous torchrun already
  provides); every rank then calls `ncclCommInitRank`. The target GPU is
  whatever CUDA context is current, asserted per thread with a 4-call
  libcuda sequence (`nccl.set_current_cuda_device`) — DDP invokes the PG
  from the autograd thread, so this is re-asserted per thread.
- Errors raised inside collectives print a full traceback to stderr before
  propagating (`_loud`): an exception escaping into the autograd engine on
  this backend can otherwise kill the process with no Python traceback.

## Cluster notes (SLURM, IB)

- Export `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas` on nodes
  whose driver is older than r580 (see the MAX GPU requirements) — in the
  sbatch script, so every rank gets it.
- `NCCL_DEBUG=WARN` (or `INFO` during bring-up) is the first knob for
  diagnosing init hangs; on multi-homed nodes set `NCCL_SOCKET_IFNAME` if
  NCCL's interface auto-detection picks a dead interface.
- First-run kernel builds: the JIT compile pool sizes itself per process
  from whole-node resources, so 8 cold ranks can oversubscribe a node.
  The kernel caches (`__mojocache__`, `~/.modular`) are shared over NFS, so
  a one-off single-process warmup run (or just letting step 1 be slow once)
  populates them for every node.
