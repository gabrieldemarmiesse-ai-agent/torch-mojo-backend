# GPT-2 XL FSDP2 throughput

Measured on 2026-09-17 with two NVIDIA H100 80GB HBM3 GPUs connected by NVLink,
on an exclusive Slurm node (`par2dc5-ai-prd-cl02s02dgx23`, job 256161).
These are end-to-end training measurements of the current implementation,
including the correctness-first MojoCCL reduce-scatter.

## Results

Throughput is aggregate input tokens/second across both GPUs. Higher is better.
Each result is the median of six synchronized 10-step windows: three windows
from each of two independent torchrun launches, with the order reversed in the
second round. The range includes every window; no samples were discarded.

### Sequence length 1024

| Configuration | Tokens/s | Step time (ms) | Window range (tokens/s) |
|---|---:|---:|---:|
| Stock PyTorch CUDA + NCCL | 9,150.4 | 223.82 | 8,689.3–9,167.5 |
| Torch Mojo + NCCL | 6,088.0 | 336.40 | 5,741.0–6,110.3 |
| Torch Mojo + MojoCCL | 4,114.8 | 497.72 | 4,023.3–4,130.4 |

### Sequence length 64

| Configuration | Tokens/s | Step time (ms) | Window range (tokens/s) |
|---|---:|---:|---:|
| Stock PyTorch CUDA + NCCL | 572.1 | 223.73 | 570.4–573.7 |
| Torch Mojo + NCCL | 362.7 | 352.88 | 340.4–364.2 |
| Torch Mojo + MojoCCL | 259.1 | 493.97 | 254.2–260.9 |

At sequence length 1024, Mojo + NCCL delivers 66.5% of the stock CUDA
throughput. Mojo + MojoCCL delivers 45.0% of stock CUDA and 67.6% of
Mojo + NCCL throughput (32.4% lower throughput when replacing NCCL).

## Training configuration

- Standard Hugging Face GPT-2 XL: 48 layers, 25 heads, width 1600,
  vocabulary 50257, 1,557,611,200 parameters; random initialization.
- Batch size 1 per GPU, global batch size 2; fixed rank-specific synthetic
  tokens, dropout disabled, no activation checkpointing or accumulation.
- FSDP2 wraps each transformer block and the root; default resharding.
- BF16 block parameters, FP32 gradient reduction, and BF16 autocast.
  The root retains FP32 embedding/head parameters on every configuration,
  matching the current Mojo embedding-backward requirement.
- AdamW with learning rate 1e-4, default betas/epsilon/weight decay,
  `foreach=False`, gradient norm clipping at 1.0 with `foreach=False`.
- Eager execution, without `torch.compile`; Transformers selects its SDPA
  interface on both devices, with each device using its current dispatch.
  Mojo currently uses the differentiable math decomposition for SDPA.
- PyTorch 2.11.0+cu128, Transformers 5.4.0, MAX/Mojo 26.5.0,
  NCCL 2.28.9 for both vendor-library configurations; driver 570.211.01.
- Same CPU allocation and `OMP_NUM_THREADS=1`; physical GPUs 0 and 1,
  protected by both `/tmp/gpu_lock_0.lock` and `/tmp/gpu_lock_1.lock`.
- GPU clock locking was denied by the driver. Five-second telemetry samples
  with nonzero GPU utilization all reported 1980 MHz SM clocks; no tuning
  or clock settings were changed.

Five complete training steps warm up each process before timing. Each timed
window includes forward, backward, clipping, optimizer update, and gradient
clearing. Initialization, compilation, warmup, loss checks, and logging are
excluded. The device is synchronized before and after each window; a CPU
Gloo control group selects the maximum elapsed time across ranks. The token
count is `world_size * batch_size * sequence_length * steps`.

All runs completed with finite losses. Synthetic repeated batches make this
a throughput comparison, not an assessment of model quality. These results
apply to the stated configuration, not maximum throughput after batch-size,
optimizer, or compiler tuning. MojoCCL reduce-scatter currently performs
extra communication and synchronizes the CPU; its implementation prioritizes
correctness over performance.

## Reproduction

Use a CUDA-compatible PyTorch environment (this machine needs the CUDA 12.8
wheel; the project environment’s CUDA 13 wheel cannot initialize its driver).
Pin the workspace on `PYTHONPATH` when using an external environment to avoid
accidentally importing another editable checkout.

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
export TORCH_MOJO_BACKEND_CCL=vendor
flock /tmp/gpu_lock_0.lock flock /tmp/gpu_lock_1.lock \
  uv run --no-project --python /path/to/cu128-venv/bin/python \
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
  demo_scripts/gpt2_fsdp2.py --model gpt2-xl --device cuda \
  --dtype bfloat16 --sequence-length 1024 --batch-size 1 \
  --benchmark --warmup 5 --steps 10 --windows 3 --output cuda.json
```

For Mojo + NCCL, use `--device mojo` with `TORCH_MOJO_BACKEND_CCL=vendor`.
For Mojo + MojoCCL, use `--device mojo` with `TORCH_MOJO_BACKEND_CCL=mojo`.
Repeat with `--sequence-length 64`, then reverse the configuration and sequence
order for the second round. The JSON output preserves all timed windows.

The working tree was based on `46ce5ba8da4361eb24b71a28f252eedf4dbed740`,
with the FSDP2/MojoCCL support and benchmark additions present. The source
snapshot and raw run records are retained locally under
`current_bench_train/fsdp2_throughput/`.
