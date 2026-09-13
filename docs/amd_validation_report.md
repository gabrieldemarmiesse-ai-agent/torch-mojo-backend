# Native backend on AMD MI300A (ROCm 6.4.3): validation report

Run of `docs/amd_validation_plan.md` on the Adastra cluster (CINES),
2026-09-13, branch `mojo-native-backend` at 588f776. Everything below was run
by an agent on one exclusive MI300A node; the sections follow the plan.

## 1. Environment

| item | value |
|---|---|
| node | Adastra `a1018` (SLURM job 5408061, `--constraint=MI300 --exclusive`), 4x AMD Instinct MI300A (gfx942, APU: 128 GB HBM per device shared with the host), 501 GB host-visible RAM |
| kernel / driver | Linux 5.14.0-570.120.1.el9_6, amdgpu 6.12.12 |
| ROCm | 6.4.3 (`/opt/rocm`, hipconfig 6.4.43484), RCCL 2.22.3 from `/opt/rocm/lib/librccl.so.1` |
| mojo | Mojo 1.0.0 (ed45d567), `mojo-compiler` 1.0.0 |
| MAX | max / max-core / max-mojo-libs 26.5.0 |
| torch | 2.11.0+cpu (the CPU wheel; `uv sync` had resolved `2.11.0+cu130` and was corrected with `uv pip install --reinstall-package torch torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu`; a bare `torch==2.11.*` is a no-op because the CUDA build satisfies it) |
| triton | 3.6.0 (pulled by the CUDA wheel, kept for section 5) |
| filesystems | checkout, `.venv`, uv cache and `TORCH_MOJO_BACKEND_CACHE_DIR` on `/lus/scratch` (the `/lus/work` and login-node `/tmp` uv cache were unusable: see notes) |
| env | `ROCM_PATH=/opt/rocm`, `LD_LIBRARY_PATH=/opt/cray/pe/gcc-libs:/opt/rocm/lib:...` (GLIBCXX_3.4.30), `ROCR_VISIBLE_DEVICES=0,1,2,3` (set by SLURM), `TORCH_MOJO_BACKEND_CACHE_DIR=$SCRATCH/native-amd/cache`, `TORCH_MOJO_BACKEND_TRACE=1`, `TMPDIR=/tmp` (node-local) |
| `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM` | **unset for every single-process run** and `=1` only for the multi-rank DDP runs. With it set, every process segfaults at exit (exit code 139, HSA runtime atexit; MAX 26.5 + ROCm 6.4.3, known from the RCCL work, `demo_scripts/nanogpt_ddp.py` ends in `os._exit(0)` for that reason), which would turn every subprocess-based test into a failure. Without it one process reserves ~115 GB of the APU's memory at first use, which is fine for one process on a 501 GB node. |
| clocks | not locked: `rocm-smi --setperfdeterminism` needs root on this site; sclk level 1 (1506 MHz) / mclk 1300 MHz at rest |

Setup notes:
- `uv sync` on the login node failed twice out of the default uv cache
  (`/tmp/<user>/.cache/uv`, 30 GB): `ModuleNotFoundError: hatchling.build`
  then `trove_classifiers has no attribute classifiers`, both truncated
  archives. A fresh `UV_CACHE_DIR` on scratch fixed it.

## First contact

`mojo --print-supported-accelerators` lists `amdgpu:gfx942` (and the other
gfx targets). The first-contact script of the plan, under the lock:

| step | result |
|---|---|
| `register_mojo_devices()` | C++ shim built in 6.20 s, Mojo base library in 12.72 s, "native mojo backend ready in 15.29s (5 devices)", registered in 15.6 s |
| `get_accelerators()` | 4 `Device(type=gpu)` + the CPU device; accelerator identity `hip:gpu,hip:gpu,hip:gpu,hip:gpu,cpu:cpu` (api = hip) |
| `torch.ones(3) * 2` | `[2.0, 2.0, 2.0]`; first op 31.3 s (five extension builds: `empty.memory_format` 4.0 s, `fill_.Scalar` 6.5 s, `mul.Tensor` 7.6 s, `elementwise_ops MulScalarSpec` 5.6 s, `_to_copy` 7.4 s) |
| fp32 `torch.mm` 256x256 | **max abs error 21.7 against CPU: wrong** (see finding 1); `MatmulSpec` fp32 spec compiled in 344 s |
| stream / event | `stream/event ok`, `sum` correct (`SumSpec` 7.3 s, `_local_scalar_dense` 4.1 s) |
| `stream_native_handle` | 207391536 (non-zero `hipStream_t`) |
| process exit | exit code 139: the segfault in the HSA runtime's atexit handler that `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` causes (pre-existing, Modular's; the variable was set for this run only) |

Total 7 min 26 s wall, of which 344 s the one fp32 GEMM specialization.
The HIP vendor bindings (`hipEvent*`, `hipStreamWaitEvent`, ...) resolved and
worked at first try: `test_bringup.py` and `test_stream_ordering.py` below
pass.

Cache warm (`native.prebuild_ops()`, outside the lock): **29 min 39 s** wall
on this node, 239 extensions built in that process (1771 s of compile;
concurrent test runs built the rest); each op extension is 4 to 8 s, the
kernel specializations built on first call are the expensive part (GEMM:
fp32 344 s, bf16 485 s, fp16 120 s; elementwise / reduction 5 to 8 s).

### Finding 1: fp32 GEMM is wrong on gfx942 (pre-existing)

`torch.mm` with float32 operands returns wrong values for m, n >= 64;
bfloat16 and float16 are correct, tiny fp32 shapes are correct:

| dtype | 256x256x256 | 64x64x64 | 8x8x8 | 3x5x7 | 128x1024x512 |
|---|---|---|---|---|---|
| float32 (max abs err / ref max) | 24 / 66 | 9.8 / 34 | 4.8e-7 | 1.2e-7 | 45.9 / 142 |
| bfloat16 | 0.235 / 68 | 0.062 / 28 | 0.021 | 0.015 | 0.355 / 133 |
| float16 | 0.029 / 85 | 0.0078 / 30 | 0.0018 | 0.0009 | 0.034 / 150 |

The old eager path (upstream main ba926d9, its own warmed checkout on this
node) gives the same fp32 numbers (24 / 9.8 / ok / 42.9), so this is a bug in
the shared kernel family `eager_kernels/matmul_ops/` (gfx942 fp32 route of
`_amd_dynamic_mfma_dispatch`), not in the native backend, and it predates
the branch: the aten-level test suites had never been run on MI300A (only the
distributed tests and bf16 nanoGPT had). Root cause and fix: see "Fixes".

## 2. Runtime and op groups (`tests/native/`)

One `pytest` per file, serial, under the lock, no VMM knob. Wall time
includes waiting for the lock behind other steps and every first-call
compile; the pytest time is the suite's own.

| file | AMD result | pytest time | H100 (plan) |
|---|---|---|---|
| `test_bringup.py` | 28 passed | 155 s | 32 pass (28 collected on this tree) |
| `test_loader.py` | 7 passed | 283 s | pass |
| `test_register_retry.py` | 1 passed | 7 s | pass |
| `test_prebuilt.py` | 18 passed | 2 s | pass |
| `test_stream_ordering.py` | 4 passed | 15 s | 4 pass |
| `test_profiler.py` | 2 passed | 10 s | pass |
| `test_binary.py` | 183 passed | 495 s | |
| `test_unary.py` | 241 passed, 2 skipped (GELU bit patterns recorded on H100), 2 xfailed | 308 s | |
| `test_compare.py` | 203 passed | 2253 s | |
| `test_reductions.py` | **15 failed**, 432 passed, 4 skipped (before the cumsum fix; 40/40 cumsum cases pass after it, see finding 2) | 441 s | |
| `test_data_movement.py` | 360 passed | 467 s | |
| `test_factories.py` | 97 passed, 2 skipped (CPU-device-only case; no native GPU reference on this accelerator) | 75 s | |
| `test_foreach.py` | 30 passed, 3 xfailed, 4 xpassed (stale non-strict xfails about `linalg_vector_norm` not being registered yet) | 107 s | |

### Finding 2: cumsum bf16/f16 and outer-dim routes were declined on HIP

All 15 `test_reductions.py` failures were `NotImplementedError: cumsum ...`:
`op_cumsum` (`native/mojo/ops_reductions.mojo`) declined bfloat16 / float16
and the dim-0-of-rank-2 route on every device whose api is not `cuda`, with
the comment "only ever MEASURED on NVIDIA". Measured on MI300A: every
declined case is correct (40 of 40 cumsum tests, integer cumsum bit-exact,
float errors at accumulation level), because HIP runs the portable
one-thread-per-line kernels of `nn_ops.mojo` (the NVIDIA `block.prefix_sum`
fast path is gated separately, inside the kernels, on `ctx.api() == "cuda"`
and is untouched). The gate is widened to `cuda or hip`; Metal and the CPU
device keep the old surface. Commit 3f609ed.

## 8. Prebuilt libraries

Compute nodes have no internet here, so the shim builds (which create a
throwaway venv per torch series) ran on the login node; everything else on
the node.

| step | result |
|---|---|
| `scripts/build_prebuilt.py --torch 2.7 ... 2.14` | shims for torch 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14 (268 to 284 KiB each, glibc >= 2.32, cxx11abi1) and the base library `libtmb_backend-max26.5.0-linux-x86_64.so` (333 KiB, glibc >= 2.34); ~15 min including the wheel downloads. `--report` lists all nine. Note: `--report` alone on an empty out dir raises "no manifest entries" (it never builds). |
| `uv build` | `torch_mojo_backend-0.3.1-py3-none-any.whl` 1.93 MB (13.0 MB / 161 files uncompressed, of which 2.6 MB prebuilt libraries) |
| wheel venv | `uv pip install dist/*.whl "torch==2.11.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu`. The plan's `--index-url` form fails: it hides PyPI and `max==26.5` is not on the torch index. |
| `scripts/smoke_prebuilt_wheel.py` | "using prebuilt C++ shim for torch 2.11" and "using prebuilt Mojo base library for MAX 26.5.0", backend ready in 4.12 s, `ones(3) * 2` on the CPU device OK, rc 0 |
| register from the wheel venv on the GPU node | prebuilt shim used, ready in 2.23 s, `(arange(6)*2+1).sum()` = 36 and a 2x3 fp32 `mm` exact on `mojo:0`: the base library built with no accelerator in sight drives the MI300A |
| `TORCH_MOJO_BACKEND_PREBUILT=0` | ready in 15.49 s (both compiled), op correct: the fallback works |
| `benchmarks/test_coverage.py` | 2 passed (no GPU) |

The CI-artifact wheel was not available on this box, so the wheel tested is
the one built here.
