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
