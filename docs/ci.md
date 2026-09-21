# Unit tests in CI

The `CI` workflow runs CPU unit tests on GitHub-hosted `ubuntu-latest` runners
and GPU unit tests on the self-hosted L4 runner. The L4 uses the default
`self-hosted`, `linux`, and `x64` labels; these should identify the GPU runner.
Lint, type checking, and benchmark bookkeeping stay on GitHub-hosted runners.

The CPU selection is `uv run pytest tests/ -m "not gpu"`, sharded 40 ways.
The GPU selection is `uv run pytest tests/ -m gpu`, divided into three jobs:

| Job | Selection | PyTorch wheel |
| --- | --- | --- |
| CUDA compiler | `gpu and cuda` | Locked CUDA wheel |
| Mojo device | `gpu and not cuda and not cpu_torch` | Locked CUDA wheel |
| Mojo with CPU torch (Inductor/Triton and allocator) | `gpu and cpu_torch` | CPU wheel, same version |

CUDA compiler tests run in a separate process because registering Mojo as
PyTorch's accelerator breaks CUDA autograd. Inductor/Triton integration needs
a CPU torch wheel because its process-wide driver cannot coexist with the CUDA
torch driver. The CPU wheel still runs Mojo kernels on the L4. GPU jobs check
device availability first, so a missing driver cannot produce a successful run
with every GPU test skipped. Tests requiring multiple GPUs, another vendor, or
an H100 still skip when that hardware is absent.

The root `conftest.py` marks GPU tests from their fixture dependencies, including
indirect dependencies such as `mojo_inductor` → `mojo_gpu`. The shared `device`
fixture marks only its CUDA parameter, keeping the CPU cases on hosted runners.
Tests that manage devices themselves must use `@pytest.mark.gpu`; CUDA compiler
tests use `@pytest.mark.cuda`, which also implies `gpu`. For mixed parameter
sets, put the marker on the GPU `pytest.param` only. Pure validation tests in
GPU-related modules still belong to the CPU selection.

The self-hosted jobs run serially and take `/tmp/gpu_lock_0.lock` around GPU
commands. They preserve the runner's uv, Mojo, and native build caches, using
`UV_CACHE_DIR`, `MODULAR_HOME`, and `TORCH_MOJO_BACKEND_CACHE_DIR` if configured,
otherwise persistent directories under the user's cache directory. No cache
cleanup or pruning runs between jobs. The dependency environment is synced
against `uv.lock` each job; cached downloads and builds remain available.

Performance benchmarks and the separately invoked OpInfo conformance suite are
not part of these unit-test jobs. `benchmarks/test_coverage.py` retains its own
CPU job.
