"""Exactness of the NVIDIA `gpu_elementwise` launcher against MAX's own
`elementwise`, run as a standalone Mojo probe.

`tmb/kernels/common/gpu_elementwise.mojo` replaces `max.algorithm.elementwise`
on the NVIDIA GPU rank-1 path for every eager call site (`elementwise`,
`logic`, `data_movement`, `nn`, `matmul`, `common/op_utils`); AMD, Apple
GPUs, CPU and rank > 1 keep forwarding to MAX unchanged and are not
exercised here (unmeasured targets, per the launcher's own docstring).

This is the ported form of the harness's own exactness test (candidate vs
MAX, bit-exact, guarded with sentinels on both sides, ranks 1-3, widths
1/2/4/8/16, both the capturing-closure and unified-closure forms, sizes
around every dispatch threshold, and one >2**31-element case that exercises
the 64-bit index path). It needs no torch device registration -- the
launcher is called directly through its Mojo API -- so it is built and run
as its own binary the way `tests/native/kernel_call_probe.mojo` and
`tests/fa4_selfload_soak_probe.mojo` are, rather than through the
`mojo_gpu` fixture.

The second test here, `test_eager_uses_our_launcher_graph_keeps_max`, checks
the other half of the contract: the launcher's fast path is gated on
`-D TMB_EAGER_ELEMENTWISE=1`, which only an eager kernel-family build gets
(`tmb/backend/loader.mojo`'s `entry()`), never the torch.compile graph
package (`native.build_graph_package()`'s `mojo precompile`). It drives a
shared helper (`op_utils._parallel_for`, called by both the eager `Gather0`
kernel and the `native_embedding` graph custom op) through the public torch
API in both modes and reads the launched kernel names back from
`torch.profiler`.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from scripts.compare_kernel_asm import build_env, mojo_cli
from torch_mojo_backend import get_accelerators, mojo_backend

pytestmark = pytest.mark.gpu

# `_FlatKernel.__call__`'s `@__name` in gpu_elementwise.mojo:
# `{trace_description}_r1_w{simd_width}_b{block_size}_u{unroll}.flat`. MAX's
# own generic elementwise kernel also starts `..._r1_w{width}_b{block}` but
# ends `_gs_{True|False}` (its grid-stride flag), never `_u{N}_flat`, so this
# is what tells the two launchers' kernels apart in a profiler trace.
_OUR_LAUNCHER_KERNEL = re.compile(r"_r1_w\d+_b\d+_u\d+_flat")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE = Path(__file__).resolve().parent / "gpu_elementwise_exactness_probe.mojo"
_MOJO_ROOT = _REPO_ROOT / "torch_mojo_backend" / "mojo"


def _require_nvidia_gpu():
    accelerators = list(get_accelerators())
    if not accelerators or accelerators[0].api != "cuda":
        pytest.skip("the gpu_elementwise launcher is NVIDIA-only")


@pytest.fixture(scope="module")
def exactness_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_nvidia_gpu()
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")

    out_dir = tmp_path_factory.mktemp("gpu_elementwise_exactness")
    out_path = out_dir / "gpu_elementwise_exactness_probe"
    command = [
        str(mojo),
        "build",
        str(_PROBE),
        "-I",
        str(_MOJO_ROOT),
        # Without this the launcher's fast path never activates (it is
        # gated on the eager build, not just NVIDIA/rank-1 -- see
        # gpu_elementwise.mojo's docstring and tmb/backend/loader.mojo's
        # `entry()`), and the probe would just compare MAX against itself.
        "-D",
        "TMB_EAGER_ELEMENTWISE=1",
        "-o",
        str(out_path),
    ]
    # Never under the GPU lock: this only compiles.
    result = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "mojo build failed for the gpu_elementwise exactness probe:\n"
        f"{result.stderr or result.stdout}"
    )
    return out_path


def test_gpu_elementwise_matches_max_bit_exact(exactness_binary: Path):
    """Every (dtype, simd_width, shape, closure form) the launcher supports
    must be bit-identical to MAX's own `elementwise` -- this is what makes
    the import swap in every eager call site behavior-preserving."""
    result = subprocess.run(
        [str(exactness_binary)],
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"gpu_elementwise exactness probe failed:\n{output}"
    assert "ALL PASS" in output, (
        f"gpu_elementwise exactness probe did not report PASS:\n{output}"
    )


def _device_kernel_names(run: Callable[[], object]) -> set[str]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.accelerator.synchronize()
    return {
        event.name
        for event in prof.events()
        if str(event.device_type) in ("DeviceType.CUDA", "DeviceType.PrivateUse1")
    }


def test_eager_uses_our_launcher_graph_keeps_max(mojo_gpu: str):
    """`op_utils._parallel_for` (`simd_width=1`) is a shared helper: the
    eager `Gather0` kernel calls it directly, and the SAME Mojo function is
    reachable from a torch.compile graph through the `native_embedding`
    custom op (`tmb/graph/nn.mojo`). It must launch our fast NVIDIA kernel
    in the first case and MAX's own kernel in the second -- proving
    `TMB_EAGER_ELEMENTWISE` actually gates what `tmb/backend/loader.mojo`
    and `native.build_graph_package()` build, not just what the source
    says.
    """
    _require_nvidia_gpu()
    device = torch.device(mojo_gpu)
    torch.manual_seed(0)
    weight = torch.randn(1000, 64, device=device)
    idx = torch.randint(0, 1000, (128,), device=device)

    eager_kernels = _device_kernel_names(lambda: F.embedding(idx, weight))
    assert any(_OUR_LAUNCHER_KERNEL.search(name) for name in eager_kernels), (
        "eager F.embedding did not launch a gpu_elementwise kernel "
        f"(device kernels seen: {sorted(eager_kernels)})"
    )

    compiled = torch.compile(F.embedding, backend=mojo_backend, fullgraph=True)
    graph_kernels = _device_kernel_names(lambda: compiled(idx, weight))
    assert not any(_OUR_LAUNCHER_KERNEL.search(name) for name in graph_kernels), (
        "a torch.compile graph launched our NVIDIA launcher instead of "
        f"MAX's own elementwise (device kernels seen: {sorted(graph_kernels)})"
    )
