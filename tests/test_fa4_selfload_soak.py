"""Race regression soak for the self-loading FA4 fwd kernel (CI-lite).

Ported from agent A's phase-2c harness
(``/scratch/fa4-fwd-harness-2c/soak_v7.mojo``), shortened to two shapes and
fewer reps for the permanent suite: bursts of back-to-back self-load kernel
launches with no inter-launch sync (so consecutive kernels' CTAs are
co-resident and warp scheduling gets perturbed), checked for bitwise
determinism across reps plus a tolerance check against the structurally
different phase-2b kernel. The full soak (more shapes, more reps, multiple
clock pins) stays in the harness; see ``tests/fa4_selfload_soak_probe.mojo``
for what this CI-lite variant covers and why (the PREFETCH invariant this
guards is documented next to its definition in
``fa4_fwd_selfload_kernel.mojo``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.compare_kernel_asm import build_env, mojo_cli
from torch_mojo_backend import get_accelerators

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE = Path(__file__).resolve().parent / "fa4_selfload_soak_probe.mojo"
_FA4_DIR = _REPO_ROOT / "torch_mojo_backend" / "eager_flash_attention"


@pytest.fixture(scope="module")
def selfload_soak_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    accelerators = list(get_accelerators())
    if (
        not accelerators
        or accelerators[0].api != "cuda"
        or (accelerators[0].architecture_name != "sm_90a")
    ):
        pytest.skip("the self-load FA4 fwd kernel is H100 (sm_90a) only")
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")

    out_dir = tmp_path_factory.mktemp("fa4_selfload_soak")
    out_path = out_dir / "fa4_selfload_soak_probe"
    command = [
        str(mojo),
        "build",
        str(_PROBE),
        "-I",
        str(_FA4_DIR),
        "-o",
        str(out_path),
    ]
    result = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "mojo build failed for the self-load soak probe:\n"
        f"{result.stderr or result.stdout}"
    )
    return out_path


def test_selfload_soak_no_race(selfload_soak_binary: Path) -> None:
    """64 back-to-back self-load launches per shape must be bitwise
    deterministic and match the phase-2b kernel within tolerance."""
    result = subprocess.run(
        [str(selfload_soak_binary)],
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"self-load soak failed:\n{output}"
    assert "SOAK PASS" in output, f"self-load soak did not report PASS:\n{output}"
