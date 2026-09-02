"""PTX ordering regression check for the self-loading FA4 fwd kernel.

``fa4_fwd_selfload_kernel.mojo`` deleted the empty[] mbarriers a phase-2b
producer/consumer split used to prove a ring slot's smem read had retired
before the next TMA refill into that same slot. The substitute proof is
program order plus ``wgmma.wait_group`` (see the PREFETCH invariant comment
next to its definition in that file): the refill's `cp.async.bulk.tensor` /
`mbarrier.arrive.expect_tx` must be textually emitted AFTER the
`wgmma.wait_group` that proves the slot is free. That is true of today's
Mojo/LLVM toolchain, but ``wgmma.wait_group`` carries no LLVM memory
clobber, so nothing stops a *future* toolchain from reordering a refill
ahead of its wait -- this test is the regression guard for that drift, not
a check on this PR's own logic (which the soak in
``test_fa4_selfload_soak.py`` covers at runtime).

Cross-compiles with ``mojo build --emit asm --target-accelerator sm_90a``
(needs no GPU, never touches ``/tmp/gpu_lock_0.lock``) the same way
``scripts/compare_kernel_asm.py`` does, and greps the emitted PTX.

Builds land in a STABLE directory (``tests/__mojocache__/...``, matching the
already-gitignored ``__mojocache__/`` pattern), not a fresh temp dir per run:
Mojo's own transform cache is keyed by kernel content hash independent of
``-o``, and replays a previously built kernel's sidecar to the path it was
FIRST built at -- a fresh ``tempfile.TemporaryDirectory()`` each run means
that path is already gone by the second run in one session (see
``compare_kernel_asm.py``'s own docstring: "Both trees are built into one
shared directory per specialization, which is load-bearing").
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts.compare_kernel_asm import build_env, mojo_cli

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE = Path(__file__).resolve().parent / "fa4_selfload_ptx_probe.mojo"
_D128_GUARD_PROBE = (
    Path(__file__).resolve().parent / "fa4_selfload_d128_guard_probe.mojo"
)
_FA4_DIR = _REPO_ROOT / "torch_mojo_backend" / "eager_flash_attention"
_BUILD_DIR = Path(__file__).resolve().parent / "__mojocache__" / "fa4_selfload_ptx"
_D128_GUARD_BUILD_DIR = (
    Path(__file__).resolve().parent / "__mojocache__" / "fa4_selfload_d128_guard"
)

_WAIT_GROUP_RE = re.compile(r"\bwgmma\.wait_group\.sync\.aligned\b")
_REFILL_COPY_RE = re.compile(r"\bcp\.async\.bulk\.tensor\b.*mbarrier::complete_tx")
_EXPECT_TX_RE = re.compile(r"\bmbarrier\.arrive\.expect_tx\b")


def _build_probe_ptx(out_dir: Path) -> str:
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stale sidecars from a previous source edit would otherwise accumulate
    # (the directory is stable across runs -- see the module docstring) and
    # break the "exactly one .ptx" sanity check below.
    for stale in list(out_dir.glob("*.ptx")) + list(out_dir.glob("*.s")):
        stale.unlink()
    out_stem = out_dir / "fa4_selfload_ptx_probe"
    command = [
        str(mojo),
        "build",
        str(_PROBE),
        "-I",
        str(_FA4_DIR),
        "--emit",
        "asm",
        "--target-accelerator",
        "sm_90a",
        "-o",
        str(out_stem) + ".s",
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
        "mojo build --emit asm failed for the self-load fwd probe:\n"
        f"{result.stderr or result.stdout}"
    )
    ptx_files = sorted(out_dir.glob("*.ptx"))
    assert ptx_files, (
        "mojo build --emit asm produced no .ptx sidecar -- the probe did"
        " not reach a GPU kernel instantiation"
    )
    selfload_ptx = [p for p in ptx_files if "selfload" in p.name]
    assert selfload_ptx, (
        f"no PTX sidecar named for the self-load kernel among {ptx_files}"
        " -- check fwd_fa4_selfload_kernel's naming"
    )
    assert len(selfload_ptx) == 1, (
        f"expected exactly one self-load kernel instantiation, got {selfload_ptx}"
    )
    return selfload_ptx[0].read_text()


@pytest.fixture(scope="module")
def selfload_ptx() -> str:
    return _build_probe_ptx(_BUILD_DIR)


def _line_numbers(pattern: re.Pattern[str], text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines()) if pattern.search(line)]


def _assert_every_event_follows_a_wait(events: list[tuple[int, str]], event_label: str):
    """Walk (line_no, kind) events in line order: every 'event_label' kind
    must have a 'wait' kind since the previous 'event_label' (or since the
    start of the region, for the first one)."""
    waits_since_last_event = 0
    seen_event = 0
    for _line_no, kind in sorted(events):
        if kind == "wait":
            waits_since_last_event += 1
        else:
            assert waits_since_last_event >= 1, (
                f"a {event_label} at line {_line_no} has no preceding"
                " wgmma.wait_group since the previous one (or region start)"
                " -- the refill was not textually ordered after the wait"
                " that proves its slot is free"
            )
            waits_since_last_event = 0
            seen_event += 1
    assert seen_event >= 3, (
        f"only {seen_event} {event_label} event(s) found after the first"
        " wgmma.wait_group -- expected at least 3 (1 prologue refill + 2"
        " per loop trip), the check would otherwise be nearly vacuous"
    )


def test_selfload_refills_follow_their_wait_group(selfload_ptx: str):
    """Every K/V refill TMA copy is textually ordered after a wgmma.wait_group.

    The prologue's initial fill (Q plus the first PREFETCH tile-pairs) has
    no preceding wait -- there is nothing to wait for, no read has happened
    yet -- so this only checks copies/expects that occur AFTER the first
    wgmma.wait_group in the file, which excludes exactly that initial fill.
    """
    wait_lines = _line_numbers(_WAIT_GROUP_RE, selfload_ptx)
    assert len(wait_lines) >= 3, (
        f"expected at least 3 wgmma.wait_group sites (prologue + 2 per loop"
        f" trip), found {len(wait_lines)} -- the probe may not have reached"
        " the kernel body"
    )
    first_wait = wait_lines[0]

    refill_lines = _line_numbers(_REFILL_COPY_RE, selfload_ptx)
    expect_lines = _line_numbers(_EXPECT_TX_RE, selfload_ptx)

    wait_events = [(line, "wait") for line in wait_lines if line >= first_wait]
    refill_events = wait_events + [
        (line, "refill") for line in refill_lines if line > first_wait
    ]
    _assert_every_event_follows_a_wait(refill_events, "cp.async.bulk.tensor refill")

    expect_events = wait_events + [
        (line, "expect_tx") for line in expect_lines if line > first_wait
    ]
    _assert_every_event_follows_a_wait(expect_events, "mbarrier.arrive.expect_tx")


def test_selfload_rejects_d128_at_compile_time():
    """Scope enforcement, not just documentation: instantiating the
    self-loading kernel at head_dim=128 must fail the BUILD (ported from
    agent A2's review artifact, d128_guard_v7.mojo)."""
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")
    _D128_GUARD_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _D128_GUARD_BUILD_DIR / "d128_guard.s"
    command = [
        str(mojo),
        "build",
        str(_D128_GUARD_PROBE),
        "-I",
        str(_FA4_DIR),
        "--emit",
        "asm",
        "--target-accelerator",
        "sm_90a",
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
    output = result.stderr + result.stdout
    assert result.returncode != 0, (
        "instantiating the self-load kernel at head_dim=128 built"
        " successfully -- the d64-only comptime assert did not fire"
    )
    assert "d64-only" in output, (
        f"build failed for an unexpected reason (not the d64-only scope"
        f" guard):\n{output}"
    )
