"""The ptxas static-shared-memory gate: probe, defines, and the fallback.

ptxas caps a kernel's *static* `.shared` at 48 KiB (0xc000) through CUDA 12.8
and lifts the cap in CUDA 13. It enforces that at assembly time inside
`mojo build`, and one over-limit kernel fails the whole shared library — so
on a machine whose `MODULAR_NVPTX_COMPILER_PATH` names an older assembler the
matmul extensions do not merely lose their fast routes, they stop importing,
and the loader has no run-time fallback to soften that
(docs/kernel_call_queue.md). `PTXAS_BIG_SMEM` tells the Mojo side which
assembler it is being built for, and the kernels whose tiles do not fit take
them from the dynamic (`extern`) shared window instead, which was never
subject to that cap — so every fast route is compiled and reachable under
either assembler.

Two properties matter and are tested separately: the probe must answer for
the assembler that will actually be used, and every regime must still reach
a kernel — the routes that remain behind an ordinary architecture gate need
a fallback outside it, and the ones that switched allocation must serve the
same shapes in both regimes (the end-to-end worker at the bottom).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest
from max.dtype import DType

from torch_mojo_backend import eager_kernels, get_accelerators

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KERNEL_DIR = _REPO_ROOT / "torch_mojo_backend" / "eager_kernels"

# The probe asks for this many bytes of static shared memory, which is over
# every ptxas's un-opted-in ceiling, and falls back to asking for exactly the
# ceiling. Both numbers are part of the contract with the fake assemblers
# below, so read them from the module rather than restating them.
_BIG, _CONTROL = eager_kernels._PTXAS_PROBE_SIZES


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> Iterator[None]:
    """The probe is process-cached; every test here sets up its own answer."""
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()
    yield
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()


def _fake_ptxas(tmp_path: Path, *, ceiling: int | None, exit_code: int = 1) -> str:
    """A stand-in `ptxas` that accepts static shared memory up to `ceiling`.

    It reads the `.shared` request out of the probe PTX the way the real one
    does, so a probe that stopped emitting a shared array (and therefore
    stopped testing anything) fails here instead of silently passing.
    `ceiling=None` refuses everything, standing in for an assembler that
    cannot answer the question at all.
    """
    script = tmp_path / "fake_ptxas"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import re, sys\n"
        "source = [a for a in sys.argv[1:] if a.endswith('.ptx')]\n"
        "if not source:\n"
        "    sys.exit(2)\n"
        "text = open(source[0]).read()\n"
        "sizes = [int(n) for n in re.findall(r'\\.shared[^;]*\\[(\\d+)\\]', text)]\n"
        "if not sizes:\n"
        "    sys.stderr.write('probe declares no static shared memory\\n')\n"
        "    sys.exit(3)\n"
        f"ceiling = {ceiling!r}\n"
        "if ceiling is None or max(sizes) > ceiling:\n"
        "    sys.stderr.write('ptxas error   : uses too much shared data\\n')\n"
        f"    sys.exit({exit_code})\n"
        "sys.exit(0)\n"
    )
    script.chmod(0o755)
    return str(script)


def test_bundled_assembler_is_assumed_to_take_big_static_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `MODULAR_NVPTX_COMPILER_PATH` means MAX assembles in-process.

    There is no external binary to interrogate, and the compiler MAX ships
    is new enough — so the default install must compile exactly the kernels
    it always did. Anything else would be a silent slowdown for every user
    who never set the variable.
    """
    monkeypatch.delenv("MODULAR_NVPTX_COMPILER_PATH", raising=False)
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the probe must not exec anything with no compiler set")

    monkeypatch.setattr(eager_kernels, "_ptxas_assembles", fail)

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels.big_static_smem_flags() == {"PTXAS_BIG_SMEM": 1}


@pytest.mark.parametrize(("override", "expected"), [("0", False), ("1", True)])
def test_env_override_beats_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: str, expected: bool
) -> None:
    """The escape hatch wins in both directions, including over a live probe."""
    # A fake assembler that answers the *opposite* of the override, so a test
    # that passed by accident (probe consulted, override ignored) fails.
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH",
        _fake_ptxas(tmp_path, ceiling=None if expected else _BIG),
    )
    monkeypatch.setenv(eager_kernels._PTXAS_BIG_SMEM_ENV, override)

    assert eager_kernels._ptxas_supports_big_static_smem() is expected
    assert bool(eager_kernels.big_static_smem_flags()) is expected


def test_assembler_that_takes_the_big_request_keeps_the_fast_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH", _fake_ptxas(tmp_path, ceiling=_BIG)
    )

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels.big_static_smem_flags() == {"PTXAS_BIG_SMEM": 1}


def test_assembler_capped_at_48kib_sends_no_define(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CUDA 12.x case: the control leg passes, the big one does not.

    `_CONTROL` is exactly the ceiling and must be accepted; that is what
    separates a real cap from an assembler that fails for its own reasons.
    """
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH", _fake_ptxas(tmp_path, ceiling=_CONTROL)
    )

    assert eager_kernels._ptxas_supports_big_static_smem() is False
    assert eager_kernels.big_static_smem_flags() == {}


def test_assembler_that_fails_both_legs_is_not_evidence_of_a_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An uninformative probe must keep today's behaviour, not silently
    downgrade every matmul on the machine forever."""
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH", _fake_ptxas(tmp_path, ceiling=None)
    )

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels._PTXAS_BIG_SMEM_ENV in capsys.readouterr().err


def test_assembler_that_cannot_be_run_is_not_evidence_of_a_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv("MODULAR_NVPTX_COMPILER_PATH", str(tmp_path / "no-such-ptxas"))

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels._PTXAS_BIG_SMEM_ENV in capsys.readouterr().err


def test_probe_ptx_declares_more_than_the_ceiling_and_uses_it() -> None:
    """Guard rails on the probe source itself.

    Assembling PTX that declares no shared memory, or whose array the
    assembler can prove dead, would answer True everywhere — a probe that
    always passes is worse than no probe, because it looks like evidence.
    """
    text = eager_kernels._PTXAS_PROBE_PTX
    assert _BIG > _CONTROL == 49152
    assert ".shared .align 16 .b8 probe_shared_buffer[__SIZE__];" in text
    # Written, read back, and stored to a caller-visible pointer: nothing in
    # that chain is removable.
    assert "st.shared.u32" in text
    assert "ld.shared.u32" in text
    assert "st.global.u32" in text
    assert ".target sm_90a" in text


def test_only_the_two_families_with_big_smem_routes_send_the_define() -> None:
    """A define no gate reads forks a second, byte-identical build of the
    module that sends it (see the eager_kernels module docstring), so this
    flag belongs at exactly the call sites whose Mojo reads it."""
    aten_fast = (_KERNEL_DIR / "aten_fast.py").read_text()
    senders = aten_fast.count("big_static_smem_flags()")
    # Gemm16, Bmm16, and the shared flag builder of _MatmulSpecExtension.
    assert senders == 3, "a new sender must have a Mojo gate to justify it"

    readers = {
        path.relative_to(_KERNEL_DIR).as_posix()
        for path in _KERNEL_DIR.rglob("*.mojo")
        if "_big_static_smem_on" in path.read_text()
    }
    # One reader per file that owns an allocation decision. The gemm16 family
    # shares `_v4_dyn_smem_tile` (declared in the NN v4 file, imported by the
    # NT/TN v4 and v3 files), so those importers are not readers themselves;
    # the v5 BMM file decides per instantiation and so reads the gate itself.
    assert readers == {
        "variant_gates.mojo",
        "gemm16_matmul_ops/gemm16_nn_v4_kernels.mojo",
        "gemm16_matmul_ops/gemm16_bmm_v5_kernels.mojo",
        "matmul_ops/tn_f32_gemm_kernels.mojo",
    }


def test_matmul_spec_defines_carry_the_flag_and_agree_with_the_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`make_defines` writes the compile line and `make_canonical_defines`
    writes the cache name; a flag in one and not the other would build a
    gated .so and remember it under the ungated key."""
    from torch_mojo_backend.eager_kernels import aten_fast

    for supported in (True, False):
        monkeypatch.setattr(
            eager_kernels,
            "big_static_smem_flags",
            lambda supported=supported: {"PTXAS_BIG_SMEM": 1} if supported else {},
        )
        tensors = tuple(
            _StubTensor() for _ in range(2)
        )  # only `_dtype` is read by either hook
        defines = aten_fast._MatmulSpecExtension.make_defines("MatmulSpec", tensors, 0)
        canonical = aten_fast._MatmulSpecExtension.make_canonical_defines(
            "MatmulSpec", tensors, 0
        )
        assert ("PTXAS_BIG_SMEM" in defines) is supported
        assert eager_kernels.normalize_defines(defines) == canonical


class _StubTensor:
    """A `MojoTensorLike` stand-in; the define hooks read only `_dtype`."""

    _shape: ClassVar[tuple[int, ...]] = (4, 4)
    _mojo_strides: ClassVar[tuple[int, ...]] = (4, 1)
    _dtype: ClassVar[DType] = DType.float32
    _device: ClassVar[object] = None


def test_gated_routes_leave_an_unconditional_fallback_behind() -> None:
    """Gated routes exist; something must still serve their shapes.

    Structural, because the alternative is a build that enqueues nothing and
    returns a buffer of garbage. The 16-bit fast routes no longer depend on
    the assembler — their tiles move to dynamic shared memory instead of
    being compiled out — but they are still architecture-gated (`comptime if
    _has_sm_9x()`) and shape-gated, so an sm_80 build, a biased GEMM or an
    unaligned shape reaches nothing above the mma.sync ladder. That call has
    to sit *outside* every `comptime if`.
    """
    v3 = (_KERNEL_DIR / "gemm16_matmul_ops" / "gemm16_v3_kernels.mojo").read_text()
    for fallback in ("_enqueue_accepted_bf16_gemm(", "_enqueue_accepted_bf16_bmm("):
        # Column 4 is the body of `enqueue_gemm16_gemm`/`enqueue_gemm16_bmm`
        # itself; anything deeper is nested inside a branch.
        assert f"\n    {fallback}" in v3, fallback


def test_fp32_tn_routes_serve_every_regime_instead_of_gating_out() -> None:
    """fp32 TN took the other fix: dynamic shared memory, not a fallback.

    Unlike the 16-bit GEMMs above, the 128x128 TN cores' tiles moved to
    `external_memory` on an assembler that cannot take their static
    allocation (tn_f32_gemm_core.mojo), so `use_t128` in
    tn_f32_gemm_kernels.mojo is plain shape-based selection again -- no
    `_big_static_smem_on()` override, no gate around the 128x128 launches,
    and the 64x64 core sits behind an ordinary `else:` next to them rather
    than an unconditional fallback call outside a `comptime if`.
    """
    tn_kernels = (_KERNEL_DIR / "matmul_ops" / "tn_f32_gemm_kernels.mojo").read_text()
    assert "use_t128 = False" not in tn_kernels
    assert "\n    else:\n        _tn_core_launch[64, 64," in tn_kernels

    tn_core = (_KERNEL_DIR / "matmul_ops" / "tn_f32_gemm_core.mojo").read_text()
    assert "external_memory" in tn_core
    assert "comptime if STATIC_SMEM:" in tn_core


@pytest.fixture(scope="module")
def sm90_mojo_gpu() -> None:
    accelerators = list(get_accelerators())
    if not accelerators or accelerators[0].api != "cuda":
        pytest.skip("the gated routes are NVIDIA-only")
    if accelerators[0].architecture_name != "sm_90a":
        pytest.skip("the gated routes are compiled only for sm_90a")


# Runs out of process because the gate is answered once per process, behind
# `functools.cache` and behind the loader's own memo of every specialization
# key: flipping the environment variable inside a live session would leave
# extensions already built under the other answer.
_FALLBACK_WORKER = '''
"""Exercise every route the >48 KiB gate re-allocates, against a CPU reference."""

import sys

import torch

from torch_mojo_backend import eager_kernels, register_mojo_devices

assert eager_kernels._ptxas_supports_big_static_smem() is False
assert eager_kernels.big_static_smem_flags() == {}

register_mojo_devices()
DEV = "mojo:0"
torch.manual_seed(0)


FP32, BF16, FP16 = 2e-5, 3e-2, 1e-2  # relative error each dtype is allowed


def check(name, got, ref, rel):
    # One relative tolerance, applied both element-wise and against the
    # largest reference entry: a GEMM's error scales with the magnitude of
    # the whole reduction, not with the element that happened to land near
    # zero, and a fixed atol would mean something different on every shape
    # below.
    ref = ref.float()
    torch.testing.assert_close(
        got.float().cpu(),
        ref,
        rtol=rel,
        atol=rel * ref.abs().max().clamp(min=1e-6).item(),
        msg=lambda m: name + ": " + m,
    )
    print("ok", name)


def rand(*shape, dtype=torch.float32):
    return torch.randn(*shape).to(dtype)


for dtype, rel in ((torch.float32, FP32), (torch.bfloat16, BF16), (torch.float16, FP16)):
    for m, n, k in ((256, 256, 256), (1024, 512, 2048)):
        a, b = rand(m, k, dtype=dtype), rand(k, n, dtype=dtype)
        ref = a.float() @ b.float()
        check(f"mm {dtype} {m}x{n}x{k}", a.to(DEV) @ b.to(DEV), ref, rel)

# Transposed-A mm: the weight-gradient layout, and the only shape that
# reaches the gated fp32 128x128 TN cores.
for dtype, rel in ((torch.float32, FP32), (torch.bfloat16, BF16)):
    at, b = rand(512, 256, dtype=dtype), rand(512, 128, dtype=dtype)
    ref = at.float().t() @ b.float()
    check(f"mm TN {dtype}", at.to(DEV).t() @ b.to(DEV), ref, rel)

for dtype, rel in ((torch.bfloat16, BF16), (torch.float16, FP16)):
    a, b = rand(8, 128, 256, dtype=dtype), rand(8, 256, 192, dtype=dtype)
    ref = torch.bmm(a.float(), b.float())
    check(f"bmm {dtype}", torch.bmm(a.to(DEV), b.to(DEV)), ref, rel)

# linear forward and backward under bf16 autocast: mm, addmm and both
# gradient GEMMs (dX = dY @ W, dW = dY^T @ X) in one call. The reference is
# the exact fp32 computation, so the bf16 tolerance covers the cast too.
x, w, bias = rand(256, 512), rand(384, 512), rand(384)
for t in (x, w, bias):
    t.requires_grad_()
xd, wd, bd = (t.detach().to(DEV).requires_grad_() for t in (x, w, bias))
ref_out = torch.nn.functional.linear(x, w, bias)
with torch.amp.autocast("mojo", dtype=torch.bfloat16):
    got_out = torch.nn.functional.linear(xd, wd, bd)
    assert got_out.dtype == torch.bfloat16, got_out.dtype
check("linear fwd", got_out, ref_out, BF16)
ref_out.float().square().sum().backward()
got_out.float().square().sum().backward()
for name, got, want in (("x", xd, x), ("w", wd, w), ("bias", bd, bias)):
    check("linear bwd d" + name, got.grad, want.grad, BF16)

# Causal SDPA, nanoGPT geometry (head_dim 64). Untouched by the gate, but
# its math path reaches the same 16-bit BMM routes.
q, k_, v = (rand(4, 6, 256, 64, dtype=torch.bfloat16) for _ in range(3))
for t in (q, k_, v):
    t.requires_grad_()
qd, kd, vd = (t.detach().to(DEV).requires_grad_() for t in (q, k_, v))
ref_att = torch.nn.functional.scaled_dot_product_attention(
    q.float(), k_.float(), v.float(), is_causal=True
)
got_att = torch.nn.functional.scaled_dot_product_attention(qd, kd, vd, is_causal=True)
check("sdpa causal fwd", got_att, ref_att, BF16)
ref_att.square().sum().backward()
got_att.float().square().sum().backward()
for name, got, want in (("q", qd, q), ("k", kd, k_), ("v", vd, v)):
    check("sdpa causal bwd d" + name, got.grad, want.grad, BF16)

print("FALLBACK PASS")
sys.exit(0)
'''


def test_gated_build_still_computes_the_right_answers(
    sm90_mojo_gpu: None, tmp_path: Path
) -> None:
    """End to end with the define absent: mm, bmm, linear and SDPA must
    still match a CPU reference.

    That build is not the one CI usually exercises -- every >48 KiB kernel
    in it stages its tiles in dynamic shared memory instead of static, with
    a different launch (`shared_mem_bytes` plus the opt-in attribute) and
    every tile at a different address -- so this is the check that the
    regime an old assembler forces still computes the same numbers. It
    compiles two extension families the first time it runs on a machine.
    """
    worker = tmp_path / "ptxas_fallback_worker.py"
    worker.write_text(_FALLBACK_WORKER)
    env = dict(os.environ)
    env[eager_kernels._PTXAS_BIG_SMEM_ENV] = "0"
    result = subprocess.run(
        [sys.executable, str(worker)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"gated-build worker failed:\n{output}"
    assert "FALLBACK PASS" in output, f"worker did not reach the end:\n{output}"
