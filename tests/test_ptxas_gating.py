"""The ptxas static-shared-memory gate: probe, defines, and the fallback.

ptxas caps a kernel's *static* `.shared` at 48 KiB (0xc000) through CUDA 12.8
and lifts the cap in CUDA 13. It enforces that at assembly time inside
`mojo build`, and one over-limit kernel fails the whole shared library — so
on a machine whose `MODULAR_NVPTX_COMPILER_PATH` names an older assembler the
matmul extensions do not merely lose their fast routes, they stop importing,
and the loader has no run-time fallback to soften that
(docs/kernel_call_queue.md). `PTXAS_BIG_SMEM` compiles those routes out
instead, leaving the ones that already fit.

The probe itself works by actually building `ptxas_probe.mojo` with
`mojo build` (see eager_kernels._ptxas_assembles): the fakes below stand in
for the assembler that build shells out to, not for a compiler this test
invokes directly. `mojo build` only shells out to it when auto-detecting a
real, physically attached accelerator, so tests that need at least one leg to
genuinely reach the fake are gated on `real_accelerator`; the two tests where
both legs fail regardless of *why* (missing binary, no attached accelerator)
are not.

Two properties matter and are tested separately: the probe must answer for
the assembler that will actually be used, and the gate must leave a correct
kernel behind for every shape the removed routes used to claim.
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
from max.driver import accelerator_count

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
    """The probe is process-cached; every test here sets up its own answer.

    Only the in-process `functools.cache` layer: the on-disk verdict cache is
    keyed by the fake assembler's own path plus its file's size+mtime
    (`_ptxas_probe_cache_path`), and every test here builds its fake under
    pytest's per-test `tmp_path`, so distinct tests never share an entry.
    """
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()
    yield
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()


@pytest.fixture(scope="module")
def real_accelerator() -> None:
    """`mojo build` shells out to `MODULAR_NVPTX_COMPILER_PATH` only when it
    auto-detects a real, physically attached accelerator to build for
    (verified empirically: an explicit `--target-accelerator`, or none
    attached at all, both make it defer to PTX-only embedding and never
    touch that env var — see eager_kernels._ptxas_assembles). A test that
    needs the fake to actually be reached and accept a leg needs one.

    `get_accelerators()` always appends a CPU pseudo-device (so `mojo:0`
    resolves somewhere even without one), so its list is truthy either way;
    `accelerator_count()` is the actual "is a real accelerator attached?"
    primitive it and `sm90_mojo_gpu` are both built from.
    """
    if accelerator_count() == 0:
        pytest.skip("the build-based ptxas probe needs a real accelerator")


def _fake_ptxas(tmp_path: Path, *, ceiling: int | None, exit_code: int = 1) -> str:
    """A stand-in assembler `mojo build` shells out to, accepting static
    shared memory up to `ceiling`.

    It reads the `.shared` request out of the PTX `mojo build` hands it the
    way the real ptxas does, so a probe kernel that stopped declaring a
    shared array (and therefore stopped testing anything) fails here instead
    of silently passing. `ceiling=None` refuses everything, standing in for
    an assembler that cannot answer the question at all. It also writes the
    `-o` output path on success, matching a real assembler closely enough
    that a future `mojo build` which starts checking for it still works.
    """
    script = tmp_path / "fake_ptxas"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import re, sys\n"
        "args = sys.argv[1:]\n"
        "source = [a for a in args if a.endswith('.ptx')]\n"
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
        "if '-o' in args:\n"
        "    open(args[args.index('-o') + 1], 'wb').close()\n"
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
    real_accelerator: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH", _fake_ptxas(tmp_path, ceiling=_BIG)
    )

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels.big_static_smem_flags() == {"PTXAS_BIG_SMEM": 1}


def test_assembler_capped_at_48kib_compiles_the_big_routes_out(
    real_accelerator: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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


def test_disk_cache_reprobes_only_when_the_assembler_fingerprint_changes(
    real_accelerator: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A warm process must read the on-disk verdict, not rebuild, unless the
    binary `MODULAR_NVPTX_COMPILER_PATH` names actually changed.

    The "changed" leg uses a second, distinct path rather than overwriting
    the first path's file in place: `mojo build` turns out to keep its own
    persistent compilation cache (under `MODULAR_HOME`, confirmed to survive
    across separate processes) keyed on the compiler *path string*, not on
    the bytes at that path, so replacing a binary in place while `mojo`
    itself has already cached an answer for that exact path is not something
    this probe -- or any caller of `mojo build` -- can observe. A distinct
    install path (a different CUDA toolkit directory, the realistic trigger
    for "the assembler changed") is exactly the case the size+mtime
    fingerprint has to catch, and it is what this test exercises.
    """
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    path = _fake_ptxas(tmp_path, ceiling=_BIG)
    monkeypatch.setenv("MODULAR_NVPTX_COMPILER_PATH", path)

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    cache_file = eager_kernels._ptxas_probe_cache_path(path)
    assert cache_file.is_file()

    # A fresh functools.cache (a new process, in effect) with the same
    # fingerprint must read the disk verdict and never call mojo build again.
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()
    calls: list[int] = []
    real_ptxas_assembles = eager_kernels._ptxas_assembles
    monkeypatch.setattr(
        eager_kernels,
        "_ptxas_assembles",
        lambda size: calls.append(size) or real_ptxas_assembles(size),
    )
    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert calls == []

    # A different assembler at a different path is a different fingerprint
    # (different path string alone already changes the cache key; a fresh
    # binary's distinct size/mtime is the part this test is really after),
    # which must force a fresh probe with the new, correct answer.
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_path = _fake_ptxas(other_dir, ceiling=_CONTROL)
    monkeypatch.setenv("MODULAR_NVPTX_COMPILER_PATH", other_path)
    assert eager_kernels._ptxas_probe_cache_path(other_path) != cache_file
    assert eager_kernels._ptxas_supports_big_static_smem() is False
    assert calls, "a changed fingerprint must not be served from the stale entry"


def test_probe_source_declares_more_than_the_ceiling_and_uses_it() -> None:
    """Guard rails on the probe source itself.

    A kernel whose shared buffer the compiler can prove dead would assemble
    everywhere regardless of size — a probe that always passes is worse than
    no probe, because it looks like evidence.
    """
    text = eager_kernels._PTXAS_PROBE_SOURCE.read_text()
    assert _BIG > _CONTROL == 49152
    assert "address_space=AddressSpace.SHARED" in text
    assert "stack_allocation[" in text
    # Written, barrier-synced, read back, and stored to a caller-visible
    # pointer: nothing in that chain is provably dead.
    assert "barrier()" in text
    assert "out_ptr[0]" in text


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
    assert readers == {
        "variant_gates.mojo",
        "gemm16_matmul_ops/gemm16_v3_kernels.mojo",
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
    """The gate removes routes; something must still serve their shapes.

    Structural, because the alternative is a build on an old assembler that
    enqueues nothing and returns a buffer of garbage. For the 16-bit GEMMs
    the survivor is the mma.sync ladder, for fp32 the 64x64 TN core, and in
    both files that call has to sit *outside* every `comptime if`.
    """
    v3 = (_KERNEL_DIR / "gemm16_matmul_ops" / "gemm16_v3_kernels.mojo").read_text()
    for fallback in ("_enqueue_accepted_bf16_gemm(", "_enqueue_accepted_bf16_bmm("):
        # Column 4 is the body of `enqueue_gemm16_gemm`/`enqueue_gemm16_bmm`
        # itself; anything deeper is nested inside a branch.
        assert f"\n    {fallback}" in v3, fallback

    tn = (_KERNEL_DIR / "matmul_ops" / "tn_f32_gemm_kernels.mojo").read_text()
    assert "comptime if not _big_static_smem_on():\n        use_t128 = False" in tn
    assert "\n    if not use_t128:\n        _tn_core_launch[64, 64," in tn


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
"""Exercise every route the >48 KiB gate removes, against a CPU reference."""

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
    """End to end with the fast routes compiled out: mm, bmm, linear and
    SDPA must still match a CPU reference.

    This is the only check that the surviving routes actually *cover* the
    shapes the removed ones claimed. It compiles two extension families the
    first time it runs on a machine.
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
