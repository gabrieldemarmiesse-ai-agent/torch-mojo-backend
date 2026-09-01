"""The ptxas static-shared-memory gate (`_ptxas_supports_big_static_smem`).

The fakes stand in for the assembler `mojo build` shells out to
(`MODULAR_NVPTX_COMPILER_PATH`), which it only does with a real accelerator
attached; tests where a leg must reach the fake need `real_accelerator`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from max.driver import accelerator_count
from max.dtype import DType

from torch_mojo_backend import eager_kernels, get_accelerators

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KERNEL_DIR = _REPO_ROOT / "torch_mojo_backend" / "eager_kernels"

_BIG, _CONTROL = eager_kernels._PTXAS_PROBE_SIZES


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> Iterator[None]:
    """In-process layer only; the disk cache is per test via the fake's tmp_path."""
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()
    yield
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()


@pytest.fixture(scope="module")
def real_accelerator():
    # `get_accelerators()` is truthy even without one (CPU pseudo-device).
    if accelerator_count() == 0:
        pytest.skip("the build-based ptxas probe needs a real accelerator")


def _fake_ptxas(tmp_path: Path, *, ceiling: int | None, exit_code: int = 1) -> str:
    """An assembler accepting static `.shared` up to `ceiling` (None: nothing).

    It parses the sizes out of the PTX, so a probe kernel that stopped
    declaring shared memory fails here instead of passing everywhere.
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
):
    """No compiler path set: MAX assembles in-process and nothing is probed."""
    monkeypatch.delenv("MODULAR_NVPTX_COMPILER_PATH", raising=False)
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)

    def fail(*args: object, **kwargs: object):
        raise AssertionError("the probe must not exec anything with no compiler set")

    monkeypatch.setattr(eager_kernels, "_ptxas_assembles", fail)

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels.big_static_smem_flags() == {"PTXAS_BIG_SMEM": 1}


@pytest.mark.parametrize(("override", "expected"), [("0", False), ("1", True)])
def test_env_override_beats_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: str, expected: bool
):
    """The override wins in both directions over a live probe."""
    # The fake answers the opposite of the override, so an ignored override fails.
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH",
        _fake_ptxas(tmp_path, ceiling=None if expected else _BIG),
    )
    monkeypatch.setenv(eager_kernels._PTXAS_BIG_SMEM_ENV, override)

    assert eager_kernels._ptxas_supports_big_static_smem() is expected
    assert bool(eager_kernels.big_static_smem_flags()) is expected


def test_assembler_that_takes_the_big_request_keeps_the_fast_routes(
    real_accelerator: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH", _fake_ptxas(tmp_path, ceiling=_BIG)
    )

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels.big_static_smem_flags() == {"PTXAS_BIG_SMEM": 1}


def test_assembler_capped_at_48kib_sends_no_define(
    real_accelerator: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The CUDA 12.x case: the control leg passes, the big one does not."""
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH", _fake_ptxas(tmp_path, ceiling=_CONTROL)
    )

    assert eager_kernels._ptxas_supports_big_static_smem() is False
    assert eager_kernels.big_static_smem_flags() == {}


def test_assembler_that_fails_both_legs_is_not_evidence_of_a_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """An uninformative probe keeps the default rather than downgrading forever."""
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv(
        "MODULAR_NVPTX_COMPILER_PATH", _fake_ptxas(tmp_path, ceiling=None)
    )

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels._PTXAS_BIG_SMEM_ENV in capsys.readouterr().err


def test_assembler_that_cannot_be_run_is_not_evidence_of_a_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    monkeypatch.setenv("MODULAR_NVPTX_COMPILER_PATH", str(tmp_path / "no-such-ptxas"))

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    assert eager_kernels._PTXAS_BIG_SMEM_ENV in capsys.readouterr().err


def test_disk_cache_reprobes_only_when_the_assembler_fingerprint_changes(
    real_accelerator: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A warm process reads the disk verdict; a different binary reprobes.

    The changed leg uses a second path, not an in-place overwrite: `mojo
    build`'s own cache is keyed on the compiler path string and would answer
    stale for the same path (see `_ptxas_probe_cache_path`).
    """
    monkeypatch.delenv(eager_kernels._PTXAS_BIG_SMEM_ENV, raising=False)
    path = _fake_ptxas(tmp_path, ceiling=_BIG)
    monkeypatch.setenv("MODULAR_NVPTX_COMPILER_PATH", path)

    assert eager_kernels._ptxas_supports_big_static_smem() is True
    cache_file = eager_kernels._ptxas_probe_cache_path(path)
    assert cache_file.is_file()

    # Same fingerprint, fresh process (in effect): served from disk, no build.
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

    # A different binary is a different fingerprint and must reprobe.
    eager_kernels._ptxas_supports_big_static_smem.cache_clear()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_path = _fake_ptxas(other_dir, ceiling=_CONTROL)
    monkeypatch.setenv("MODULAR_NVPTX_COMPILER_PATH", other_path)
    assert eager_kernels._ptxas_probe_cache_path(other_path) != cache_file
    assert eager_kernels._ptxas_supports_big_static_smem() is False
    assert calls, "a changed fingerprint must not be served from the stale entry"


def test_probe_source_declares_more_than_the_ceiling_and_uses_it():
    """A probe whose buffer the compiler can prove dead would pass everywhere."""
    text = eager_kernels._PTXAS_PROBE_SOURCE.read_text()
    assert _BIG > _CONTROL == 49152
    assert "address_space=AddressSpace.SHARED" in text
    assert "stack_allocation[" in text
    assert "barrier()" in text
    assert "out_ptr[0]" in text


def test_only_the_two_families_with_big_smem_routes_send_the_define():
    """An unread define forks a byte-identical build, so senders must have readers."""
    aten_fast = (_KERNEL_DIR / "aten_fast.py").read_text()
    senders = aten_fast.count("big_static_smem_flags()")
    # Gemm16, Bmm16, _MatmulSpecExtension._flag_items.
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
):
    """A flag on the compile line but not in the cache key would file a gated
    .so under the ungated name."""
    from torch_mojo_backend.eager_kernels import aten_fast

    for supported in (True, False):
        monkeypatch.setattr(
            eager_kernels,
            "big_static_smem_flags",
            lambda supported=supported: {"PTXAS_BIG_SMEM": 1} if supported else {},
        )
        tensors = tuple(_StubTensor() for _ in range(2))
        defines = aten_fast._MatmulSpecExtension.make_defines("MatmulSpec", tensors, 0)
        canonical = aten_fast._MatmulSpecExtension.make_canonical_defines(
            "MatmulSpec", tensors, 0
        )
        assert ("PTXAS_BIG_SMEM" in defines) is supported
        assert eager_kernels.normalize_defines(defines) == canonical


class _StubTensor:
    """A `MojoTensorLike` stand-in; the define hooks read only `_dtype`."""

    _shape: tuple[int, ...] = (4, 4)
    _mojo_strides: tuple[int, ...] = (4, 1)
    _dtype: DType = DType.float32
    _device: object = None


def test_gated_routes_leave_an_unconditional_fallback_behind():
    """The mma.sync fallback must sit outside every `comptime if`: the fast
    routes are still sm_9x- and shape-gated, and a build reaching none of
    them must not enqueue nothing."""
    v3 = (_KERNEL_DIR / "gemm16_matmul_ops" / "gemm16_v3_kernels.mojo").read_text()
    for fallback in ("_enqueue_accepted_bf16_gemm(", "_enqueue_accepted_bf16_bmm("):
        # Column 4 is the function body; deeper is nested inside a branch.
        assert f"\n    {fallback}" in v3, fallback


def test_fp32_tn_routes_serve_every_regime_instead_of_gating_out():
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
def sm90_mojo_gpu():
    accelerators = list(get_accelerators())
    if not accelerators or accelerators[0].api != "cuda":
        pytest.skip("the gated routes are NVIDIA-only")
    if accelerators[0].architecture_name != "sm_90a":
        pytest.skip("the gated routes are compiled only for sm_90a")


# Out of process: the gate is answered once per process and the loader
# memoizes every specialization key, so the env var cannot be flipped live.
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
    # atol scales with the largest reference entry: GEMM error follows the
    # reduction's magnitude, not the element that landed near zero.
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

# Transposed-A mm: the only layout reaching the gated fp32 128x128 TN cores.
for dtype, rel in ((torch.float32, FP32), (torch.bfloat16, BF16)):
    at, b = rand(512, 256, dtype=dtype), rand(512, 128, dtype=dtype)
    ref = at.float().t() @ b.float()
    check(f"mm TN {dtype}", at.to(DEV).t() @ b.to(DEV), ref, rel)

for dtype, rel in ((torch.bfloat16, BF16), (torch.float16, FP16)):
    a, b = rand(8, 128, 256, dtype=dtype), rand(8, 256, 192, dtype=dtype)
    ref = torch.bmm(a.float(), b.float())
    check(f"bmm {dtype}", torch.bmm(a.to(DEV), b.to(DEV)), ref, rel)

# linear fwd+bwd under bf16 autocast: addmm and both gradient GEMMs.
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

# Causal SDPA: its math path reaches the same 16-bit BMM routes.
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
):
    """With the define absent, mm/bmm/linear/SDPA still match CPU: the >48 KiB
    kernels run with their tiles in dynamic shared memory, a build CI does not
    otherwise exercise. Compiles two extension families on first run."""
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
