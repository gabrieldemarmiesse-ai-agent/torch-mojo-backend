"""The eager-mode house rules that are properties of the kernel SOURCES.

AGENTS.md, "Rules about the eager mode": the point of this project is that a
CPU-only PyTorch install plus a Mojo compiler drives the GPU, so no kernel may
reach for a vendor BLAS/DNN library; and a kernel runs on the DeviceContext
its caller hands it, so it may not synchronize on its own -- that would
serialize a user's stream behind our op.

Nothing else catches a violation: an import of a vendor-backed routine
compiles and runs, and the answer is right. It just quietly makes the wheel
depend on cuBLAS. This check used to live inside one gemm16 test in
tests/test_eager_kernels.py (deleted with the old eager path); it applies to
every family, so it is repo-wide here.
"""

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "torch_mojo_backend"
KERNEL_ROOTS = (
    PACKAGE / "mojo" / "tmb" / "kernels",
    PACKAGE / "mojo" / "tmb" / "graph",
)

# Lowercased substrings that must not appear in kernel CODE. Prose is
# exempt: a comment or docstring recording that a kernel was benchmarked
# against cuBLAS is a measurement note, not a dependency (an earlier version
# of this check tripped on exactly that).
VENDOR_LIBRARIES = ("cublas", "cudnn", "rocblas", "miopen", "triton")

# Modular's own Mojo kernels under `linalg` are fine -- that is the
# documented way to reuse them. `linalg.vendor_blas` is the one subpackage
# that dispatches to the vendor library instead.
VENDOR_ROUTES = ("from linalg.vendor_blas", "import vendor_blas")

_DOCSTRING = re.compile(r'"""(?:.|\n)*?"""')


def _kernel_sources() -> list[Path]:
    paths = [p for root in KERNEL_ROOTS for p in sorted(root.rglob("*.mojo"))]
    assert paths, f"no kernel sources found under {[str(r) for r in KERNEL_ROOTS]}"
    return paths


def _code_only(source: str) -> str:
    """Drop docstrings and `#` comments: the rules are about imports and
    calls, not about what a tuning note mentions."""
    without_docstrings = _DOCSTRING.sub("", source)
    return "\n".join(
        line.split("#", 1)[0] for line in without_docstrings.splitlines()
    ).lower()


@pytest.mark.parametrize(
    "path", _kernel_sources(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_kernel_source_takes_no_vendor_library(path: Path):
    code = _code_only(path.read_text())
    for forbidden in VENDOR_LIBRARIES + VENDOR_ROUTES:
        assert forbidden not in code, (
            f"{path.relative_to(PACKAGE)} reaches for {forbidden!r}. The whole "
            "point of this backend is that a CPU-only torch install drives the "
            "GPU (AGENTS.md, 'Rules about the eager mode', rule 1)."
        )


@pytest.mark.parametrize(
    "path", _kernel_sources(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_kernel_source_does_not_synchronize(path: Path):
    code = _code_only(path.read_text())
    assert ".synchronize(" not in code, (
        f"{path.relative_to(PACKAGE)} synchronizes: a kernel runs on the "
        "DeviceContext its caller hands it (tmb/backend/abi.mojo's `ctx_for`) "
        "and returns; blocking inside an op serializes the caller's stream."
    )


# `tmb/kernels/common/gpu_elementwise.mojo` is the one drop-in replacement
# for MAX's `elementwise` on the NVIDIA eager path (faster launcher, same
# API, gated on `-D TMB_EAGER_ELEMENTWISE=1` so torch.compile graphs still
# get MAX's own kernel -- see that module's docstring). Every OTHER source
# must import `elementwise` through it rather than straight from MAX, or an
# eager call site quietly stops benefiting from the faster launcher and
# nobody notices because both compile and both are correct.
MOJO_ROOT = PACKAGE / "mojo"
_GPU_ELEMENTWISE_MODULE = (
    MOJO_ROOT / "tmb" / "kernels" / "common" / "gpu_elementwise.mojo"
)
# The import statement, single- or parenthesized-multi-line
# (`from max.algorithm import (\n    elementwise,\n    ...\n)`); whether
# `elementwise` is among the named imports is checked separately, over the
# captured name list, so either form is caught the same way.
_MAX_ALGORITHM_IMPORT = re.compile(
    r"from\s+max\.algorithm(?:\.functional)?\s+import\s+(\((?:.|\n)*?\)|[^\n]*)"
)
_ELEMENTWISE_NAME = re.compile(r"\belementwise\b")


def _mojo_sources() -> list[Path]:
    paths = sorted(MOJO_ROOT.rglob("*.mojo"))
    assert paths, f"no Mojo sources found under {MOJO_ROOT}"
    return paths


@pytest.mark.parametrize(
    "path", _mojo_sources(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_only_gpu_elementwise_imports_max_algorithm_elementwise(path: Path):
    if path == _GPU_ELEMENTWISE_MODULE:
        return
    code = _code_only(path.read_text())
    for match in _MAX_ALGORITHM_IMPORT.finditer(code):
        assert not _ELEMENTWISE_NAME.search(match.group(1)), (
            f"{path.relative_to(PACKAGE)} imports `elementwise` straight "
            "from max.algorithm. Every eager call site must import it from "
            "tmb.kernels.common.gpu_elementwise instead (its NVIDIA/eager-"
            "build fast path forwards to MAX everywhere else), or that call "
            "site silently loses the faster launcher."
        )
