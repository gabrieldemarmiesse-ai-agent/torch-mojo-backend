"""Changes to shared graph/native math must invalidate native kernel builds."""

import subprocess
from pathlib import Path

import pytest

from torch_mojo_backend import native


@pytest.mark.parametrize("through_op_utils", [False, True])
def test_shared_math_invalidates_native_cache(tmp_path: Path, through_op_utils: bool):
    package = Path(__file__).resolve().parents[1] / "torch_mojo_backend"
    kernels = tmp_path / "eager_kernels"
    family = kernels / "example"
    family.mkdir(parents=True)
    entry = family / "example.mojo"
    if through_op_utils:
        entry.write_text("from op_utils import elementwise_unary\n")
        helpers = kernels / "op_utils"
        helpers.mkdir()
        entry = helpers / "__init__.mojo"
    entry.write_text("from mojo_kernels.unary_math import elementwise_unary\n")
    shared = tmp_path / "mojo_kernels"
    shared.mkdir()
    (shared / "__init__.mojo").write_text("")
    math = shared / "unary_math.mojo"
    math.write_text("# first implementation\n")
    source = tmp_path / "cache_probe.mojo"
    source.write_text(
        "from loader import Loader\n"
        "from std.sys import argv\n\n"
        "def main() raises:\n"
        '    var loader = Loader(argv()[1], "", "", "", "test", False)\n'
        '    print(loader.source_hash("example"))\n'
    )
    executable = tmp_path / "cache_probe"
    build = subprocess.run(
        [
            native._find_mojo(),
            "build",
            str(source),
            "-I",
            str(package / "native" / "mojo"),
            "-I",
            str(package / "eager_kernels"),
            "-I",
            str(package),
            "-o",
            str(executable),
            "--Werror",
        ],
        capture_output=True,
        text=True,
        env=native.compiler_env(),
    )
    assert build.returncode == 0, build.stdout + build.stderr

    def source_hash() -> str:
        return subprocess.check_output(
            [str(executable), str(kernels)], text=True
        ).strip()

    before = source_hash()
    assert source_hash() == before
    math.write_text("# changed implementation\n")
    assert source_hash() != before


def test_shared_math_invalidates_backend_cache(tmp_path: Path, monkeypatch):
    kernels = tmp_path / "eager_kernels"
    kernels.mkdir()
    (kernels / "variant_gates.mojo").write_text("# specialization gates\n")
    shared = tmp_path / "mojo_kernels"
    shared.mkdir()
    math = shared / "math_utils.mojo"
    math.write_text("# original math\n")
    backend = tmp_path / "native"
    backend.mkdir()
    (backend / "backend.mojo").write_text("# backend\n")
    monkeypatch.setattr(native, "_KERNELS_DIR", kernels)
    monkeypatch.setattr(native, "_MOJO_SRC", backend)
    before = native._hash_files(native._mojo_closure(), "test")
    math.write_text("# changed math\n")
    assert native._hash_files(native._mojo_closure(), "test") != before
