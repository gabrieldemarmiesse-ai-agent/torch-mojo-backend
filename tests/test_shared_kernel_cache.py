"""Changes to shared graph/native math must invalidate native kernel builds."""

import subprocess
from pathlib import Path

from torch_mojo_backend import native


def test_shared_math_invalidates_native_cache(tmp_path: Path):
    package = Path(__file__).resolve().parents[1] / "torch_mojo_backend"
    kernels = tmp_path / "eager_kernels"
    family = kernels / "example"
    family.mkdir(parents=True)
    (family / "example.mojo").write_text(
        "from mojo_kernels.unary_math import acos_value\n"
    )
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
