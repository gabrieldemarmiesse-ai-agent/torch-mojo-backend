"""`torch-mojo-backend cache dir` / `cache clean` against a relocated cache,
and `torch-mojo-backend ptxas` on whatever machine this is."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from torch_mojo_backend import _ptxas, cli


def _run(cache_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # conftest turns on VERBOSE, which prints import-time diagnostics to
    # stdout; the CLI's stdout must be exactly the answer.
    env = dict(
        os.environ,
        TORCH_MOJO_BACKEND_CACHE_DIR=str(cache_dir),
        TORCH_MOJO_BACKEND_VERBOSE="0",
    )
    return subprocess.run(
        [sys.executable, "-m", "torch_mojo_backend.cli", *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cache_dir_prints_the_relocated_path(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    assert _run(cache_dir, "cache", "dir").stdout.strip() == str(cache_dir)


def test_cache_clean_removes_the_directory(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "libtmb_shim.hash-0.so").write_bytes(b"stale")
    proc = _run(cache_dir, "cache", "clean")
    assert not cache_dir.exists()
    assert str(cache_dir) in proc.stderr


def test_cache_clean_on_a_missing_directory_succeeds(tmp_path: Path):
    proc = _run(tmp_path / "absent", "cache", "clean")
    assert "nothing to remove" in proc.stderr


@pytest.mark.skipif(
    _ptxas.driver_cuda_version() is None, reason="needs an NVIDIA driver"
)
def test_ptxas_prints_the_assembler_picture(tmp_path: Path):
    out = _run(tmp_path / "cache", "ptxas").stdout
    assert "driver" in out and "ptxas found" in out


def test_ptxas_refuses_without_an_nvidia_driver(monkeypatch, capsys):
    """An AMD, Apple or CPU-only box has no driver CUDA version to judge the
    candidates by; the answer belongs to the machine the cubins will run on."""
    monkeypatch.setattr(_ptxas, "driver_cuda_version", lambda: None)
    assert cli.main(["ptxas"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no NVIDIA driver" in captured.err
