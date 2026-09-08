"""The CUDA 12.8 ptxas from the nvidia-cuda-nvcc-cu12 wheel is MAX's default
assembler: ``torch_mojo_backend`` sets ``MODULAR_NVPTX_COMPILER_PATH`` to it
at import unless the variable is already set."""

import os
import subprocess
import sys

import pytest

from torch_mojo_backend import _ptxas

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the nvcc wheel is a Linux dependency"
)


def test_wheel_ships_a_cuda_12_8_ptxas():
    ptxas = _ptxas.wheel_ptxas()
    assert ptxas is not None and ptxas.is_file() and os.access(ptxas, os.X_OK)
    version = subprocess.run(
        [str(ptxas), "--version"], capture_output=True, text=True, check=True
    ).stdout
    assert "release 12.8" in version


def _import_and_read(env_value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != _ptxas.ENV_VAR}
    if env_value is not None:
        env[_ptxas.ENV_VAR] = env_value
    code = (
        "import os, torch_mojo_backend; "
        f"print('PTXAS=' + os.environ.get({_ptxas.ENV_VAR!r}, ''))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    # The import may print its own lines (verbose mode); ours is tagged.
    return next(line for line in out.splitlines() if line.startswith("PTXAS="))[6:]


def test_import_sets_the_default_when_unset():
    assert _import_and_read(None) == str(_ptxas.wheel_ptxas())


def test_an_explicit_setting_wins():
    assert _import_and_read("/custom/ptxas") == "/custom/ptxas"
