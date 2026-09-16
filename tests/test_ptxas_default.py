"""``torch_mojo_backend`` sets ``MODULAR_NVPTX_COMPILER_PATH`` at import,
unless the variable is already set, to the newest assembler on the machine
that this driver loads -- or leaves it unset when that is MAX's own compiler.
The nvidia-cuda-nvcc-cu12 wheel is a development dependency and one of the
candidates.

That is the import-time half, which knows only the driver. What happens when
that choice does not suit this machine's GPU is
tests/test_ptxas_selection.py."""

import os
import subprocess
import sys

import pytest

from torch_mojo_backend import _ptxas

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the nvcc wheel is a Linux dev dependency"
)


def test_wheel_ships_a_cuda_12_8_ptxas():
    ptxas = _ptxas.wheel_ptxas()
    assert ptxas is not None and ptxas.is_file() and os.access(ptxas, os.X_OK)
    version = subprocess.run(
        [str(ptxas), "--version"], capture_output=True, text=True, check=True
    ).stdout
    assert "release 12.8" in version


def _import_and_read(env_value: str | None, var: str | None = None) -> str:
    dropped = (_ptxas.ENV_VAR, _ptxas.AUTO_ENV_VAR)
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    if env_value is not None:
        env[_ptxas.ENV_VAR] = env_value
    code = (
        "import os, torch_mojo_backend; "
        f"print('PTXAS=' + os.environ.get({var or _ptxas.ENV_VAR!r}, ''))"
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


def _expected() -> _ptxas.Ptxas:
    chosen, _ = _ptxas.choose(_ptxas.driver_cuda_version())
    assert chosen is not None, "this machine has to have an assembler that fits"
    return chosen


def test_import_sets_the_default_when_unset():
    expected = _expected()
    path = "" if expected.is_builtin else str(expected.path)
    assert _import_and_read(None) == path


def test_an_explicit_setting_wins():
    assert _import_and_read("/custom/ptxas") == "/custom/ptxas"


def test_the_default_is_marked_as_ours_for_the_children_to_read():
    """A child inherits the environment and nothing else, so the pick is
    marked there: without it every torchrun rank would read an inherited
    default as a setting of the user's (tests/test_ptxas_selection.py)."""
    assert _import_and_read(None, _ptxas.AUTO_ENV_VAR) == _expected().mark
    assert _import_and_read("/custom/ptxas", _ptxas.AUTO_ENV_VAR) == ""
