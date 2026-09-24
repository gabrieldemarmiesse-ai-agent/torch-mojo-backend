"""The shim builder must import without MAX, as in manylinux CI."""

import subprocess
import sys
from pathlib import Path


def test_shim_builder_imports_without_max():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_prebuilt.py"
    code = """
import runpy
import sys

# Model the torch-only build venv in a fresh interpreter, even when the
# test environment has MAX installed.
sys.modules["max"] = None
builder = runpy.run_path(sys.argv[1])
native = builder["_native_module"]()
assert native.prebuilt_shim_spec()["kind"] == "shim"
assert not any(name.startswith("max.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
