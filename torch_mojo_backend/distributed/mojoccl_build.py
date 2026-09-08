"""Builds libmojoccl.so on first use and returns its cached path.

Reuses the eager-kernel loader's ungated-build machinery (`_build_extension`
in `torch_mojo_backend/eager_kernels/__init__.py`): the same file-lock,
atomic-rename and `__mojocache__` cache tensor_holder.mojo builds through,
just pointed at `mojoccl.mojo` instead. That path builds a Python CPython
extension normally, but `_build_extension` itself only runs `mojo build
--emit shared-lib` and returns the output path -- nothing about it assumes a
`PyInit_*` symbol, so it works unchanged for a plain C-ABI shared library
loaded with ctypes rather than importlib.
"""

import threading
from pathlib import Path

from torch_mojo_backend.eager_kernels import _build_extension

_ENTRY = Path(__file__).parent / "mojoccl" / "mojoccl.mojo"

_LOCK = threading.Lock()
_CACHED: list[Path] = []


def ensure_built() -> str:
    """Build (if needed) and return the path to libmojoccl.so."""
    with _LOCK:
        if _CACHED:
            return str(_CACHED[0])
        path = _build_extension(_ENTRY, None)
        _CACHED.append(path)
        return str(path)
