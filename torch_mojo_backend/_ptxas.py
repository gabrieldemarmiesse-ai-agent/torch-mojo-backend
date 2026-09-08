"""Default MAX to the CUDA 12.8 ptxas from the nvidia-cuda-nvcc-cu12 wheel.

MAX bundles the newest CUDA's ptxas, and the cubins it assembles need a
driver at least as new (r580 for CUDA 13): on an older driver MAX refuses to
create a device at all. The wheel pinned in pyproject.toml ships ptxas 12.8,
whose cubins every r570+ driver loads, so it is the default for every
``mojo build`` subprocess and for MAX's in-process compiler. The package
imports this module first, before ``max`` loads, so the setting is in place
before anything reads it. An explicit ``MODULAR_NVPTX_COMPILER_PATH`` always
wins.
"""

import importlib.util
import os
from pathlib import Path

ENV_VAR = "MODULAR_NVPTX_COMPILER_PATH"


def wheel_ptxas() -> Path | None:
    """The ptxas shipped by nvidia-cuda-nvcc-cu12, or None when not installed."""
    spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "bin" / "ptxas"
        if candidate.is_file():
            return candidate
    return None


def apply_default():
    """Point MAX at the wheel's ptxas unless the user chose one."""
    if os.environ.get(ENV_VAR):
        return
    ptxas = wheel_ptxas()
    if ptxas is not None:
        os.environ[ENV_VAR] = str(ptxas)


apply_default()
