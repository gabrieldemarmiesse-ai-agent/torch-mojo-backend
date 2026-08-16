from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class RandomExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("random_ops/random_ops.mojo")
