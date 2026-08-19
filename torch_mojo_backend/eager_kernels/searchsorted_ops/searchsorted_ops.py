from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class SearchsortedExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("searchsorted_ops/searchsorted_ops.mojo")
