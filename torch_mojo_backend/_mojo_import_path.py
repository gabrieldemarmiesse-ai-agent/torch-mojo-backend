"""Point MAX's in-process graph compiler at this package's Mojo sources.

`tmb/graph/gemm.mojo` and `nn.mojo` register MAX custom ops whose bodies are
the eager kernels, imported by name (`from tmb.kernels.matmul.entry import
...`). A precompiled Mojo package keeps such imports unresolved, and the graph
compiler resolves them when it compiles a graph -- along
`MODULAR_MOJO_MAX_IMPORT_PATH`, the comma-separated import path the whole Mojo
toolchain reads, which by default names only MAX's own `lib/mojo`. Setting the
variable REPLACES that default rather than extending it (measured: MAX then
fails to find its built-in kernel package), so the default is put back first,
from the SDK's own helper, and the Mojo source root
(`native.mojo_import_roots()`) is appended. A value the user set is extended,
never replaced. The variable has to be in the environment before `max` reads
it, so `torch_mojo_backend/__init__.py` imports this module first, the way it
imports `_ptxas`.

Every `mojo` the package runs inherits the variable too, and its entries
outrank `-I`. That is safe here because the root holds Mojo sources only and
one package, `tmb`: nothing on it can shadow a toolchain module.
"""

from __future__ import annotations

import os

from mojo.run import _sdk_default_env

from torch_mojo_backend import native

IMPORT_PATH_ENV_VAR = "MODULAR_MOJO_MAX_IMPORT_PATH"
_SEPARATOR = ","


def configure():
    """Extend the Mojo import path with this package's Mojo source root."""
    roots = [str(root.resolve()) for root in native.mojo_import_roots()]
    current = os.environ.get(IMPORT_PATH_ENV_VAR)
    if current:
        entries = current.split(_SEPARATOR)
    else:
        default = _sdk_default_env().get(IMPORT_PATH_ENV_VAR)
        if default is None:
            # An SDK layout the helper does not know (Bazel): leave MAX's
            # own resolution alone rather than break it; the graph backend
            # then reports the unresolved import at its first native op.
            return
        entries = [default]
    for root in roots:
        if root not in entries:
            entries.append(root)
    os.environ[IMPORT_PATH_ENV_VAR] = _SEPARATOR.join(entries)


configure()
