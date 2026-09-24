import os

POSITIVE_VALUES = ("1", "true", "yes")


def profiling_enabled() -> bool:
    """
    Check if profiling is enabled by looking for the environment variable.
    """
    x = os.environ.get("TORCH_MOJO_BACKEND_PROFILE", "0").lower()
    py_x = os.environ.get("PYTORCH_MOJO_BACKEND_PROFILE", "0").lower()

    return x in POSITIVE_VALUES or py_x in POSITIVE_VALUES


def verbose_enabled() -> bool:
    """
    Check if verbose mode is enabled by looking for the environment variable.
    """
    x = os.environ.get("TORCH_MOJO_BACKEND_VERBOSE", "0").lower()
    py_x = os.environ.get("PYTORCH_MOJO_BACKEND_VERBOSE", "0").lower()

    return x in POSITIVE_VALUES or py_x in POSITIVE_VALUES


def debug_graph() -> bool:
    """
    Check if graph debugging is enabled by looking for the environment variable.
    """
    x = os.environ.get("TORCH_MOJO_BACKEND_DEBUG_GRAPH", "0").lower()
    py_x = os.environ.get("PYTORCH_MOJO_BACKEND_DEBUG_GRAPH", "0").lower()

    return x in POSITIVE_VALUES or py_x in POSITIVE_VALUES


def compile_native_kernels_enabled() -> bool:
    """Whether the torch.compile backend calls this repository's eager kernels
    (`tmb/graph/gemm.mojo`, `nn.mojo`) for the ops that have one -- the GEMMs, softmax,
    layer norm, embedding -- instead of composing MAX's own ops.
    `TORCH_MOJO_BACKEND_COMPILE_NATIVE_KERNELS=0` keeps MAX's kernels.
    """
    x = os.environ.get("TORCH_MOJO_BACKEND_COMPILE_NATIVE_KERNELS", "1").lower()
    return x not in ("0", "false", "no")
