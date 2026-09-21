from typing import Literal

import torch
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.torch.torch import torch_dtype_to_max
from max.graph import Dim, TensorType

from torch_mojo_backend.torch_compile_backend import compiler
from torch_mojo_backend.types import MaxTensor, Scalar


def _scalar_to_tensor(input: MaxTensor, other: Scalar) -> MaxTensor:
    # `Scalar` also covers a symbolic Dim for ops that legitimately take one;
    # the bitwise ops that call this never do (ATen's Scalar there is a
    # genuine number), and F.constant only accepts a real number.
    if isinstance(other, Dim):
        raise TypeError(f"bitwise scalar ops expect a number, got a Dim: {other!r}")
    return F.broadcast_to(
        F.constant(other, dtype=input.dtype, device=input.device), input.shape
    )


def bitwise_and(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and operation.
    """

    return F.custom(
        name="bitwise_and",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_and_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and_scalar operation.
    """
    return bitwise_and(input, _scalar_to_tensor(input, other))


def bitwise_not(input: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_not operation.
    """

    return F.custom(
        name="bitwise_not",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_or(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or operation.
    """

    return F.custom(
        name="bitwise_or",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_or_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or_scalar operation.
    """
    return bitwise_or(input, _scalar_to_tensor(input, other))


def bitwise_xor(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor operation.
    """

    return F.custom(
        name="bitwise_xor",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_xor_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor_scalar operation.
    """
    return bitwise_xor(input, _scalar_to_tensor(input, other))


def elementwise(
    input: MaxTensor,
    kind: Literal[
        "abs",
        "acos",
        "asinh",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "erf",
        "exp",
        "floor",
        "gelu_none",
        "gelu_tanh",
        "isnan",
        "logical_not",
        "log",
        "log1p",
        "log2",
        "neg",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sign",
        "silu",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ],
) -> MaxTensor:
    """Call shared unary math through MAX's fusible Mojo registrations."""
    if (
        kind
        not in {
            "abs",
            "ceil",
            "floor",
            "gelu_none",
            "gelu_tanh",
            "isnan",
            "logical_not",
            "neg",
            "relu",
            "sign",
            "silu",
        }
        and not input.dtype.is_float()
    ):
        # ATen unary_float_op promotes integer and bool inputs to the default
        # floating dtype, whereas ElementwiseUnaryOp preserves its input dtype.
        input = F.cast(input, dtype=torch_dtype_to_max(torch.get_default_dtype()))
    output_dtype = DType.bool if kind in {"isnan", "logical_not"} else input.dtype
    return F.custom(
        name=f"elementwise_{kind}",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=output_dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.paths_to_mojo_kernels,
    )[0]


def gelu_backward(
    grad_output: MaxTensor, input: MaxTensor, *, approximate: str = "none"
) -> MaxTensor:
    """
    Custom Mojo kernel for gelu_backward operation.
    """
    kernel_name = "gelu_backward_tanh" if approximate == "tanh" else "gelu_backward"
    return F.custom(
        name=kernel_name,
        device=input.device,
        values=[grad_output, input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.paths_to_mojo_kernels,
    )[0]
