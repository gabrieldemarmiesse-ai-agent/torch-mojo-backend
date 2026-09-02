import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import max.graph.ops as max_ops
import torch
from max.dtype import DType
from max.experimental.torch import CustomOpLibrary
from max.graph import TensorType, TensorValue

import torch_mojo_backend
import torch_mojo_backend.torch_compile_backend.compiler


class _SignedTorchOp(Protocol):
    """A callable carrying the `__signature__` `torch.library.custom_op`
    introspects; `functools`-style wrapping can't express the extra
    attribute, so the callable this wraps is cast to this structural type."""

    __signature__: inspect.Signature

    def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor): ...


def make_torch_op_from_mojo(
    path_to_kernels: Path,
    mojo_custom_op_str: str,
    allocate_outputs_fn: Callable[
        ..., torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    ],
) -> Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]:
    ops = CustomOpLibrary(path_to_kernels)
    mojo_custom_op = ops.__getattr__(mojo_custom_op_str)

    def compiler_fn(
        *args: TensorValue, **kwargs: bool | int | str | DType
    ) -> tuple[object, ...]:
        # We split args into outputs and inputs
        # We assume outputs are first
        out_args = args[: mojo_custom_op.num_outputs]
        in_args = args[mojo_custom_op.num_outputs :]

        out_types = [
            TensorType(dtype=x.dtype, shape=x.shape, device=x.device) for x in out_args
        ]
        return (
            NotImplementedError("Accessing the first output is not handled yet"),
            *max_ops.custom(
                mojo_custom_op.name,
                device=args[0].device,
                values=list(in_args),
                parameters=kwargs,
                out_types=out_types,
            ),
        )

    torch_mojo_backend.torch_compile_backend.compiler._global_max_objects = None

    def mojo_custom_op_with_signature(*args: torch.Tensor, **kwargs: torch.Tensor):
        mojo_custom_op(*args, **kwargs)

    signed_op = cast(_SignedTorchOp, mojo_custom_op_with_signature)
    signed_op.__signature__ = mojo_custom_op.torch_signature

    torch_mojo_backend.torch_compile_backend.compiler.paths_to_mojo_kernels.append(
        path_to_kernels
    )
    torch_mojo_backend.MAPPING_TORCH_ATEN_TO_MOJO[
        f"{path_to_kernels.name}.{mojo_custom_op_str}"
    ] = compiler_fn
    mutates_args = []
    for param_name in mojo_custom_op.torch_signature.parameters:
        mutates_args.append(param_name)
        if len(mutates_args) == mojo_custom_op.num_outputs:
            break

    torch_custom_op = torch.library.custom_op(
        f"{path_to_kernels.name}::{mojo_custom_op_str}", mutates_args=mutates_args
    )(signed_op)

    def fn(
        *args: object, **kwargs: object
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]:
        output_tensors = allocate_outputs_fn(*args, **kwargs)
        single_output: bool
        if isinstance(output_tensors, torch.Tensor):
            output_tensors = (output_tensors,)
            single_output = True
        elif isinstance(output_tensors, tuple) or isinstance(output_tensors, list):
            single_output = False
            for t in output_tensors:
                if not isinstance(t, torch.Tensor):
                    raise ValueError(
                        f"allocate_outputs_fn must return a torch."
                        f"Tensor or a tuple/list of torch.Tensor. Found a list/tuple with "
                        f"an element of type {type(t)}"
                    )
        else:
            raise ValueError(
                f"allocate_outputs_fn must return a torch.Tensor or a tuple/list of "
                f"torch.Tensor. Found {type(output_tensors)}"
            )

        torch_custom_op(*output_tensors, *args, **kwargs)
        if single_output:
            return output_tensors[0]
        return output_tensors

    return fn
