import warnings
from collections.abc import Callable, Mapping, Sequence

import torch
from max.driver import CPU, Accelerator, Device, accelerator_count


def get_accelerators() -> list[Device]:
    result = []
    if accelerator_count() > 0:
        for i in range(accelerator_count()):
            try:
                result.append(Accelerator(i))
            except ValueError as e:
                warnings.warn(f"Failed to create accelerator {i}. {e}")
    # This way, people can do torch.device("mojo:0") even if there is
    # no accelerator and get gpu or cpu automatically.
    result.append(CPU())
    return result


def get_fully_qualified_name(func: Callable[..., object] | str) -> str:
    if isinstance(func, str):
        return f"torch.Tensor.{func}"
    result = ""
    module = getattr(func, "__module__", None)
    if isinstance(module, str):
        result += module + "."

    qualname = getattr(func, "__qualname__", None)
    if isinstance(qualname, str):
        result += qualname

    result += " of type " + str(type(func)) + " "
    return result


def get_error_message(
    node: torch.fx.Node,
    node_idx: int,
    func_args: Sequence[object],
    func_kwargs: Mapping[str, object],
) -> str:
    if node.stack_trace is None:
        stack_trace = "No stack trace available, likely because this node is the result of a decomposition."
    else:
        stack_trace = node.stack_trace
    return (
        f"Failing at node {node_idx} when executing function {get_fully_qualified_name(node.target)}. "
        f"inputs of node were: args={func_args}, kwargs={func_kwargs}. "
        f"You can open an issue at https://github.com/gabrieldemarmiesse/torch-mojo-backend/issues . "
        f"It comes from there in your code: \n"
        f"{stack_trace}\n"
    )
