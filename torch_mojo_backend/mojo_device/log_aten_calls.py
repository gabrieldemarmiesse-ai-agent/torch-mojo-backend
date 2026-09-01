from collections.abc import Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class LoggingMode(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: Sequence[type],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        print(f"Aten function called: {func}")
        return func(*args, **kwargs or {})


def log_aten_calls() -> None:
    LoggingMode().__enter__()
