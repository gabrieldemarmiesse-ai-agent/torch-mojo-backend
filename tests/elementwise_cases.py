"""Shared numerical edge inputs for native and compiled unary tests."""

import torch


def log1p_edge_input(dtype: torch.dtype) -> torch.Tensor:
    """Probe near zero, the domain boundary, and the full finite exponent range."""
    one = torch.tensor(1, dtype=dtype)
    zero = torch.tensor(0, dtype=dtype)
    smallest = torch.nextafter(zero, one).item()
    tiny = torch.finfo(dtype).tiny
    eps = torch.finfo(dtype).eps
    inside = torch.nextafter(-one, zero).item()
    outside = torch.nextafter(-one, -torch.tensor(float("inf"), dtype=dtype)).item()
    edges = torch.tensor(
        [
            -float("inf"),
            -2,
            outside,
            -1,
            inside,
            -0.5,
            -0.01,
            -0.0001,
            -eps,
            -eps / 2,
            -eps / 4,
            -2 * tiny,
            -tiny,
            -smallest,
            -0.0,
            0.0,
            smallest,
            tiny,
            2 * tiny,
            eps / 4,
            eps / 2,
            eps,
            0.0001,
            0.01,
            0.5,
            1,
            2,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
    )
    positive = torch.logspace(-45, 38, 10001, dtype=torch.float64)
    finite = torch.cat(
        (
            positive,
            -positive[positive < 1],
            torch.linspace(-0.9999999, 4, 10001, dtype=torch.float64),
        )
    ).to(dtype)
    return torch.cat((finite, edges))


def log1p_rtol(dtype: torch.dtype) -> float:
    return {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}[dtype]
