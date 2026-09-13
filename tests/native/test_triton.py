"""Triton kernels on the mojo device (torch_mojo_backend.triton_driver): Triton
compiles and launches through its own CUDA backend; the driver only answers
the device and stream questions with the mojo device."""

import pytest
import torch

from torch_mojo_backend.triton_driver import enable_triton

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


@pytest.fixture
def mojo_triton(mojo_gpu):
    enable_triton()
    return mojo_gpu


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
        mask=mask,
    )


def test_triton_kernel_runs_on_mojo_tensors(mojo_triton):
    x = torch.randn(100_003, device=mojo_triton)
    y = torch.randn(100_003, device=mojo_triton)
    out = torch.empty_like(x)
    _add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    torch.testing.assert_close(out.cpu(), x.cpu() + y.cpu())


def test_triton_launch_follows_the_current_mojo_stream(mojo_triton):
    s = torch.Stream(device=mojo_triton)
    with torch.mojo.stream(s):
        a = torch.ones(1 << 22, device=mojo_triton) * 3
        b = torch.empty_like(a)
        _add_kernel[(triton.cdiv(a.numel(), 1024),)](a, a, b, a.numel(), BLOCK=1024)
        c = b * 2
    torch.accelerator.synchronize()
    assert float(c.sum().cpu()) == 12 * (1 << 22)
