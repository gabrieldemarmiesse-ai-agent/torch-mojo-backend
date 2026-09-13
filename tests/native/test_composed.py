"""Ops composed from registered ops through the dispatcher (ops_composed.mojo)."""

import pytest
import torch


@pytest.mark.parametrize("act", ["relu", "sigmoid", "tanh"])
def test_activation_backward_composed_through_the_dispatcher(mojo_gpu, act):
    """threshold/sigmoid/tanh backward have no kernel of their own; they are
    composed from registered ops and must match CPU autograd."""
    x = torch.randn(4, 7, device=mojo_gpu, requires_grad=True)
    y = getattr(torch, act)(x)
    grad = torch.randn_like(y)
    y.backward(grad)
    ref = x.detach().cpu().requires_grad_(True)
    getattr(torch, act)(ref).backward(grad.cpu())
    assert x.grad is not None and ref.grad is not None
    torch.testing.assert_close(x.grad.cpu(), ref.grad, atol=1e-5, rtol=1e-5)


def test_relu_module_trains(mojo_gpu):
    layer = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()).to(mojo_gpu)
    out = layer(torch.randn(3, 8, device=mojo_gpu)).sum()
    out.backward()
    assert layer[0].weight.grad is not None


def test_isneginf_isposinf(mojo_device):
    x = torch.tensor([float("-inf"), -1.0, 0.0, float("inf"), float("nan")]).to(
        mojo_device
    )
    assert torch.isneginf(x).cpu().tolist() == [True, False, False, False, False]
    assert torch.isposinf(x).cpu().tolist() == [False, False, False, True, False]
    out = torch.empty(5, dtype=torch.bool, device=mojo_device)
    torch.isneginf(x, out=out)
    assert out.cpu().tolist() == [True, False, False, False, False]
    assert not torch.isposinf(torch.arange(3, device=mojo_device)).cpu().any()
