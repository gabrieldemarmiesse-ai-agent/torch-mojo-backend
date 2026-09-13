"""End-to-end checks of the native backend's core: factories, transfers,
views, fills, item, autograd, streams, events, RNG (public torch API only)."""

import pytest
import torch

from torch_mojo_backend import native
from torch_mojo_backend.native import device_module


def _arange(n: int, device: str) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).to(device)


def test_registration_and_devices(mojo_device):
    assert native.is_registered()
    assert device_module.device_count() >= 1
    assert device_module.cpu() == torch.device(
        f"mojo:{device_module.device_count() - 1}"
    )


def test_empty_to_and_back(mojo_device):
    x = torch.empty(2, 3, device=mojo_device)
    assert x.device.type == "mojo" and x.dtype == torch.float32 and x.is_contiguous()
    a = _arange(6, mojo_device).reshape(2, 3)
    assert a.cpu().tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_add_mul_views_item(mojo_device):
    a = _arange(6, mojo_device).reshape(2, 3)
    b = torch.full((2, 3), 2.0).to(mojo_device)
    torch.testing.assert_close((a + b).cpu(), a.cpu() + 2)
    torch.testing.assert_close((a * b).cpu(), a.cpu() * 2)
    assert a.view(6).cpu().tolist() == list(range(6))
    assert a.reshape(3, 2).cpu().tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert a[1, 2].item() == 5.0
    assert a.t().contiguous().cpu().tolist() == a.cpu().t().tolist()


def test_fills_and_strided_copies(mojo_device):
    assert torch.zeros(3, device=mojo_device).cpu().tolist() == [0.0] * 3
    assert torch.ones(2, dtype=torch.int64, device=mojo_device).cpu().tolist() == [1, 1]
    z = torch.empty(4, device=mojo_device)
    z.fill_(7.5)
    assert z.cpu().tolist() == [7.5] * 4
    v = torch.zeros(3, 4, device=mojo_device)
    v[:, 1] = 5.0
    assert v.cpu()[:, 1].tolist() == [5.0] * 3
    w = torch.zeros(3, 4, device=mojo_device)
    w[:, 1].copy_(_arange(3, mojo_device))
    assert w.cpu()[:, 1].tolist() == [0.0, 1.0, 2.0]


def test_dtype_cast(mojo_device):
    m = _arange(12, mojo_device).reshape(3, 4)
    torch.testing.assert_close(
        m.to(torch.bfloat16).float().cpu(), m.cpu().to(torch.bfloat16).float()
    )


def test_error_paths(mojo_device):
    a = _arange(6, mojo_device).reshape(2, 3)
    with pytest.raises(RuntimeError):
        a * _arange(5, mojo_device)


def test_autograd_uses_aten_formulas(mojo_device):
    x = _arange(4, mojo_device).requires_grad_()
    y = torch.full((4,), 3.0).to(mojo_device).requires_grad_()
    (x * y).backward(torch.ones(4).to(mojo_device))
    assert x.grad is not None and y.grad is not None
    assert x.grad.cpu().tolist() == [3.0] * 4
    assert y.grad.cpu().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_streams_and_events(mojo_gpu):
    s = torch.Stream(device=mojo_gpu)
    assert s.device.type == "mojo"
    assert s.stream_id != torch.accelerator.current_stream().stream_id
    a = _arange(6, mojo_gpu)
    with device_module.stream(s):
        e1 = torch.Event(device=mojo_gpu, enable_timing=True)
        e1.record()
        _ = a * a
        e2 = torch.Event(device=mojo_gpu, enable_timing=True)
        e2.record()
        assert torch.accelerator.current_stream().stream_id == s.stream_id
    e2.synchronize()
    assert e1.elapsed_time(e2) >= 0.0
    assert e2.query()
    torch.accelerator.synchronize()


def test_rng_state_and_generator(mojo_device):
    device_module.manual_seed_all(123)
    state = device_module.get_rng_state()
    assert state.dtype == torch.uint8 and state.numel() == 16
    assert int.from_bytes(bytes(state.tolist()[:8]), "little") == 123
    g = torch.Generator(device=mojo_device)
    g.manual_seed(5)
    assert g.initial_seed() == 5
    device_module.set_rng_state(state)
    assert device_module.get_rng_state().tolist() == state.tolist()


def test_device_oom_is_not_disguised_as_unsupported(mojo_gpu):
    """An allocation the device cannot satisfy must surface as an OOM
    carrying the allocator's own message -- not as `NotImplementedError`
    ("unsupported dtype/shape"), which would send the reader looking for a
    missing kernel, and not silently at some later synchronize.
    """
    with pytest.raises(Exception) as excinfo:  # noqa: B017 -- the point is WHICH type
        torch.empty(2**44, dtype=torch.float64, device=mojo_gpu).fill_(1.0)
    assert not isinstance(excinfo.value, NotImplementedError), excinfo.value
    assert isinstance(excinfo.value, (torch.OutOfMemoryError, RuntimeError)), (
        type(excinfo.value),
        excinfo.value,
    )
