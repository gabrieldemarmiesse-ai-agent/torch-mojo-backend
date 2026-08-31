"""torch.Stream / torch.Event support on the mojo device (mojo_device/streams.py)."""

import pytest
import torch

from torch_mojo_backend import register_mojo_devices


def test_dispatch_installed_and_cpu_delegation():
    register_mojo_devices()
    assert getattr(torch.Stream, "_torch_mojo_backend", False)
    assert getattr(torch.Event, "_torch_mojo_backend", False)
    cpu_stream = torch.Stream(device="cpu")
    assert isinstance(cpu_stream, torch.Stream)
    from torch_mojo_backend.mojo_device.streams import Stream as MojoStream

    assert not isinstance(cpu_stream, MojoStream)


def test_stream_construction_and_identity(mojo_gpu: str):
    stream = torch.Stream(device=mojo_gpu)
    assert isinstance(stream, torch.Stream)
    from torch_mojo_backend.mojo_device.streams import Stream as MojoStream

    assert isinstance(stream, MojoStream)
    assert stream.device == torch.device("mojo", 0)
    assert stream.device_index == 0
    assert stream.device_type == "mojo"
    assert stream.native_handle != 0
    assert stream.stream_id == stream.native_handle
    assert stream.is_capturing() is False
    assert stream != torch.mojo.default_stream()
    assert stream == stream


def test_current_stream_and_context_manager(mojo_gpu: str):
    default = torch.mojo.default_stream()
    assert torch.accelerator.current_stream() == default
    assert torch.mojo.current_stream() == default
    side = torch.Stream(device=mojo_gpu)
    with side:
        assert torch.accelerator.current_stream() == side
        assert torch.mojo.current_stream() == side
    assert torch.accelerator.current_stream() == default
    torch.mojo.set_stream(side)
    assert torch.mojo.current_stream() == side
    torch.mojo.set_stream(default)
    assert torch.mojo.current_stream() == default


def test_documented_device_agnostic_pattern(mojo_gpu: str):
    stream = torch.Stream(device=torch.accelerator.current_accelerator())
    current = torch.accelerator.current_stream()
    stream.wait_stream(current)
    with stream:
        assert torch.accelerator.current_stream() == stream
    event = stream.record_event()
    assert isinstance(event, torch.Event)
    current.wait_event(event)
    stream.synchronize()
    assert stream.query()


def test_wait_stream_orders_real_work(mojo_gpu: str):
    x = torch.randn(2048, 2048, device=mojo_gpu)
    y = (x * 1.5 + 0.25).tanh()
    from torch_mojo_backend.mojo_device import deferred_compile

    deferred_compile.drain()
    side = torch.Stream(device=mojo_gpu)
    side.wait_stream(torch.mojo.current_stream())
    event = side.record_event()
    event.synchronize()
    assert event.query()
    torch.testing.assert_close(
        y.cpu(), (x.cpu() * 1.5 + 0.25).tanh(), atol=1e-4, rtol=1e-4
    )


def test_event_semantics(mojo_gpu: str):
    unrecorded = torch.Event(device=mojo_gpu)
    assert unrecorded.query() is True
    unrecorded.synchronize()  # no-op by contract
    with pytest.raises(RuntimeError):
        unrecorded.wait(torch.mojo.default_stream())
    with pytest.raises(NotImplementedError):
        torch.Event(device=mojo_gpu, interprocess=True)

    start = torch.Event(device=mojo_gpu, enable_timing=True)
    end = torch.Event(device=mojo_gpu, enable_timing=True)
    stream = torch.mojo.default_stream()
    torch.mojo.synchronize()
    start.record(stream)
    x = torch.randn(1024, 1024, device=mojo_gpu)
    (x * 2.0).cpu()  # forces the work through the queue and the device
    end.record(stream)
    end.synchronize()
    assert start.query() and end.query()
    assert start.elapsed_time(end) >= 0.0
    assert end.device == torch.device("mojo", 0)

    untimed = torch.Event(device=mojo_gpu)
    untimed.record(stream)
    with pytest.raises(RuntimeError):
        untimed.elapsed_time(end)
