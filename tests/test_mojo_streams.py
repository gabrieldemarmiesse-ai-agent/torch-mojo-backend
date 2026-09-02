"""torch.Stream / torch.Event support (mojo_device/streams.py) and the
underlying device streams / free fence (mojo_device/device_streams.py)."""

import pytest
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device import torch_mojo_device_module as torch_mojo
from torch_mojo_backend.mojo_device.streams import Stream as MojoStream
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor


def _mt(tensor: torch.Tensor) -> TorchMojoTensor:
    assert isinstance(tensor, TorchMojoTensor)
    return tensor


def _fenced_streams(tensor: torch.Tensor) -> set[int] | None:
    events = _mt(tensor)._holder._events
    return None if events is None else set(events)


def test_dispatch_installed_and_cpu_delegation():
    register_mojo_devices()
    # `torch.mojo` is registered onto `torch` at runtime; no stub knows it.
    assert torch.mojo is torch_mojo  # ty: ignore[unresolved-attribute]
    assert getattr(torch.Stream, "_torch_mojo_backend", False)
    assert getattr(torch.Event, "_torch_mojo_backend", False)
    cpu_stream = torch.Stream(device="cpu")
    assert isinstance(cpu_stream, torch.Stream)
    assert not isinstance(cpu_stream, MojoStream)


def test_stream_construction_and_identity(mojo_gpu: str):
    stream = torch.Stream(device=mojo_gpu)
    assert isinstance(stream, torch.Stream)
    assert isinstance(stream, MojoStream)
    assert stream.device == torch.device("mojo", 0)
    assert stream.device_index == 0
    assert stream.device_type == "mojo"
    assert stream.native_handle != 0
    assert stream.stream_id == stream._device_stream.ctx_ptr
    assert stream.stream_id != stream.native_handle  # a context, not a stream
    assert stream.is_capturing() is False
    assert stream != torch_mojo.default_stream()
    assert stream == stream


def test_current_stream_and_context_manager(mojo_gpu: str):
    default = torch_mojo.default_stream()
    assert torch.accelerator.current_stream() == default
    assert torch_mojo.current_stream() == default
    side = torch.Stream(device=mojo_gpu)
    assert isinstance(side, MojoStream)
    with side:
        assert torch.accelerator.current_stream() == side
        assert torch_mojo.current_stream() == side
    assert torch.accelerator.current_stream() == default
    torch_mojo.set_stream(side)
    assert torch_mojo.current_stream() == side
    torch_mojo.set_stream(default)
    assert torch_mojo.current_stream() == default


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
    side.wait_stream(torch_mojo.current_stream())
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
        unrecorded.wait(torch_mojo.default_stream())
    with pytest.raises(NotImplementedError):
        torch.Event(device=mojo_gpu, interprocess=True)

    start = torch.Event(device=mojo_gpu, enable_timing=True)
    end = torch.Event(device=mojo_gpu, enable_timing=True)
    stream = torch_mojo.default_stream()
    torch_mojo.synchronize()
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


def test_stream_is_a_real_torch_stream_for_cpp_argument_parsing(mojo_gpu: str):
    """THPStream_Check requires the concrete torch._C.Stream type, not duck typing."""
    stream = torch_mojo.current_stream()
    assert isinstance(stream, torch._C.Stream)
    assert stream.stream_id == stream._device_stream.ctx_ptr


def test_record_stream_accepts_mojo_streams(mojo_gpu: str):
    tensor = torch.ones(1024, device=mojo_gpu)
    # The device's own stream owns the free already: nothing to fence.
    tensor.record_stream(torch_mojo.current_stream())
    assert _fenced_streams(tensor) is None

    side = torch_mojo.Stream()
    tensor.record_stream(side)
    assert _fenced_streams(tensor) == {side.stream_id}

    # Recording the same stream twice keeps one (newest) event for it.
    tensor.record_stream(side)
    assert _fenced_streams(tensor) == {side.stream_id}
    assert tensor.cpu().sum().item() == 1024.0


def test_record_stream_under_no_dispatch(mojo_gpu: str):
    """record_stream must work under no_dispatch(), as torch.distributed calls it."""
    from torch.utils._mode_utils import no_dispatch

    tensor = torch.ones(64, device=mojo_gpu)
    side = torch_mojo.Stream()
    with no_dispatch():
        tensor.record_stream(side)
    assert _fenced_streams(tensor) == {side.stream_id}


def test_side_stream_is_distinct_from_default(mojo_gpu: str):
    """A side stream is a real second stream, not an alias of the default."""
    from torch_mojo_backend.mojo_device.device_streams import (
        default_stream,
        default_stream_ctx_ptr,
        get_stream,
    )
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        find_equivalent_max_device,
    )

    device = find_equivalent_max_device(torch.device("mojo", 0))
    owner_ctx = default_stream_ctx_ptr(device)
    assert device.default_stream._device_context_ptr() == owner_ctx
    assert default_stream(device).ctx_ptr == owner_ctx

    side = get_stream(device, "distinctness-test")
    assert side.handle != device.default_stream.native_stream_handle
    assert side.ctx_ptr != owner_ctx
    assert get_stream(device, "distinctness-test") is side  # cached per name
    assert get_stream(device, "distinctness-test-2").ctx_ptr != side.ctx_ptr


def test_record_use_fences_free_against_side_stream_reader(mojo_gpu: str):
    """record_use fences a free so a side-stream reader can't race pool reuse."""
    from torch_mojo_backend.mojo_device import deferred_compile
    from torch_mojo_backend.mojo_device.device_streams import get_stream, record_use
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        _holder_mod,
        find_equivalent_max_device,
    )

    device = find_equivalent_max_device(torch.device("mojo", 0))
    stream = get_stream(device, "lifetime-test")
    n = 8 * 1024 * 1024
    copies = 4

    reused_any = False
    for _ in range(10):
        source = torch.full((n,), 1.0, device=mojo_gpu)
        sink = torch.empty((copies * n,), device=mojo_gpu)
        deferred_compile.drain()
        stream.wait_default_stream()
        source_ptr = _mt(source)._ptr
        for k in range(copies):
            _holder_mod().copy_d2d(
                stream.ctx_ptr, _mt(sink)._ptr + k * n * 4, source_ptr, n * 4
            )
        record_use(_mt(source)._holder, stream)
        del source  # free is now fenced behind the stream's copies
        overwriter = torch.full((n,), 2.0, device=mojo_gpu)
        deferred_compile.drain()
        reused_any = reused_any or _mt(overwriter)._ptr == source_ptr
        stream.synchronize()
        torch_mojo.synchronize()
        assert int((sink.cpu() != 1.0).sum().item()) == 0
        del overwriter, sink
    assert reused_any, "allocator never reused the freed block; test inconclusive"
