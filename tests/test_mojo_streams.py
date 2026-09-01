"""Side device streams and the cross-stream free fence."""

import torch


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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.mojo_device import deferred_compile
    from torch_mojo_backend.mojo_device.device_streams import get_stream, record_use
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
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
        source_ptr = source._ptr
        for k in range(copies):
            eager_kernels.tensor_holder.copy_d2d(
                stream.ctx_ptr, sink._ptr + k * n * 4, source_ptr, n * 4
            )
        record_use(source._holder, stream)
        del source  # free is now fenced behind the stream's copies
        overwriter = torch.full((n,), 2.0, device=mojo_gpu)
        deferred_compile.drain()
        reused_any = reused_any or overwriter._ptr == source_ptr
        stream.synchronize()
        torch.mojo.synchronize()
        assert int((sink.cpu() != 1.0).sum().item()) == 0
        del overwriter, sink
    assert reused_any, "allocator never reused the freed block; test inconclusive"
