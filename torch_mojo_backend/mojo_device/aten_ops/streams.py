"""Stream-lifetime ops: this backend's answer to ``record_stream``."""

import torch

from torch_mojo_backend.mojo_device import channels
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

from .support import _unsupported


# aten::record_stream(Tensor(a!) self, Stream s) -> ()
def mojo_device_record_stream(self: TorchMojoTensor, s: torch._C.Stream) -> None:
    """Order ``self``'s eventual free after work already on stream ``s``.

    ``record_stream`` exists because a caching allocator frees
    stream-ordered on ONE stream: a buffer handed to another stream can be
    recycled while that stream still reads it. MAX has the same rule and
    this backend already has the remedy — ``channels.record_use`` records an
    event on the foreign stream and the holder's destructor fences the free
    behind it (mojo_device/channels.py). So the aten op is a thin adapter:
    take the ``c10::Stream``'s id, which for mojo streams *is* the native
    stream handle, and record against it.

    Recording against the device's own default stream is a no-op, and that
    is the common case here: eager kernels always execute on the default
    stream even inside ``with torch.mojo.stream(side)``, so a library
    pipelining on side streams (FSDP1 does) records buffers against streams
    that carry none of the work. The fence is still enqueued honestly, and
    it becomes load-bearing the day kernel launches follow the current
    stream.
    """
    if not isinstance(self, TorchMojoTensor):
        raise _unsupported("aten::record_stream", (self, s))
    device = self._device
    if getattr(device, "api", None) != "cuda":
        # Channels (and driver events) are CUDA-only; on other backends
        # everything already shares one stream, so there is nothing to fence.
        return
    channels.record_use_on_handle(
        self._holder,
        int(s.stream_id),
        device.default_stream.native_stream_handle,
        device.id,
    )
