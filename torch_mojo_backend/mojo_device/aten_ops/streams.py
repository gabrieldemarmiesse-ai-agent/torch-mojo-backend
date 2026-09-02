"""Stream-lifetime ops: this backend's answer to ``record_stream``."""

import torch

from torch_mojo_backend.mojo_device import device_streams
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

from torch_mojo_backend.mojo_device.aten_ops.support import _unsupported


# aten::record_stream(Tensor(a!) self, Stream s) -> ()
def mojo_device_record_stream(self: TorchMojoTensor, s: torch._C.Stream):
    """Order ``self``'s eventual free after work already on stream ``s``.

    A thin adapter onto ``device_streams.record_use_on_stream_ctx``: the
    ``c10::Stream``'s id *is* the mojo stream's MAX ``DeviceContext``
    pointer.

    Recording against the default stream is a no-op, and that is the common
    case here: eager kernels always execute on the default stream even
    inside ``with torch.mojo.stream(side)``, so a library pipelining on side
    streams (FSDP1 does) records buffers against streams that carry none of
    the work yet. The fence is still enqueued honestly and becomes
    load-bearing once kernel launches follow the current stream.
    """
    if not isinstance(self, TorchMojoTensor):
        raise _unsupported("aten::record_stream", (self, s))
    device = self._device
    if getattr(device, "api", None) == "cpu":
        # The CPU device has one stream; there is nothing to fence.
        return
    device_streams.record_use_on_stream_ctx(
        self._holder, int(s.stream_id), device_streams.default_stream_ctx_ptr(device)
    )
