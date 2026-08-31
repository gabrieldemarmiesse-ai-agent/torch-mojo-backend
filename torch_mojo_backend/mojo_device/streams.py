"""torch.Stream / torch.Event support for the mojo device.

PyTorch's generic ``torch.Stream``/``torch.Event`` route through a C++
device guard that is a stub for Python-backed PrivateUse1 devices: stream
construction always reports stream id 0 and record/wait/query are silent
no-ops. ``register_mojo_devices()`` therefore dispatches mojo devices to
the classes here instead — the same remedy the backend already applies to
``torch.accelerator.synchronize``. Every ordering primitive is real: user
streams are mojo-device channels (extra CUDA streams,
``mojo_device/channels.py``) and events are driver events recorded on the
streams' native handles.

Execution semantics, stated plainly: eager mojo kernels currently always
EXECUTE on the device's default stream. ``with stream:`` tracks the
current stream per thread (``torch.accelerator.current_stream()`` reports
it, and event/wait calls act on it) but does not yet redirect kernel
launches. Device-agnostic pipelining code therefore runs with the exact
semantics it would have on CUDA with a single stream — correct, without
the extra compute concurrency. Redirecting launches through per-stream MAX
DeviceContexts is the known follow-up.
"""

import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from torch_mojo_backend.mojo_device.channels import (
    Channel,
    CudaEvent,
    default_channel,
    ensure_context_current,
)

_user_stream_ids = itertools.count()
_current_stacks = threading.local()  # {device_index: [Stream, ...]} per thread
# No annotation: beartype's claw checks module-level annotated globals and
# rejects the forward reference to Stream (defined below).
_default_streams = {}  # {device_index: Stream}
_default_streams_lock = threading.Lock()


def _current_device_index() -> int:
    from torch_mojo_backend.mojo_device import torch_mojo_device_module

    return torch_mojo_device_module.current_device()


def _resolve_index(device: object) -> int:
    if device is None:
        return _current_device_index()
    if isinstance(device, int):
        return device
    resolved = torch.device(device)
    if resolved.type != "mojo":
        raise ValueError(f"expected a mojo device, got {resolved}")
    return resolved.index if resolved.index is not None else _current_device_index()


def _max_device_of(index: int) -> object:
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        find_equivalent_max_device,
    )

    return find_equivalent_max_device(torch.device("mojo", index))


class Event:
    """Mirror of torch.Event for mojo devices, backed by a driver event.

    The driver event is created lazily at first ``record`` so it lands on
    the recording stream's CUDA context. ``blocking`` is accepted and
    ignored (host waits always yield); ``interprocess`` is unsupported.
    """

    def __init__(
        self,
        device: object = None,
        *,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> None:
        if interprocess:
            raise NotImplementedError(
                "interprocess events are not supported on the mojo device"
            )
        self.enable_timing = enable_timing
        self._event: CudaEvent | None = None
        self._device_index: int | None = (
            None if device is None else _resolve_index(device)
        )

    @property
    def device(self) -> torch.device | None:
        if self._device_index is None:
            return None
        return torch.device("mojo", self._device_index)

    def record(self, stream: "Stream | None" = None) -> None:
        if stream is None:
            stream = current_stream(self._device_index)
        ensure_context_current(stream._channel.device.id)
        if self._event is None:
            self._event = CudaEvent(enable_timing=self.enable_timing)
        self._event.record(stream._channel.handle)
        self._device_index = stream.device_index

    def wait(self, stream: "Stream | None" = None) -> None:
        if self._event is None:
            raise RuntimeError("Event must be recorded before it can be waited on")
        if stream is None:
            stream = current_stream(self._device_index)
        stream._channel.wait_event(self._event)

    def query(self) -> bool:
        return True if self._event is None else self._event.query()

    def synchronize(self) -> None:
        if self._event is not None:
            self._event.synchronize()

    def elapsed_time(self, end_event: "Event") -> float:
        if not (self.enable_timing and end_event.enable_timing):
            raise RuntimeError("both events must be created with enable_timing=True")
        if self._event is None or end_event._event is None:
            raise RuntimeError("both events must be recorded before elapsed_time")
        return self._event.elapsed_time_ms(end_event._event)

    @classmethod
    def from_ipc_handle(cls, device: object, handle: bytes) -> "Event":
        raise NotImplementedError(
            "interprocess events are not supported on the mojo device"
        )

    def __repr__(self) -> str:
        return f"torch.mojo.Event(device={self.device}, recorded={self._event is not None})"


class Stream:
    """Mirror of torch.Stream for mojo devices, backed by a channel."""

    def __init__(
        self,
        device: object = None,
        priority: int = 0,
        *,
        _channel: Channel | None = None,
    ) -> None:
        index = _resolve_index(device)
        if _channel is None:
            _channel = Channel(_max_device_of(index), f"user-{next(_user_stream_ids)}")
        self._channel = _channel
        self._index = index
        # MAX's Python API exposes no stream priorities; accepted for
        # signature compatibility, always effectively 0.
        self.priority = 0

    @property
    def device(self) -> torch.device:
        return torch.device("mojo", self._index)

    @property
    def device_index(self) -> int:
        return self._index

    @property
    def device_type(self) -> str:
        return "mojo"

    @property
    def stream_id(self) -> int:
        return self._channel.handle

    @property
    def native_handle(self) -> int:
        return self._channel.handle

    def query(self) -> bool:
        return self._channel.query()

    def synchronize(self) -> None:
        self._channel.synchronize()

    def wait_event(self, event: Event) -> None:
        event.wait(self)

    def wait_stream(self, other: "Stream | torch.Stream") -> None:
        """Record a temp event on `other` and make this stream wait on it."""
        handle = _handle_of(other, self._index)
        ensure_context_current(self._channel.device.id)
        event = CudaEvent()
        event.record(handle)
        self._channel.wait_event(event)
        event.destroy()

    def record_event(self, event: Event | None = None) -> Event:
        if event is None:
            event = Event(device=self._index)
        event.record(self)
        return event

    def is_capturing(self) -> bool:
        return False  # no graph capture on the mojo device

    def __enter__(self) -> "Stream":
        stacks = getattr(_current_stacks, "stacks", None)
        if stacks is None:
            stacks = _current_stacks.stacks = {}
        stacks.setdefault(self._index, []).append(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        _current_stacks.stacks[self._index].pop()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Stream) and other._channel.handle == self._channel.handle
        )

    def __hash__(self) -> int:
        return hash(self._channel.handle)

    def __repr__(self) -> str:
        return (
            f"torch.mojo.Stream(device=mojo:{self._index}, "
            f"handle=0x{self._channel.handle:x}"
            f"{', default' if self._channel.is_default else ''})"
        )


def _handle_of(stream: object, index: int) -> int:
    """The native handle of ours OR a stub torch.Stream (= default stream)."""
    if isinstance(stream, Stream):
        return stream._channel.handle
    # A guard-backed torch.Stream for a Python PrivateUse1 device can only
    # ever describe the default stream (the stub guard knows no other).
    return default_stream(index)._channel.handle


def default_stream(device: object = None) -> Stream:
    index = _resolve_index(device)
    with _default_streams_lock:
        stream = _default_streams.get(index)
        if stream is None:
            stream = Stream(
                device=index, _channel=default_channel(_max_device_of(index))
            )
            _default_streams[index] = stream
        return stream


def current_stream(device: object = None) -> Stream:
    index = _resolve_index(device)
    stacks = getattr(_current_stacks, "stacks", None)
    if stacks and stacks.get(index):
        return stacks[index][-1]
    return default_stream(index)


def set_stream(stream: Stream) -> None:
    """Make `stream` this thread's ambient current stream (no nesting)."""
    stacks = getattr(_current_stacks, "stacks", None)
    if stacks is None:
        stacks = _current_stacks.stacks = {}
    stacks[stream.device_index] = [stream]


@contextmanager
def stream(stream: "Stream | None") -> Iterator[None]:
    """Context-manager helper mirroring torch.cuda.stream(s)."""
    if stream is None:
        yield
        return
    with stream:
        yield
