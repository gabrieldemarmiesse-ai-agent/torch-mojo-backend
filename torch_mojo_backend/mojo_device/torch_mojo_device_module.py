import threading

import torch

from torch_mojo_backend.torch_compile_backend.utils import get_accelerators

_current_device = 0
_UINT64_MASK = (1 << 64) - 1
_DEFAULT_RNG_SEED = 67_280_421_310_721
_rng_default_seed = _DEFAULT_RNG_SEED
_rng_states: dict[int, tuple[int, int]] = {}
_rng_lock = threading.Lock()


def cpu() -> torch.device:
    return torch.device(f"mojo:{len(list(get_accelerators())) - 1}")


def _is_in_bad_fork() -> bool:
    return False


def _normalize_rng_seed(seed: int) -> int:
    value = int(seed)
    if value < -(1 << 63) or value > _UINT64_MASK:
        raise ValueError("Overflow when unpacking long long")
    return value & _UINT64_MASK


def _rng_device_index(device: "int | str | torch.device | None" = None) -> int:
    if device is None:
        index = _current_device
    elif isinstance(device, int):
        index = device
    else:
        torch_device = torch.device(device)
        if torch_device.type != "mojo":
            raise ValueError(f"expected a mojo RNG device, got {torch_device}")
        index = _current_device if torch_device.index is None else torch_device.index
    if index < 0 or index >= device_count():
        raise ValueError(f"Invalid device index {index}")
    return index


def manual_seed_all(seed: int):
    """Reset every Mojo device to the same Philox seed and counter zero."""
    global _rng_default_seed
    normalized = _normalize_rng_seed(seed)
    with _rng_lock:
        _rng_default_seed = normalized
        _rng_states.clear()
        for index in range(device_count()):
            _rng_states[index] = (normalized, 0)


def device_count() -> int:
    return len(list(get_accelerators()))


def get_rng_state(device: "int | str | torch.device | None" = None) -> torch.Tensor:
    """Return the selected device's exact ``(seed, counter)`` state."""
    index = _rng_device_index(device)
    with _rng_lock:
        seed, counter = _rng_states.setdefault(index, (_rng_default_seed, 0))
    encoded = seed.to_bytes(8, "little") + counter.to_bytes(8, "little")
    return torch.tensor(list(encoded), dtype=torch.uint8)


def set_rng_state(
    new_state: torch.Tensor, device: "int | str | torch.device | None" = None
):
    """Restore an exact state produced by :func:`get_rng_state`."""
    if not isinstance(new_state, torch.Tensor):
        raise TypeError("Mojo RNG state must be a torch.Tensor")
    state = new_state.detach().cpu().contiguous()
    if state.dtype != torch.uint8 or state.numel() != 16:
        raise ValueError("Mojo RNG state must be a 16-element uint8 tensor")
    encoded = bytes(state.reshape(-1).tolist())
    seed = int.from_bytes(encoded[:8], "little")
    counter = int.from_bytes(encoded[8:], "little")
    index = _rng_device_index(device)
    with _rng_lock:
        _rng_states[index] = (seed, counter)


def _reserve_philox_state(
    device: "int | str | torch.device | None", counter_increment: int
) -> tuple[int, int]:
    """Atomically reserve a per-device Philox counter interval.

    The caller passes the returned seed/base counter to an asynchronous device
    kernel. Reserving host state never inspects a tensor or synchronizes a
    device queue.
    """
    if type(counter_increment) is not int or counter_increment < 0:
        raise ValueError("Philox counter increment must be a nonnegative integer")
    index = _rng_device_index(device)
    with _rng_lock:
        seed, counter = _rng_states.setdefault(index, (_rng_default_seed, 0))
        if counter_increment > _UINT64_MASK - counter:
            raise OverflowError("Philox counter reservation would wrap uint64")
        _rng_states[index] = (seed, counter + counter_increment)
    return seed, counter


def is_available() -> bool:
    # Always true as there is at least the CPU
    return True


def is_initialized() -> bool:
    return True


def current_device() -> int:
    return _current_device


def set_device(device_idx: int):
    global _current_device
    if device_idx < 0 or device_idx >= device_count():
        raise ValueError(f"Invalid device index {device_idx}")
    _current_device = device_idx


class device:
    """Context manager that swaps the current mojo device, mirroring
    ``torch.cuda.device``. ``torch.serialization`` requires it on the backend
    module when loading a checkpoint with ``map_location="mojo"``
    (``torch._utils._to`` enters ``device_module.device(...)``)."""

    def __init__(self, device: "int | str | torch.device | None"):
        if device is None:
            self.idx = -1
            return
        if isinstance(device, int):
            self.idx = device
            return
        torch_device = torch.device(device)
        self.idx = -1 if torch_device.index is None else torch_device.index

    def __enter__(self):
        self.prev_idx = _current_device
        if self.idx >= 0 and self.idx != _current_device:
            set_device(self.idx)

    def __exit__(self, *exc_info: object) -> bool:
        if self.idx >= 0 and self.prev_idx != _current_device:
            set_device(self.prev_idx)
        return False


def _resolve_sync_device(device: "int | str | torch.device | None") -> torch.device:
    if device is None:
        return torch.device(f"mojo:{_current_device}")
    if isinstance(device, int):
        return torch.device(f"mojo:{device}")
    torch_device = torch.device(device)
    if torch_device.type == "mojo" and torch_device.index is None:
        return torch.device(f"mojo:{_current_device}")
    return torch_device


def _device_synchronize(device: "int | str | torch.device | None" = None):
    """Device-only barrier: wait for already-launched work and release the
    completed asynchronous transfer owners.

    Deliberately does NOT drain the kernel-call queue. This is the ordering
    primitive the queue itself uses when a launch must be barriered against
    another thread's device work (``call_queue._device_only_synchronize``,
    reached from ``order_direct_launch`` / ``_order_queue_launch_locked``),
    where a drain would re-enter the queue in the middle of a launch —
    running items 2..N before the item already popped, and freeing its
    keep-alive. It is also what that path
    actually needs: the queued items have not been launched at all, so there
    is nothing of theirs to wait for; only the *other* thread's issued work
    must land first, which is exactly a stream synchronize.
    """
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        _release_synchronized_d2h_owners,
        _release_synchronized_h2d_sources,
        find_equivalent_max_device,
    )

    max_device = find_equivalent_max_device(_resolve_sync_device(device))
    max_device.default_stream.synchronize()
    _release_synchronized_h2d_sources(max_device)
    _release_synchronized_d2h_owners(max_device)


def synchronize(device: "int | str | torch.device | None" = None):
    """Public: wait for work and release completed asynchronous transfer
    owners. Pending kernel launches count as work, so the queue drains
    first — a caller of ``torch.mojo.synchronize()`` is entitled to assume
    every op it issued has actually run on the device. So does a collective
    still flying on the comm stream, which only the default stream is waited
    on here: fence it onto that stream first (mojo_device/comm_fence.py)."""
    from torch_mojo_backend.mojo_device import comm_fence, deferred_compile

    comm_fence.fence_all()
    deferred_compile.drain()
    _device_synchronize(device)


def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.bfloat16]  # TODO change


def memory_stats(device: "int | str | torch.device | None" = None) -> dict[str, int]:
    """Whole-device memory counters, straight from the MAX driver.

    Deliberately NOT torch.cuda's key names ("allocated_bytes.all.current"
    and friends): those describe a caching allocator's per-process
    accounting, and MAX exposes no such thing — ``Device.stats`` reports the
    driver's device-wide free/total plus its graph pools. Inventing the CUDA
    keys from those would mean reporting another process's allocations as
    this one's. Callers that want the CUDA schema should read
    ``torch.cuda.memory_stats`` on a CUDA build; callers that want a number
    for this device get honest ones here.
    """
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        find_equivalent_max_device,
    )

    stats = find_equivalent_max_device(_resolve_sync_device(device)).stats
    return {name: int(value) for name, value in dict(stats).items()}


def memory_summary(
    device: "int | str | torch.device | None" = None, abbreviated: bool = False
) -> str:
    """Human-readable ``memory_stats``. FSDP prints this in OOM messages."""
    resolved = _resolve_sync_device(device)
    lines = [f"Mojo device memory ({resolved}), as reported by the MAX driver:"]
    for name, value in memory_stats(device).items():
        lines.append(f"  {name:24s} {value:>18,d}")
    return "\n".join(lines)


# Streams: real torch.Stream/torch.Event support, dispatched here by
# register_mojo_devices() because the generic classes' C++ guard is a stub
# for Python PrivateUse1 backends. See mojo_device/streams.py.
from torch_mojo_backend.mojo_device.streams import (  # noqa: E402
    Event as Event,
    Stream as Stream,
    current_stream as current_stream,
    default_stream as default_stream,
    set_stream as set_stream,
    stream as stream,
)
