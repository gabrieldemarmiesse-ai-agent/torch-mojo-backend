"""``_alloc_with_recovery``: a device-OOM allocation synchronizes the device
(so stream-ordered frees land) and retries exactly once. Host-only: the
holder and the device are stand-ins.

Workaround for https://github.com/modular/modular/issues/6801 (MAX's device
allocator OOMs instead of reclaiming freed-but-pending memory); delete this
file together with ``_alloc_with_recovery`` once that issue is fixed."""

from types import SimpleNamespace

import pytest

from torch_mojo_backend.mojo_device import torch_mojo_tensor


class _FlakyHolder:
    def __init__(self, failures: list[BaseException]):
        self.failures = failures
        self.calls = 0

    def alloc(self, ctx_ptr: int, nbytes: int) -> tuple[object, int]:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return (object(), 0x1000)


def _oom() -> Exception:
    return Exception("CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)")


@pytest.fixture
def synced(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    monkeypatch.setattr(torch_mojo_tensor, "_ctx_ptr", lambda _device: 7)
    return []


def _device(synced: list[bool]) -> SimpleNamespace:
    return SimpleNamespace(
        default_stream=SimpleNamespace(synchronize=lambda: synced.append(True))
    )


def _install(monkeypatch: pytest.MonkeyPatch, holder: _FlakyHolder):
    monkeypatch.setattr(torch_mojo_tensor, "_holder_mod", lambda: holder)


def test_oom_synchronizes_and_retries_once(monkeypatch, synced):
    holder = _FlakyHolder([_oom()])
    _install(monkeypatch, holder)
    result = torch_mojo_tensor._alloc_with_recovery(_device(synced), 4096)
    assert result[1] == 0x1000
    assert holder.calls == 2
    assert synced == [True]


def test_non_oom_error_propagates_immediately(monkeypatch, synced):
    holder = _FlakyHolder([ValueError("not a memory problem")])
    _install(monkeypatch, holder)
    with pytest.raises(ValueError, match="not a memory problem"):
        torch_mojo_tensor._alloc_with_recovery(_device(synced), 4096)
    assert holder.calls == 1
    assert synced == []


def test_second_oom_surfaces(monkeypatch, synced):
    holder = _FlakyHolder([_oom(), _oom()])
    _install(monkeypatch, holder)
    with pytest.raises(Exception, match="OUT_OF_MEMORY"):
        torch_mojo_tensor._alloc_with_recovery(_device(synced), 4096)
    assert holder.calls == 2
    assert synced == [True]
