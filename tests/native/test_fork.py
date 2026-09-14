"""fork() after the mojo device is up (docs/native_backend.md, "Fork").

The MAX runtime is not fork-safe: its worker threads and device contexts do
not exist in a forked child, and a device call there waits forever on a
thread that is gone (measured: `DeviceContext.synchronize` under `_to_copy`,
a futex never signalled). The shim installs a pthread_atfork child handler
at registration so the child gets CUDA's answer instead: `_is_in_bad_fork()`
is True, `torch.manual_seed` skips the device, and any allocation or op
raises a RuntimeError that names the 'spawn' start method. What DataLoader
relies on keeps working: `torch.accelerator.is_available()` and
`device_count()` are reads of registration state, and forked workers that
only touch CPU tensors run normally.
"""

from __future__ import annotations

import os
import select
import signal
import warnings
from collections.abc import Callable

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_mojo_backend.native import device_module  # what torch.mojo is

_CHILD_TIMEOUT_S = 120


def _in_forked_child(fn: Callable[[], str]) -> str:
    """fn's return value from a forked child, or a failure if the child hangs
    (which is what this fix turns into an error)."""
    r, w = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # multi-threaded fork
        pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = fn()
        except BaseException as e:  # noqa: BLE001 -- reported to the parent, whatever it is
            out = f"EXC {type(e).__name__}: {e}"
        os.write(w, out.encode())
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        ready, _, _ = select.select([r], [], [], _CHILD_TIMEOUT_S)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail(
                f"the forked child hung for {_CHILD_TIMEOUT_S}s; partial output: "
                + b"".join(chunks).decode()
            )
        chunk = os.read(r, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r)
    os.waitpid(pid, 0)
    return b"".join(chunks).decode()


def test_device_use_in_a_forked_child_raises_instead_of_hanging(mojo_device: str):
    count = torch.accelerator.device_count()
    assert not device_module._is_in_bad_fork()

    def child() -> str:
        parts = [
            f"bad_fork={device_module._is_in_bad_fork()}",
            f"available={torch.accelerator.is_available()}",
            f"count={torch.accelerator.device_count()}",
        ]
        torch.manual_seed(1)  # skips the device in a bad fork, must not raise
        try:
            torch.ones(3, device=mojo_device)
            parts.append("op=no error")
        except RuntimeError as e:
            parts.append(f"op=raised spawn={'spawn' in str(e)}")
        return " ".join(parts)

    out = _in_forked_child(child)
    assert out == f"bad_fork=True available=True count={count} op=raised spawn=True"
    assert not device_module._is_in_bad_fork()  # the parent is untouched


@pytest.mark.filterwarnings("ignore:This process:DeprecationWarning")
@pytest.mark.parametrize("pin_memory", [False, True])
def test_dataloader_with_forked_workers_after_registration(pin_memory: bool):
    ds = TensorDataset(torch.arange(64.0).view(16, 4))
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        multiprocessing_context="fork",
        pin_memory=pin_memory,
        timeout=_CHILD_TIMEOUT_S,  # a hung worker fails instead of blocking
    )
    assert sum(batch[0].shape[0] for batch in loader) == 16
