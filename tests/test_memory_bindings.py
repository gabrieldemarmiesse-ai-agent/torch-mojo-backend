"""Missing torch memory bindings report the installed version and requirement."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torch_mojo_backend.native import device_module


@pytest.mark.parametrize("version", ["2.8.0+cpu", "2.11.0+custom"])
@pytest.mark.parametrize(
    "api,binding,minimum",
    [
        (api, "_accelerator_getDeviceStats", "2.9")
        for api in (
            "memory_stats_as_nested_dict",
            "memory_stats",
            "memory_allocated",
            "max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
            "memory_summary",
        )
    ]
    + [
        ("reset_peak_memory_stats", "_accelerator_resetPeakStats", "2.9"),
        ("reset_accumulated_memory_stats", "_accelerator_resetAccumulatedStats", "2.9"),
        ("empty_cache", "_accelerator_emptyCache", "2.9"),
        ("mem_get_info", "_accelerator_getMemoryInfo", "2.10"),
    ],
)
def test_missing_memory_binding(
    monkeypatch: pytest.MonkeyPatch, version: str, api: str, binding: str, minimum: str
):
    # Replace this module's torch reference, without patching torch itself.
    monkeypatch.setattr(
        device_module,
        "torch",
        SimpleNamespace(__version__=version, _C=SimpleNamespace()),
    )
    monkeypatch.setattr(device_module, "_index", Mock(return_value=0))
    with pytest.raises(RuntimeError) as exc:
        getattr(device_module, api)()
    message = str(exc.value)
    assert f"torch {version}" in message
    assert f"torch._C.{binding}" in message
    assert f"torch>={minimum}" in message
    assert "torch-mojo-backend requires torch>=2.10" in message
