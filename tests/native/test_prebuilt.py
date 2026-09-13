"""Selection of the prebuilt libraries the wheel ships (native/prebuilt/).

The copy-into-the-cache and dlopen halves are covered end to end by
`scripts/smoke_prebuilt_wheel.py` (run against a built wheel in
.github/workflows/wheel.yml); what is worth testing cheaply is the rule that
decides whether a shipped file may be used at all, since using the wrong one
means a mismatched ABI at load time. These tests need no GPU and build
nothing: they point the lookup at a manifest in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_mojo_backend import native


def _manifest(tmp_path: Path, entry: dict[str, object]) -> Path:
    """A prebuilt directory holding one entry and its (empty) library."""
    (tmp_path / str(entry["file"])).write_bytes(b"")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"manifest_version": 1, "entries": [entry]})
    )
    return tmp_path


@pytest.fixture
def prebuilt_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(native, "_PREBUILT", tmp_path)
    monkeypatch.setattr(native, "_PREBUILT_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.delenv("TORCH_MOJO_BACKEND_PREBUILT", raising=False)
    return tmp_path


def _shim_entry() -> dict[str, object]:
    spec = native.prebuilt_shim_spec()
    return {**spec, "file": native.prebuilt_file_name(spec)}


def _backend_entry() -> dict[str, object]:
    spec = native.prebuilt_backend_spec()
    return {**spec, "file": native.prebuilt_file_name(spec)}


def test_no_manifest_means_no_prebuilt(prebuilt_dir: Path):
    assert native._prebuilt_match(native.prebuilt_shim_spec()) is None


@pytest.mark.parametrize("entry_of", [_shim_entry, _backend_entry])
def test_an_entry_for_this_environment_is_used(prebuilt_dir: Path, entry_of):
    entry = entry_of()
    _manifest(prebuilt_dir, entry)
    spec = (
        native.prebuilt_shim_spec()
        if entry["kind"] == "shim"
        else native.prebuilt_backend_spec()
    )
    assert native._prebuilt_match(spec) == prebuilt_dir / str(entry["file"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch", "1.99"),  # another torch series
        ("machine", "s390x"),
        ("platform", "aix"),
        ("cxx11abi", 0),  # the other libstdc++ ABI
        ("source_hash", "0" * 16),  # csrc/ has moved on since this was built
        ("kind", "backend"),
    ],
)
def test_an_entry_that_does_not_match_is_ignored(
    prebuilt_dir: Path, field: str, value: object
):
    _manifest(prebuilt_dir, {**_shim_entry(), field: value})
    assert native._prebuilt_match(native.prebuilt_shim_spec()) is None


def test_a_missing_library_file_is_ignored(prebuilt_dir: Path):
    entry = _shim_entry()
    _manifest(prebuilt_dir, entry)
    (prebuilt_dir / str(entry["file"])).unlink()
    assert native._prebuilt_match(native.prebuilt_shim_spec()) is None


def test_env_var_turns_the_lookup_off(prebuilt_dir: Path, monkeypatch):
    _manifest(prebuilt_dir, _shim_entry())
    monkeypatch.setenv("TORCH_MOJO_BACKEND_PREBUILT", "0")
    assert native._prebuilt_match(native.prebuilt_shim_spec()) is None


def test_file_names_carry_what_the_match_is_keyed_by():
    shim = native.prebuilt_file_name(native.prebuilt_shim_spec())
    backend = native.prebuilt_file_name(native.prebuilt_backend_spec())
    assert shim.startswith("libtmb_shim-torch") and "-cxx11abi" in shim
    assert backend.startswith("libtmb_backend-max")
    assert shim.endswith((".so", ".dylib")) and backend.endswith((".so", ".dylib"))


def test_source_hashes_are_stable_and_distinct():
    """They key the manifest entries, so they must depend on the sources
    alone -- not on the toolchain, which differs between the machine that
    built the wheel and the machine that installs it."""
    assert native.shim_source_hash() == native.shim_source_hash()
    assert native.backend_source_hash() == native.backend_source_hash()
    assert native.shim_source_hash() != native.backend_source_hash()
