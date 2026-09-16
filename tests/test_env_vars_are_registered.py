"""Every environment variable this project reads must be listed in
``torch_mojo_backend/env_vars.py`` (see its module docstring).

That list is what ``register_mojo_devices()`` holds the user's environment up
against, so a variable missing from it costs the user the typo check on it:
they export a name with a letter wrong, nothing reads it, nothing complains,
and the default it was meant to override quietly stays in force. Hence a test
rather than a convention.

Two checks, over every tracked ``.py`` / ``.mojo`` / ``.c`` / ``.cpp`` /
``.h`` file in the repository:

* any token spelled like one of ours -- ``TORCH_MOJO_BACKEND_*``,
  ``PYTORCH_MOJO_BACKEND_*``, ``MOJOCCL_*`` -- anywhere in a source file,
  literal or comment, is a name in our namespace and must be registered. This
  is the check that reaches the Mojo side: the two ``env_vars.mojo`` files
  build separately from Python and cannot import the table, so this is what
  keeps all three in step.
* any literal name handed to ``os.environ`` / ``getenv`` inside the shipped
  package must be registered too, as one of ours or as somebody else's
  (``ROCM_PATH``, ``CXX``, ...) that we happen to read.

Shell scripts and CI workflows are not scanned: they set variables for our
processes rather than read any, and their own locals share the prefix.
"""

import re
import subprocess
from pathlib import Path

import pytest

from torch_mojo_backend import env_vars

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "torch_mojo_backend"
# The table itself names every variable by construction, so it can neither
# offend the checks below nor stand in for a real use in the one above.
REGISTRY = PACKAGE / "env_vars.py"
# This file spells deliberate misspellings.
THIS_FILE = Path(__file__).resolve()
SCANNED_SUFFIXES = (".py", ".mojo", ".c", ".cpp", ".h")

# Longest prefix first: `TORCH_MOJO_BACKEND` would otherwise swallow the tail
# of a `PYTORCH_` name. The trailing `[A-Z0-9]` keeps a bare prefix constant
# (`"MOJOCCL_"`) from reading as a variable name.
OURS = re.compile(
    r"\b(?:PYTORCH_MOJO_BACKEND|TORCH_MOJO_BACKEND|MOJOCCL)_[A-Z0-9_]*[A-Z0-9]\b"
)

# A literal name handed to something that reads the environment. Names reached
# through a constant are caught by OURS instead (ours), or are unreachable
# statically (everyone else's) -- which is why the package keeps foreign names
# as literals at the call site.
READS = tuple(
    re.compile(pattern)
    for pattern in (
        r"""os\.environ\.(?:get|pop|setdefault)\(\s*["'](\w+)["']""",
        r"""os\.environ\[\s*["'](\w+)["']\s*\]""",
        r"""os\.getenv\(\s*["'](\w+)["']""",
        r"""["'](\w+)["']\s+(?:not\s+)?in\s+os\.environ""",
        r"""(?:std::)?getenv\(\s*"(\w+)\"""",
    )
)


def _tracked_sources(root: Path) -> list[Path]:
    """Every scannable file git knows about under `root`, so neither the venv
    nor an untracked scratch clone joins in."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = [REPO / name for name in out.split("\0") if name]
    return sorted(
        p
        for p in paths
        if p.suffix in SCANNED_SUFFIXES and p.resolve() != THIS_FILE and p.is_file()
    )


def _names_in(path: Path, patterns) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if isinstance(patterns, re.Pattern):
        return set(patterns.findall(text))
    return {name for pattern in patterns for name in pattern.findall(text)}


def _fix_it(offenders: dict[str, list[str]], kind: str) -> str:
    lines = [
        f"{name} is {kind} but is not registered (used in "
        + ", ".join(sorted(set(where)))
        + ")"
        for name, where in sorted(offenders.items())
    ]
    return (
        "\n".join(lines)
        + "\n\nAdd each one to OWN_ENV_VARS (ours) or FOREIGN_ENV_VARS "
        "(torch's, MAX's, the vendor runtime's, the OS's) in "
        "torch_mojo_backend/env_vars.py, with a line saying what setting it "
        "does. That table is what register_mojo_devices() checks the user's "
        "environment against, so a name missing from it is a knob the user "
        "gets no typo warning for."
    )


def test_the_scanner_recognizes_the_registered_names():
    """A broken regex must fail loudly rather than pass everything."""
    assert "TORCH_MOJO_BACKEND_VERBOSE" in _names_in(PACKAGE / "flags.py", OURS)
    assert "MOJOCCL_REGION_MB" in _names_in(
        PACKAGE / "distributed" / "mojoccl" / "env_vars.mojo", OURS
    )
    assert "TORCH_MOJO_BACKEND_TESTING" in _names_in(
        PACKAGE / "is_running_tests.py", READS
    )
    assert "ROCM_PATH" in _names_in(PACKAGE / "mojo_device" / "hip_peer.py", READS)
    assert len(_tracked_sources(REPO)) > 100


def test_every_name_in_our_namespace_is_registered():
    offenders: dict[str, list[str]] = {}
    for path in _tracked_sources(REPO):
        for name in _names_in(path, OURS) - set(env_vars.OWN_ENV_VARS):
            offenders.setdefault(name, []).append(str(path.relative_to(REPO)))
    assert not offenders, _fix_it(offenders, "spelled like one of ours")


def test_every_environment_read_in_the_package_is_registered():
    known = env_vars.known_env_vars()
    offenders: dict[str, list[str]] = {}
    for path in _tracked_sources(PACKAGE):
        if path == REGISTRY:
            continue
        for name in _names_in(path, READS) - known:
            offenders.setdefault(name, []).append(str(path.relative_to(REPO)))
    assert not offenders, _fix_it(offenders, "read from the environment")


def test_no_registered_name_has_gone_stale():
    """The reverse: a name nothing mentions any more is a knob the docs
    promise and the code dropped."""
    mentioned: set[str] = set()
    for path in _tracked_sources(REPO):
        if path == REGISTRY:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        mentioned.update(re.findall(r"\b\w+\b", text))
    stale = sorted(env_vars.known_env_vars() - mentioned)
    assert not stale, (
        "registered in torch_mojo_backend/env_vars.py but read nowhere: "
        + ", ".join(stale)
        + ". Drop the entry, or spell the name where it is read."
    )


def test_a_misspelling_is_reported_with_a_suggestion():
    bad = "TORCH_MOJO_BACKEND_VERBOZE"
    assert env_vars.unknown_env_vars({bad: "1"}) == [
        (bad, "TORCH_MOJO_BACKEND_VERBOSE")
    ]
    with pytest.warns(env_vars.UnknownEnvVarWarning, match="Did you mean"):
        env_vars.warn_about_unknown_env_vars({bad: "1"})


def test_a_name_resembling_nothing_is_reported_without_one():
    unknown = env_vars.unknown_env_vars({"MOJOCCL_QQQQQQQQ": "1"})
    assert unknown == [("MOJOCCL_QQQQQQQQ", None)]
    with pytest.warns(env_vars.UnknownEnvVarWarning, match="no such environment"):
        env_vars.warn_about_unknown_env_vars({"MOJOCCL_QQQQQQQQ": "1"})


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"TORCH_MOJO_BACKEND_VERBOSE": "1", "MOJOCCL_REGION_MB": "64"},
        # Not ours to diagnose: neither prefix, however misspelled.
        {"ROCM_PATH": "/opt/rocm", "PYTORCH_MOJO_BACKEND_VERBOZE": "1", "PATH": "/bin"},
    ],
)
def test_nothing_else_draws_a_warning(environment, recwarn):
    assert env_vars.unknown_env_vars(environment) == []
    env_vars.warn_about_unknown_env_vars(environment)
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_the_real_environment_is_checked_at_registration():
    """The default argument reads `os.environ`, which is the whole point."""
    import os  # noqa: PLC0415 -- monkeypatching os.environ needs the module here

    name = "TORCH_MOJO_BACKEND_NOT_A_REAL_KNOB"
    os.environ[name] = "1"
    try:
        assert name in dict(env_vars.unknown_env_vars())
    finally:
        del os.environ[name]
