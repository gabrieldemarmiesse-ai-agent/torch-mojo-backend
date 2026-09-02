"""Repo-root pytest hooks.

CI shards the whole suite 20 ways (`pytest-split`) and then filters each shard
with `-k "mojo_device"` or `-k "not mojo_device"`. pytest-split has no
durations file yet, so it chunks by collection order rather than actual
runtime, and a shard's slice can legitimately contain zero tests matching the
keyword filter -- the filter didn't do anything wrong, that shard's tests just
all belong to the other side of the split. Left alone, pytest reports that as
`ExitCode.NO_TESTS_COLLECTED` (5), which CI would otherwise have to treat as a
job failure.

This must stay narrow: a `-k` filter deselecting every collected item is
expected and fine, but a bare `pytest some/typo`-path collecting nothing is
still a real mistake and must keep exiting 5.
"""

from __future__ import annotations

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    """CI shards the suite 20 ways then filters with -k; a shard whose chunk
    contains no test matching the filter must count as empty, not failed."""
    if (
        exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
        and session.config.option.keyword
    ):
        session.exitstatus = int(pytest.ExitCode.OK)
