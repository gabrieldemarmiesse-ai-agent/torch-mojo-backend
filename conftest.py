"""Repo-root pytest hooks.

CI shards the whole suite 40 ways with `pytest-split` (`--splits 40 --group N`).
pytest-split has no durations file, so it falls back to cutting the *collection
order* into 40 equal-count chunks -- and collection order is the worst possible
order to cut, because cost is clustered in it: a file's tests, and above all the
parametrizations of one test function, sit next to each other and cost the same.
A chunk is therefore a slice of one or two files rather than a sample of the
suite, and the shards came out between 41 s and 1416 s.

So we deal the tests out pseudo-randomly before pytest-split chunks them. The
seed is fixed, so every shard of a run agrees on the deal and every test lands
in exactly one shard; each chunk becomes a uniform sample of the whole suite and
the shards converge on the mean. This needs no durations file, which matters
here because durations would have to be *measured* on CI to be right: this suite
skips a different set of tests on a GPU box than on the CPU-only runners, so
timings recorded anywhere else would balance for the wrong machine.

The deal decides shard membership and nothing else. `pytest_collection_finish`
puts the surviving items back in collection order before anything runs, so
execution order within a shard is what it always was -- same neighbours, same
module grouping, same module-scoped fixture setups.
"""

from __future__ import annotations

import random

import pytest

# Any fixed value works; this one is the date the deal was introduced. Changing
# it reshuffles every shard, which is harmless but throws away the per-shard
# native build caches CI keys on the shard index.
SHARD_DEAL_SEED = 20260915

_COLLECTION_INDEX = pytest.StashKey[int]()


def _sharding(config: pytest.Config) -> bool:
    """Is this run one shard of a pytest-split run?"""
    return getattr(config.option, "splits", None) is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Deal the tests out before pytest-split cuts them into chunks.

    pytest-split's own hook is `trylast`, so this one runs first and it chunks
    the dealt order. Un-sharded runs are left alone entirely.
    """
    if not _sharding(config):
        return
    for index, item in enumerate(items):
        item.stash[_COLLECTION_INDEX] = index
    random.Random(SHARD_DEAL_SEED).shuffle(items)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_finish(session: pytest.Session):
    """Restore collection order for the items this shard kept.

    Runs after every `pytest_collection_modifyitems`, on the very list pytest is
    about to execute, so the deal above never reaches execution order.
    `tryfirst` only puts this ahead of the terminal reporter's own
    `pytest_collection_finish`, so `--collect-only` prints a shard in the order
    it will actually run rather than in dealt order.
    """
    if not _sharding(session.config):
        return
    session.items.sort(key=lambda item: item.stash[_COLLECTION_INDEX])
