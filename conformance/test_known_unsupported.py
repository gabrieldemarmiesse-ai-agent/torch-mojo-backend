"""The declaration mechanism itself: fast, no operator runs here.

`known_unsupported.py` replaces "the exception said it was not implemented"
with "someone declared this case cannot pass".  That trade is only safe while
two properties hold, and both are the kind that break silently:

* a declared case that STARTS PASSING must fail the suite.  Without that the
  file is a mute list nobody is ever told to shorten -- strictly worse than
  the exception-sniffing it replaced, which at least stopped skipping the
  moment the operator started working;
* an entry must address a node that exists.  A renamed operator or a dtype
  upstream dropped leaves an entry that can never fail again, which is the
  same mute by another route (`benchmarks/test_coverage.py` reconciles its
  own recorded keys against its test nodes for exactly this reason).
"""

from __future__ import annotations

import collections
import unittest

import known_unsupported
import pytest
import test_opinfo
import torch
from torch.testing._internal.common_methods_invocations import op_db


def test_a_declared_case_that_passes_fails_the_suite() -> None:
    """The invariant that keeps the list from rotting into a permanent mute."""
    with pytest.raises(AssertionError) as failure:
        test_opinfo._run_declared_unsupported(
            "declared unsupported: entry X", lambda: None
        )
    message = str(failure.value)
    assert "entry X" in message
    assert "Delete that entry" in message


def test_a_declared_case_that_still_fails_is_an_expected_failure() -> None:
    def run() -> None:
        raise RuntimeError("no fast implementation for aten::something")

    with pytest.raises(pytest.xfail.Exception) as outcome:
        test_opinfo._run_declared_unsupported("declared unsupported: entry Y", run)
    assert "entry Y" in str(outcome.value)


def test_a_declared_case_that_skips_still_skips() -> None:
    """A case the harness declined to compare never ran, so it is evidence
    neither for nor against the declaration."""

    def run() -> None:
        raise unittest.SkipTest("OpInfo produced no sample inputs for this dtype")

    with pytest.raises(unittest.SkipTest):
        test_opinfo._run_declared_unsupported("declared unsupported: entry Z", run)


def test_an_accelerator_delta_replaces_the_base_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One lookup rule: an operator listed for an accelerator replaces its base
    line there, and the empty tuple means "supported here"."""
    monkeypatch.setitem(
        known_unsupported._BASE["test_matches_cpu"], "abs", ("float32", "float16")
    )
    monkeypatch.setitem(
        known_unsupported._ACCELERATOR_DELTAS,
        "gpu-with-half-support",
        {"test_matches_cpu": {"abs": ("float32",)}},
    )
    monkeypatch.setitem(
        known_unsupported._ACCELERATOR_DELTAS,
        "gpu-that-does-it-all",
        {"test_matches_cpu": {"abs": ()}},
    )

    def declared(dtype: torch.dtype, accelerator: str) -> str | None:
        return known_unsupported.declared_unsupported(
            "test_matches_cpu", "abs", dtype, accelerator
        )

    assert declared(torch.float32, "gpu-with-no-delta") is not None
    assert declared(torch.float16, "gpu-with-no-delta") is not None
    assert declared(torch.float32, "gpu-with-half-support") is not None
    assert declared(torch.float16, "gpu-with-half-support") is None
    assert declared(torch.float32, "gpu-that-does-it-all") is None
    assert declared(torch.float16, "gpu-that-does-it-all") is None


def test_operator_tokens_identify_one_operator() -> None:
    """The table's key is `OpInfo.formatted_name`, so two OpInfos sharing one
    would make an entry ambiguous."""
    counts = collections.Counter(op.formatted_name for op in op_db)
    assert [name for name, count in counts.items() if count > 1] == []


def _instantiated_nodes() -> dict[str, dict[str, frozenset[str]]]:
    """test -> operator token -> dtype tokens the suite instantiates a node for.

    The same intersection `@ops(..., allowed_dtypes=...)` computes:
    `op.supported_dtypes(device_cls.device_type)` narrowed to the suite's dtype
    list, with the device type the decorator sees at parametrize time (the
    class attribute is still "privateuse1" then; `setUpClass` rewrites it to
    the backend name later, and `supported_dtypes` resolves both the same way).
    """
    nodes: dict[str, dict[str, frozenset[str]]] = {}
    for test, ops, allowed in (
        ("test_matches_cpu", op_db, test_opinfo._DTYPES),
        (
            "test_errors_match",
            [op for op in op_db if op.error_inputs_func is not None],
            (torch.float32,),
        ),
    ):
        nodes[test] = {
            op.formatted_name: frozenset(
                known_unsupported.dtype_token(dtype)
                for dtype in set(op.supported_dtypes("privateuse1")) & set(allowed)
            )
            for op in ops
        }
    return nodes


@pytest.mark.parametrize("test", sorted(known_unsupported.TABLE_NAMES))
def test_every_declared_entry_addresses_a_node_that_exists(test: str) -> None:
    """A stale entry is a mute: nothing runs, so nothing can ever fail and
    tell its author to delete it."""
    instantiated = _instantiated_nodes()[test]
    stale = sorted(
        f"{known_unsupported.TABLE_NAMES[test]}[{op!r}] lists {dtype!r}"
        for accelerator in {
            known_unsupported.BASE_ACCELERATOR,
            *known_unsupported._ACCELERATOR_DELTAS,
        }
        for op, dtypes in known_unsupported.declared(test, accelerator).items()
        for dtype in dtypes
        if dtype not in instantiated.get(op, frozenset())
    )
    assert not stale, "declared cases the suite does not instantiate: " + "; ".join(
        stale
    )
