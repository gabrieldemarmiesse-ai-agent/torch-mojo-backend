#!/usr/bin/env python
"""Regenerate the tables in `known_unsupported.py` from a real conformance run.

    # what this machine's accelerator can and cannot do (~40 min; -n shortens it)
    uv run python conformance/regenerate_known_unsupported.py -n 15

    # ... and write it into known_unsupported.py
    uv run python conformance/regenerate_known_unsupported.py --write -n 15

    # a subset: unrecognised arguments are passed through to pytest, so a PR
    # landing one operator regenerates that operator's lines and nothing else
    uv run python conformance/regenerate_known_unsupported.py --write -k bmm

Updates merge per node, like `benchmarks/--update-baselines`: a node the run
did not execute keeps whatever the table says about it, so a one-operator run
rewrites one operator's lines.  To re-derive a whole table from scratch instead,
empty its generated block first and run without `-k`.

This is the ONLY place in the suite that looks at WHAT an exception was.
`_looks_absent` recognises the shapes an unimplemented operator raises and
proposes an entry per node that failed that way.  Its output is a table a human
reads, reviews and commits -- not a runtime decision -- which is the whole
difference from the `_not_implemented(exc)` check this replaced.  A
misclassification here hides nothing either: over-declare and the suite fails
the node as an unexpected pass, under-declare and it fails it as a wrong
answer.  Both point back at this file.

How it works: the module is both this driver and a pytest plugin.  The driver
runs `conformance/test_opinfo.py` with itself loaded as `-p
regenerate_known_unsupported`; the plugin half writes one JSON line per node
outcome, and one per failing node naming the exception it raised.  Records are
per worker process, so this survives xdist and a segfaulting node
(`FINDINGS.md` #1): everything recorded before a crash still counts, and the
driver reports what it could not place instead of quietly dropping it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _pytest.runner import CallInfo

_RECORD_DIR_ENV = "CONFORMANCE_UNSUPPORTED_RECORD_DIR"
_HERE = Path(__file__).resolve().parent
_TABLE_FILE = _HERE / "known_unsupported.py"


# ---------------------------------------------------------------------------
# pytest-plugin half: record what every node did, and what the failures raised.
# ---------------------------------------------------------------------------


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """`exc` and the exceptions it was raised from.

    Required, not defensive: OpInfo's own test wrapper catches every failure
    that is not a skip and re-raises it as a plain `Exception` carrying a repro
    line (`common_device_type.py`, "Caused by sample input at index"), so the
    exception whose TYPE says what went wrong -- the NotImplementedError our
    dispatch layer and PyTorch's own raise -- is only reachable through
    `__cause__`.  This is also the clearest argument for a declared list: by
    the time a conformance failure is reported, its type is gone.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 10:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _looks_absent(exc: BaseException) -> bool:
    """Whether `exc` has the shape of "we never implemented this".

    A heuristic, knowingly: a `NotImplementedError` anywhere in the chain --
    ours ("not supported by mojo eager mode"), or PyTorch's own dispatcher
    declining an operator we never registered ("Could not run 'aten::x' with
    arguments from the 'mojo' backend") -- plus a dtype the device does not
    have at all, which is how an `error_inputs_func` corpus that builds a
    complex or float64 tensor purely to provoke the error under test fails
    before the operator is reached.  It runs in this generator and nowhere
    else.
    """
    chain = _exception_chain(exc)
    if any(isinstance(part, NotImplementedError) for part in chain):
        return True
    text = "\n".join(str(part) for part in chain).lower()
    return "no fast implementation" in text or "unsupported torch dtype" in text


def _record(event: dict[str, Any]):
    """Append one event, if recording is on.

    One file per process: xdist workers never share one, and a segfaulting node
    cannot lose what its siblings already wrote.
    """
    directory = os.environ.get(_RECORD_DIR_ENV)
    if not directory:
        return
    with open(Path(directory) / f"{os.getpid()}.jsonl", "a") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()


def pytest_exception_interact(
    node: pytest.Item | pytest.Collector,
    call: CallInfo[Any],
    report: pytest.CollectReport | pytest.TestReport,
):
    """Record what a failing node raised."""
    if not report.failed or call.excinfo is None:
        return
    exc = call.excinfo.value
    _record(
        {
            "event": "exception",
            "node": node.name,
            "absent": _looks_absent(exc),
            # The whole chain, so a table entry can be argued about afterwards
            # without rerunning the suite.
            "raised": [
                f"{type(part).__name__}: {str(part).splitlines()[0][:160]}"
                for part in _exception_chain(exc)
            ],
        }
    )


def pytest_runtest_logreport(report: pytest.TestReport):
    """Record every node's outcome, so a node that ran can be told from one that
    did not: only the first is evidence about the table."""
    if report.when != "call":
        return
    outcome = report.outcome
    reason = getattr(report, "wasxfail", None)
    if outcome == "skipped":
        outcome = "xfailed" if reason is not None else "skipped"
        if reason is None and isinstance(report.longrepr, tuple):
            reason = report.longrepr[2]
    _record(
        {
            "event": "outcome",
            "node": report.nodeid.split("::")[-1],
            "outcome": outcome,
            "reason": reason,
        }
    )


# ---------------------------------------------------------------------------
# driver half
# ---------------------------------------------------------------------------


def _import_suite() -> ModuleType:
    """The conformance test module, imported the way its conftest does."""
    os.environ.setdefault("PYTORCH_TESTING_DEVICE_FOR_CUSTOM", "privateuse1")
    sys.path.insert(0, str(_HERE))
    import torch_mojo_backend

    torch_mojo_backend.register_mojo_devices()
    import test_opinfo

    return test_opinfo


def _device_token() -> str:
    """The device token of a generated node name.

    `common_device_type` names privateuse1 nodes after the registered backend,
    not after "privateuse1" -- "mojo" for us -- so it is read from torch rather
    than spelled out here.
    """
    import torch

    return str(torch._C._get_privateuse1_backend_name())


def _node_index(suite: ModuleType) -> dict[str, tuple[str, str, str]]:
    """Generated node name -> (test name, operator token, dtype token).

    Built from the same op_db and dtypes the suite parametrizes over, so
    nothing has to be parsed back out of a node name whose operator token
    itself contains underscores.
    """
    import known_unsupported
    import torch
    from torch.testing._internal.common_methods_invocations import op_db

    device = _device_token()
    index: dict[str, tuple[str, str, str]] = {}
    for test, dtypes in (
        ("test_matches_cpu", suite._DTYPES),
        ("test_errors_match", (torch.float32,)),
    ):
        for op in op_db:
            for dtype in dtypes:
                token = known_unsupported.dtype_token(dtype)
                index[f"{test}_{op.formatted_name}_{device}_{token}"] = (
                    test,
                    op.formatted_name,
                    token,
                )
    return index


def _run_suite(record_dir: Path, pytest_args: list[str]) -> int:
    """Run the conformance suite with the recording plugin loaded."""
    env = dict(os.environ)
    env[_RECORD_DIR_ENV] = str(record_dir)
    # The plugin is this module; -p needs it importable in every xdist worker.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_HERE), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(_HERE / "test_opinfo.py"),
        "-p",
        Path(__file__).stem,
        "-q",
        *pytest_args,
    ]
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, env=env, cwd=_HERE.parent).returncode


def _read_records(record_dir: Path) -> dict[str, dict[str, Any]]:
    """Node name -> what it did, merged over every recorded event.

    The same node can be recorded twice (an xdist worker records it, the
    controller records the report it receives); the events agree, so the merge
    is by node name.
    """
    nodes: dict[str, dict[str, Any]] = {}
    for path in sorted(record_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            nodes.setdefault(event["node"], {}).update(
                {k: v for k, v in event.items() if k not in ("node", "event")}
            )
    return nodes


def _merged_tables(
    nodes: dict[str, dict[str, Any]],
    index: dict[str, tuple[str, str, str]],
    declared: dict[str, dict[str, frozenset[str]]],
) -> tuple[dict[str, dict[str, set[str]]], list[str]]:
    """What the tables should say after this run, plus the nodes not placeable.

    Starts from what is declared today and applies only what this run
    observed, so nodes it did not execute are left alone.
    """
    tables = {
        test: {op: set(dtypes) for op, dtypes in table.items()}
        for test, table in declared.items()
    }
    unresolved: list[str] = []
    for name, node in sorted(nodes.items()):
        resolved = index.get(name)
        if resolved is None:
            unresolved.append(f"{name} ({node.get('outcome', 'no outcome recorded')})")
            continue
        test, op_name, dtype = resolved
        entry = tables[test].setdefault(op_name, set())
        outcome = node.get("outcome")
        if outcome == "passed":
            entry.discard(dtype)  # works now: the declaration must go
        elif outcome == "xfailed":
            entry.add(dtype)  # declared, and still failing
        elif outcome == "failed":
            # A failure is declared only if it is an absence; a wrong answer is
            # a finding and has to stay a failure.
            if node.get("absent"):
                entry.add(dtype)
            else:
                entry.discard(dtype)
        # "skipped": the case never ran, so it says nothing either way.
    return tables, unresolved


def _render(name: str, table: dict[str, set[str]], dtype_order: list[str]) -> str:
    entries = {op: dtypes for op, dtypes in table.items() if dtypes}
    if not entries:
        return f"{name}: dict[str, tuple[str, ...]] = {{}}"
    lines = [f"{name}: dict[str, tuple[str, ...]] = {{"]
    for op_name in sorted(entries):
        dtypes = sorted(entries[op_name], key=dtype_order.index)
        rendered = ", ".join(f'"{dtype}"' for dtype in dtypes)
        lines.append(f'    "{op_name}": ({rendered},),')
    lines.append("}")
    return "\n".join(lines)


def _splice(text: str, test: str, rendered: str) -> str:
    begin = f"# --- BEGIN GENERATED {test} (regenerate_known_unsupported.py) ---"
    end = f"# --- END GENERATED {test} ---"
    head, _, rest = text.partition(begin)
    _, _, tail = rest.partition(end)
    if not rest or not tail:
        raise SystemExit(f"{_TABLE_FILE}: generated block for {test} not found")
    return f"{head}{begin}\n{rendered}\n{end}{tail}"


def _report_diff(
    test: str, merged: dict[str, set[str]], declared: dict[str, frozenset[str]]
):
    print(f"\n{test}:")
    pairs = {(op, d) for op, dtypes in merged.items() for d in dtypes}
    now = {(op, d) for op, dtypes in declared.items() for d in dtypes}
    for label, changed in (
        ("+ declare (absent, not listed yet)", pairs - now),
        ("- delete (listed, no longer absent)", now - pairs),
    ):
        print(f"  {label}: {len(changed)}")
        for op, dtype in sorted(changed):
            print(f"      {op} {dtype}")
    print(f"  unchanged: {len(pairs & now)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="splice the measured tables into known_unsupported.py",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="keep this run's records here instead of in a temporary directory",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="do not run the suite; rebuild the tables from --records",
    )
    args, pytest_args = parser.parse_known_args()
    if args.no_run and args.records is None:
        raise SystemExit("--no-run needs --records DIR from an earlier run")

    suite = _import_suite()
    import known_unsupported

    accelerator = known_unsupported.accelerator_key()
    if args.write and accelerator != known_unsupported.BASE_ACCELERATOR:
        raise SystemExit(
            f"This machine's accelerator is {accelerator!r} but the base tables "
            f"in {_TABLE_FILE.name} are {known_unsupported.BASE_ACCELERATOR!r}. "
            "Writing would overwrite another accelerator's tables: take the "
            "printed ones and record only the per-operator differences under "
            "_ACCELERATOR_DELTAS[<accelerator>]."
        )

    with tempfile.TemporaryDirectory() as temporary:
        record_dir = args.records or Path(temporary)
        record_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_run:
            _run_suite(record_dir, pytest_args)
        nodes = _read_records(record_dir)

    declared = {
        test: known_unsupported.declared(test, accelerator)
        for test in known_unsupported.TABLE_NAMES
    }
    tables, unresolved = _merged_tables(nodes, _node_index(suite), declared)
    dtype_order = [known_unsupported.dtype_token(dtype) for dtype in suite._DTYPES]

    print(f"\naccelerator: {accelerator}")
    print(f"nodes recorded: {len(nodes)}")
    for test in tables:
        _report_diff(test, tables[test], declared[test])
    if unresolved:
        print(f"\nnode names that match no (op, dtype) ({len(unresolved)}):")
        for name in unresolved:
            print(f"    {name}")

    rendered = {
        test: _render(known_unsupported.TABLE_NAMES[test], table, dtype_order)
        for test, table in tables.items()
    }
    if not args.write:
        print("\n" + "\n\n".join(rendered.values()))
        print("\n(--write splices these into known_unsupported.py)")
        return 0

    text = _TABLE_FILE.read_text()
    for test, block in rendered.items():
        text = _splice(text, test, block)
    _TABLE_FILE.write_text(text)
    print(f"\nwrote {_TABLE_FILE}")
    if subprocess.run(["ruff", "format", str(_TABLE_FILE)], check=False).returncode:
        print("run `uv run ruff format conformance/known_unsupported.py`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
