"""Record one accelerator's conformance tables as a delta over the base ones.

`regenerate_known_unsupported.py` only writes the base tables (the
`BASE_ACCELERATOR`'s); on any other accelerator it refuses `--write` and prints
that accelerator's tables instead. This script turns that printed output into
`_ACCELERATOR_DELTAS[<accelerator>]` in `known_unsupported.py`: for every
operator whose dtype set differs from the base table's, the delta holds the
accelerator's set (an empty tuple means "supported here, whatever the base
says"). Other accelerators' deltas are kept.

    uv run python conformance/regenerate_known_unsupported.py --records DIR -n 8 > regen.log
    uv run python conformance/write_accelerator_delta.py regen.log            # dry run
    uv run python conformance/write_accelerator_delta.py regen.log --write
    uv run ruff format conformance/known_unsupported.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TABLE_FILE = _HERE / "known_unsupported.py"
_DTYPE_ORDER = ["float32", "bfloat16", "float16", "int64", "bool", "float64", "int32"]


def _dtype_rank(token: str) -> int:
    return _DTYPE_ORDER.index(token) if token in _DTYPE_ORDER else len(_DTYPE_ORDER)


def _printed_tables(
    log: str, table_names: dict[str, str]
) -> tuple[str, dict[str, dict[str, tuple[str, ...]]]]:
    """The accelerator named in the log's last report and its rendered tables."""
    start = log.rfind("\naccelerator: ")
    if start < 0:
        raise SystemExit("no 'accelerator:' report in the log")
    section = log[start:]
    accelerator = section.split("\n", 2)[1].removeprefix("accelerator: ").strip()
    tables: dict[str, dict[str, tuple[str, ...]]] = {}
    for test, var in table_names.items():
        m = re.search(
            rf"^{re.escape(var)}: dict\[str, tuple\[str, \.\.\.\]\] = (\{{.*?^\}}|\{{\}})",
            section,
            re.S | re.M,
        )
        if m is None:
            raise SystemExit(f"table {var} not found in the log")
        parsed = ast.literal_eval(m.group(1))
        tables[test] = {op: tuple(dtypes) for op, dtypes in parsed.items()}
    return accelerator, tables


def _render(deltas: dict[str, dict[str, dict[str, tuple[str, ...]]]]) -> str:
    lines = [
        "_ACCELERATOR_DELTAS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {"
    ]
    for accelerator in sorted(deltas):
        lines.append(f'    "{accelerator}": {{')
        for test, table in deltas[accelerator].items():
            lines.append(f'        "{test}": {{')
            for op, dtypes in sorted(table.items()):
                rendered = ", ".join(f'"{d}"' for d in dtypes)
                comma = "," if len(dtypes) == 1 else ""
                lines.append(f'            "{op}": ({rendered}{comma}),')
            lines.append("        },")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", type=Path, help="output of regenerate_known_unsupported.py"
    )
    parser.add_argument(
        "--write", action="store_true", help="splice into known_unsupported.py"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(_HERE))
    import known_unsupported as ku  # noqa: PLC0415 -- importable only via the sys.path insert above

    accelerator, measured = _printed_tables(args.log.read_text(), ku.TABLE_NAMES)
    if accelerator == ku.BASE_ACCELERATOR:
        raise SystemExit(
            f"{accelerator} is the base accelerator: regenerate its tables with --write instead"
        )
    delta: dict[str, dict[str, tuple[str, ...]]] = {}
    for test in ku.TABLE_NAMES:
        base = ku._BASE[test]
        here = measured[test]
        changed = {
            op: tuple(sorted(here.get(op, ()), key=_dtype_rank))
            for op in sorted(set(base) | set(here))
            if set(base.get(op, ())) != set(here.get(op, ()))
        }
        delta[test] = changed
        markers = sum(1 for v in changed.values() if not v)
        print(
            f"{test}: {len(changed)} operators differ from the base ({markers} supported-here markers)"
        )

    deltas = {k: v for k, v in ku._ACCELERATOR_DELTAS.items() if k != accelerator}
    deltas[accelerator] = delta
    text = _TABLE_FILE.read_text()
    m = re.search(r"^_ACCELERATOR_DELTAS: .*?^\}\n", text, re.S | re.M)
    if m is None:
        raise SystemExit(f"{_TABLE_FILE}: _ACCELERATOR_DELTAS block not found")
    new = text[: m.start()] + _render(deltas) + "\n" + text[m.end() :]
    if not args.write:
        print(
            f"dry run for {accelerator!r}: --write splices the delta into {_TABLE_FILE.name}"
        )
        return 0
    _TABLE_FILE.write_text(new)
    print(f"wrote {_TABLE_FILE} ({accelerator!r}); run `uv run ruff format` on it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
