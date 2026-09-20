"""The import grammar of the Mojo sources: one root, one package, one way.

Every Mojo file lives under `torch_mojo_backend/mojo/`, the one `-I` of every
build, inside the one top-level package `tmb`. An in-repo import is therefore
always `from tmb.<pkg>.<module> import name`: absolute, and naming a module
(a file), not a package. Two things make that a rule rather than a taste:

- the compiler forbids relative imports in the file passed to `mojo build`
  (`cannot import relative to a top-level package`), and every entry.mojo
  is such a file, so one absolute grammar is the only one that works in
  every position;
- the two cache-key walkers (tmb/backend/loader.mojo's `_closure` and
  native.mojo_import_closure) resolve exactly that grammar; an import they
  cannot resolve is a file whose edits would not invalidate a build.

The one exception is `tmb/graph`, the MAX custom-op package: MAX precompiles
that directory on its own with no `-I`, so inside it siblings are imported
relatively (`from .unary_math import ...`) and nothing else may be.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "torch_mojo_backend" / "mojo"
GRAPH = ROOT / "tmb" / "graph"
# Test-side Mojo (probes, multinode self-tests) is built with the same -I.
TEST_MOJO = [
    p for p in (REPO / "tests").rglob("*.mojo") if "dummy_mojo_kernels" not in p.parts
]
IMPORT = re.compile(r"^(?:from\s+(\S+)\s+import\b|import\s+(\S+))")
# Top-level names that are the toolchain's: the stdlib, MAX and its kernel
# packages. Anything else must be ours, spelled `tmb.`.
TOOLCHAIN = {"std", "max", "nn", "linalg", "layout", "extensibility", "internal_utils"}


def _imports(path: Path) -> list[tuple[int, str]]:
    out = []
    in_doc = False
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.count('"""') % 2 == 1:
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        m = IMPORT.match(line)
        if m:
            out.append((n, m.group(1) or m.group(2)))
    return out


def _sources() -> list[Path]:
    paths = sorted(ROOT.rglob("*.mojo")) + sorted(TEST_MOJO)
    assert len(paths) > 100, "the Mojo tree moved; update ROOT"
    return paths


@pytest.mark.parametrize(
    "path", _sources(), ids=lambda p: p.relative_to(REPO).as_posix()
)
def test_imports_are_absolute_tmb_modules(path: Path):
    in_graph = GRAPH in path.parents
    for line, mod in _imports(path):
        where = f"{path.relative_to(REPO)}:{line}: `{mod}`"
        if mod.startswith("."):
            assert in_graph, (
                f"{where} is a relative import; outside tmb/graph spell it "
                "`from tmb.<pkg>.<module> import ...` (the compiler rejects "
                "relative imports in an entry module anyway)"
            )
            target = path.parent / (mod[1:].replace(".", "/") + ".mojo")
            assert target.is_file(), f"{where} names no sibling module"
            continue
        head = mod.split(".")[0]
        if head in TOOLCHAIN:
            continue
        assert head == "tmb", (
            f"{where} is a bare import: with one -I root every in-repo import "
            "is absolute, `from tmb.<pkg>.<module> import ...`"
        )
        assert not in_graph, (
            f"{where}: tmb/graph is precompiled by MAX without -I, so it can "
            "only import its own siblings, relatively"
        )
        target = ROOT / (mod.replace(".", "/") + ".mojo")
        assert target.is_file(), (
            f"{where} names no module file ({target.relative_to(REPO)}); "
            "import from a module, not a package"
        )


def test_every_built_library_is_an_entry_module():
    """What the drivers build: tmb/backend/entry.mojo, tmb/ccl/entry.mojo and
    one entry.mojo per kernel family, nothing named after its directory."""
    assert (ROOT / "tmb/backend/entry.mojo").is_file()
    assert (ROOT / "tmb/ccl/entry.mojo").is_file()
    families = sorted(d for d in (ROOT / "tmb/kernels").iterdir() if d.is_dir())
    with_entry = [d.name for d in families if (d / "entry.mojo").is_file()]
    assert "common" not in with_entry, "tmb/kernels/common is shared code, not a family"
    assert len(with_entry) >= 25, with_entry
    for d in families:
        for f in d.glob("*.mojo"):
            assert f.stem != d.name, (
                f"{f.relative_to(REPO)}: a module named after its directory is "
                "shadowed by the package; the entry is entry.mojo"
            )
    for name in with_entry:
        text = (ROOT / "tmb/kernels" / name / "entry.mojo").read_text()
        assert "def tmb_call" in text, (
            f"tmb/kernels/{name}/entry.mojo exports no tmb_call"
        )
    for f in (ROOT / "tmb/kernels").rglob("*.mojo"):
        if f.name != "entry.mojo":
            assert "def tmb_call" not in f.read_text(), (
                f"{f.relative_to(REPO)} exports tmb_call but is not an entry.mojo"
            )


def test_every_family_string_in_the_ops_names_a_family_directory():
    """`KernelCall("logic", ...)` and the family strings the ops hand to their
    helpers must name a directory under tmb/kernels: a family renamed on
    disk but not in an op would only fail at that op's first call."""
    kernels = ROOT / "tmb" / "kernels"
    families = {d.name for d in kernels.iterdir() if d.is_dir() and d.name != "common"}
    literal = re.compile(r'"([a-z][a-z0-9_]*)"')
    used = set()
    for path in sorted((ROOT / "tmb" / "ops").glob("*.mojo")):
        for line, text in enumerate(path.read_text().splitlines(), 1):
            for name in literal.findall(text):
                if name in families:
                    used.add(name)
                assert not (name.endswith("_ops") and name[:-4] in families), (
                    f"{path.relative_to(REPO)}:{line}: {name!r} is the old name of "
                    f"family {name[:-4]!r}"
                )
    # activation_forward and embedding_backward are only imported by other
    # families' entries; sdpa_backward has had no caller since the FA4
    # backward landed (it was unreferenced before the one-root move too).
    unreferenced = (
        families - used - {"activation_forward", "embedding_backward", "sdpa_backward"}
    )
    assert not unreferenced, (
        f"kernel families no op names: {sorted(unreferenced)} -- an op's family "
        "string was not updated, or the family is dead"
    )
