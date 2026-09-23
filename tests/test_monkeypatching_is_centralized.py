"""Every runtime patch of a torch module or class must live in
``torch_mojo_backend/monkeypatching.py`` (see its module docstring).

The check is static: an assignment whose target is an attribute or item of
a name bound by ``import torch...`` / ``from torch... import ...`` counts as
a monkeypatch. Container mutation through method calls (``.append``) is not
caught; keep those in the same file by convention.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "torch_mojo_backend"
MONKEYPATCHING = PACKAGE / "monkeypatching.py"


def _torch_bound_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    names.add(alias.asname or alias.name.partition(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "torch" or node.module.startswith("torch."):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
    return names


def _root_name(target: ast.expr) -> str | None:
    while isinstance(target, ast.Attribute | ast.Subscript):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _torch_attribute_assignments(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    torch_names = _torch_bound_names(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Attribute | ast.Subscript):
                continue
            if _root_name(target) in torch_names:
                found.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno}")
    return found


def test_the_scanner_recognizes_the_centralized_patches():
    assert _torch_attribute_assignments(MONKEYPATCHING)


def test_no_torch_monkeypatch_outside_monkeypatching_py():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == MONKEYPATCHING:
            continue
        offenders.extend(_torch_attribute_assignments(path))
    assert offenders == [], (
        "torch attribute assignments outside torch_mojo_backend/monkeypatching.py: "
        + ", ".join(offenders)
    )
