"""Separation of concerns enforced by static import analysis (Principle II, NFR-02)."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "zeroshot"


def imports_of(path: Path) -> set[str]:
    if not path.exists():
        pytest.skip(f"{path.name} not implemented yet")
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def calls_open(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"
        for n in ast.walk(tree)
    )


def starts(names: set[str], prefix: str) -> bool:
    return any(n == prefix or n.startswith(prefix + ".") for n in names)


def test_core_never_imports_streamlit():
    for path in SRC.rglob("*.py"):
        if "ui" in path.relative_to(SRC).parts:
            continue
        assert not starts(imports_of(path), "streamlit"), path


def test_ui_never_touches_ml_storage_or_files():
    for path in (SRC / "ui").rglob("*.py"):
        names = imports_of(path)
        for banned in ("transformers", "huggingface_hub", "minio", "shutil", "torch"):
            assert not starts(names, banned), f"{path} imports {banned}"
        assert not calls_open(path), f"{path} calls open()"


def test_classifier_independent_of_manager_and_storage():
    names = imports_of(SRC / "services" / "classifier.py")
    assert not starts(names, "zeroshot.services.model_manager")
    assert not starts(names, "zeroshot.services.storage")


def test_manager_independent_of_classifier():
    names = imports_of(SRC / "services" / "model_manager.py")
    assert not starts(names, "zeroshot.services.classifier")
