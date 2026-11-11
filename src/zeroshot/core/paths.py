"""Filesystem path safety for anything derived from configuration or user input (Principle X)."""

from __future__ import annotations

import re
from pathlib import Path

from zeroshot.errors import PathSafetyError

_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_REPO_ID = re.compile(rf"^{_SEGMENT}/{_SEGMENT}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def validate_repo_id(repo_id: str) -> str:
    """Return ``repo_id`` if it is a safe ``owner/name`` Hugging Face identifier."""
    if not isinstance(repo_id, str) or not _REPO_ID.fullmatch(repo_id):
        raise PathSafetyError(f"Invalid model identifier {repo_id!r} (expected 'owner/name').")
    if any(part in {".", ".."} or ".." in part for part in repo_id.split("/")):
        raise PathSafetyError(f"Invalid model identifier {repo_id!r} ('..' is not allowed).")
    return repo_id


def slug_for(repo_id: str) -> str:
    """Filesystem-safe directory name for a repo id: ``owner/name`` -> ``owner--name``."""
    return validate_repo_id(repo_id).replace("/", "--")


def is_commit_sha(revision: str) -> bool:
    return isinstance(revision, str) and bool(_SHA.fullmatch(revision))


def safe_child(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and guarantee the resolved result stays strictly inside it.

    Symlinks are resolved, so a link pointing outside ``root`` is rejected too.
    """
    base = Path(root).resolve()
    for part in parts:
        if not part or Path(part).is_absolute():
            raise PathSafetyError(f"Unsafe path component {part!r}.")
    candidate = base.joinpath(*parts).resolve()
    if candidate == base or not candidate.is_relative_to(base):
        raise PathSafetyError("Refusing to use a path outside the managed directory.")
    return candidate
