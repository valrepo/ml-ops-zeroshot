"""Fail if secrets, model artifacts, caches or oversized files are tracked by git (AC-22).

Usage: python scripts/check_hygiene.py [--repo PATH]    exit 0 = clean, 1 = violations
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BANNED_DIRS = ("models/", ".cache/", "data/", ".venv/")
ALLOWED = {"models/.gitkeep", ".env.example"}
BANNED_EXTENSIONS = (".safetensors", ".bin", ".onnx", ".msgpack", ".ot", ".h5")
BANNED_GLOBS = (".env", ".env.*", "*/.env", "*/.env.*", "homework02-*.md")


def tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"], check=True, capture_output=True
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def violations(repo: Path) -> list[str]:
    problems: list[str] = []
    for path in tracked_files(repo):
        if path in ALLOWED:
            continue
        if any(fnmatch(path, g) for g in BANNED_GLOBS):
            problems.append(f"{path}: secrets file or local-only document")
        elif path.startswith(BANNED_DIRS):
            problems.append(f"{path}: model/cache/runtime data must not be committed")
        elif path.endswith(BANNED_EXTENSIONS):
            problems.append(f"{path}: model weight file")
        else:
            full = repo / path
            if full.is_file() and full.stat().st_size > MAX_BYTES:
                problems.append(f"{path}: larger than 5 MB")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    problems = violations(args.repo)
    for problem in problems:
        print(f"hygiene: {problem}", file=sys.stderr)
    if not problems:
        print("hygiene: OK — no secrets, models, caches or large files are tracked.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
