import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_hygiene.py"


def repo_with(tmp_path, files: dict[str, bytes]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "."], check=True)
    return tmp_path


def run(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)], capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "bad",
    [
        ".env",
        "models/x/model.safetensors",
        ".cache/huggingface/hub/a",
        "weights.bin",
        "data/label_sets.yaml",
        "homework02-brief.md",
    ],
)
def test_fails_on_tracked_forbidden_file(tmp_path, bad):
    result = run(repo_with(tmp_path, {"src/app.py": b"x", bad: b"x"}))
    assert result.returncode == 1
    assert bad in result.stderr


def test_fails_on_large_file(tmp_path):
    result = run(repo_with(tmp_path, {"big.csv": b"0" * (5 * 1024 * 1024 + 1)}))
    assert result.returncode == 1 and "5 MB" in result.stderr


def test_passes_on_clean_repo(tmp_path):
    result = run(
        repo_with(
            tmp_path,
            {".env.example": b"A=1", "models/.gitkeep": b"", "src/app.py": b"print(1)"},
        )
    )
    assert result.returncode == 0, result.stderr
