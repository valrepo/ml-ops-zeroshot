import pytest

from zeroshot.core.paths import is_commit_sha, safe_child, slug_for, validate_repo_id
from zeroshot.errors import PathSafetyError


@pytest.mark.parametrize(
    "bad", ["../../etc", "a/../b", "/abs/path", "a/b/c", "", "with space/x", "a/..", "../x"]
)
def test_malicious_repo_ids_rejected(bad):
    with pytest.raises(PathSafetyError):
        validate_repo_id(bad)


@pytest.mark.parametrize(
    "good", ["MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "facebook/bart-large-mnli"]
)
def test_catalogue_ids_accepted(good):
    assert validate_repo_id(good) == good


def test_slug():
    assert slug_for("facebook/bart-large-mnli") == "facebook--bart-large-mnli"


def test_commit_sha():
    assert is_commit_sha("d7645e127eaf1aefc7862fd59a17a5aa8558b8ce")
    assert not is_commit_sha("main")
    assert not is_commit_sha("D7645E127EAF1AEFC7862FD59A17A5AA8558B8CE")


def test_safe_child_inside(tmp_path):
    assert safe_child(tmp_path, "a", "b") == (tmp_path / "a" / "b").resolve()


def test_safe_child_rejects_root_and_traversal(tmp_path):
    with pytest.raises(PathSafetyError):
        safe_child(tmp_path)
    with pytest.raises(PathSafetyError):
        safe_child(tmp_path, "..", "x")
    with pytest.raises(PathSafetyError):
        safe_child(tmp_path, "/etc")


def test_safe_child_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(PathSafetyError):
        safe_child(root, "link")
