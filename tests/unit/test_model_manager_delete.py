import logging
import shutil

import pytest
from fakes import file_only_validator

from zeroshot.domain.models import ModelStatus
from zeroshot.errors import ModelDeleteError, ModelInUseError, ModelNotReadyError, PathSafetyError
from zeroshot.services.model_manager import ModelManager


@pytest.fixture
def mm(settings, catalogue, registry):
    return ModelManager(settings, catalogue, registry, artifact_validator=file_only_validator)


def test_delete_removes_only_that_model(mm, settings, catalogue, ready_model):
    a = ready_model(catalogue[0])
    b = ready_model(catalogue[1])
    (settings.hf_home / "hub").mkdir(parents=True)
    (settings.hf_home / "hub" / "keep.txt").write_text("x")
    mm.delete(catalogue[0].key)
    assert not a.exists() and not a.parent.exists()  # empty <slug>/ removed too
    assert b.exists()
    assert (settings.hf_home / "hub" / "keep.txt").exists()
    assert mm.get(catalogue[0].key).status is ModelStatus.NOT_AVAILABLE


def test_delete_refused_while_in_use(mm, catalogue, ready_model, registry, caplog):
    path = ready_model(catalogue[0])
    with (
        caplog.at_level(logging.INFO, logger="zeroshot"),
        registry.lease(catalogue[0].key),
        pytest.raises(ModelInUseError),
    ):
        mm.delete(catalogue[0].key)
    assert path.exists()
    assert "model.delete.refused" in [getattr(r, "event", "") for r in caplog.records]


def test_new_lease_refused_during_deletion(mm, catalogue, ready_model, registry, monkeypatch):
    ready_model(catalogue[0])
    seen = []
    real_rmtree = shutil.rmtree

    def rmtree_spy(path, *a, **k):
        with pytest.raises(ModelInUseError) as exc, registry.lease(catalogue[0].key):
            pass
        seen.append(exc.value.code)
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr("zeroshot.services.model_manager.shutil.rmtree", rmtree_spy)
    mm.delete(catalogue[0].key)
    assert seen == ["model_being_deleted"]


def test_symlinked_slug_escaping_model_dir_is_refused(mm, settings, catalogue, tmp_path):
    outside = tmp_path / "outside" / catalogue[0].revision
    outside.mkdir(parents=True)
    (outside / "precious.txt").write_text("keep me")
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    (settings.model_dir / catalogue[0].slug).symlink_to(tmp_path / "outside")
    with pytest.raises(PathSafetyError):
        mm.delete(catalogue[0].key)
    assert (outside / "precious.txt").exists()


def test_rmtree_failure_is_delete_error(mm, catalogue, ready_model, monkeypatch):
    ready_model(catalogue[0])

    def boom(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr("zeroshot.services.model_manager.shutil.rmtree", boom)
    with pytest.raises(ModelDeleteError):
        mm.delete(catalogue[0].key)
    assert mm.get(catalogue[0].key).status is ModelStatus.READY  # state refreshed, still there


def test_delete_notifies_registry(mm, catalogue, ready_model, registry):
    ready_model(catalogue[0])
    removed = []
    registry.on_removed(removed.append)
    mm.delete(catalogue[0].key)
    assert removed == [catalogue[0].key]


def test_delete_not_ready_model(mm, catalogue):
    with pytest.raises(ModelNotReadyError):
        mm.delete(catalogue[0].key)


def test_incomplete_installation_can_be_deleted(mm, settings, catalogue):
    d = settings.model_dir / catalogue[0].slug / catalogue[0].revision
    d.mkdir(parents=True)
    (d / "partial.bin.part").write_text("x")
    mm.delete(catalogue[0].key)
    assert not d.exists()
