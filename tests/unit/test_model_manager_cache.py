import logging

import pytest

from zeroshot.domain.models import ModelStatus
from zeroshot.errors import CacheCleanupError, CacheInUseError
from zeroshot.services.model_manager import ModelManager


@pytest.fixture
def mm(settings, catalogue, registry):
    return ModelManager(settings, catalogue, registry)


def populate(settings):
    hub = settings.hf_home / "hub" / "models--x"
    xet = settings.hf_home / "xet" / "chunks"
    hub.mkdir(parents=True)
    xet.mkdir(parents=True)
    (hub / "weights").write_bytes(b"a" * 1000)
    (xet / "c1").write_bytes(b"b" * 500)
    (settings.hf_home / "token").write_text("hf_keep")


def test_clears_hub_and_xet_only(mm, settings, catalogue, ready_model, caplog):
    populate(settings)
    ready_model(catalogue[0])
    with caplog.at_level(logging.INFO, logger="zeroshot"):
        report = mm.clear_hf_cache()
    assert report.status == "cleared" and report.removed_bytes == 1500
    assert list((settings.hf_home / "hub").iterdir()) == []
    assert list((settings.hf_home / "xet").iterdir()) == []
    assert (settings.hf_home / "token").read_text() == "hf_keep"
    assert mm.get(catalogue[0].key).status is ModelStatus.READY
    assert "cache.cleared" in [getattr(r, "event", "") for r in caplog.records]


def test_missing_cache_is_noop(mm, settings):
    assert not settings.hf_home.exists()
    report = mm.clear_hf_cache()
    assert report.status == "nothing_to_clear" and report.removed_bytes == 0


def test_empty_subdirs_is_noop(mm, settings):
    (settings.hf_home / "hub").mkdir(parents=True)
    assert mm.clear_hf_cache().status == "nothing_to_clear"


def test_refused_while_lease_or_download(mm, settings, registry):
    populate(settings)
    with registry.lease("m"), pytest.raises(CacheInUseError):
        mm.clear_hf_cache()
    with registry.downloading("m"), pytest.raises(CacheInUseError):
        mm.clear_hf_cache()
    assert (settings.hf_home / "hub" / "models--x" / "weights").exists()


def test_oserror_is_cleanup_error(mm, settings, monkeypatch, caplog):
    populate(settings)

    def boom(*_a, **_k):
        raise OSError("busy")

    monkeypatch.setattr("zeroshot.services.model_manager.shutil.rmtree", boom)
    with caplog.at_level(logging.INFO, logger="zeroshot"), pytest.raises(CacheCleanupError):
        mm.clear_hf_cache()
    assert "cache.cleanup.failed" in [getattr(r, "event", "") for r in caplog.records]


def test_symlink_is_unlinked_not_followed(mm, settings, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "precious.txt").write_text("keep")
    (settings.hf_home / "hub").mkdir(parents=True)
    (settings.hf_home / "hub" / "link").symlink_to(target)
    mm.clear_hf_cache()
    assert not (settings.hf_home / "hub" / "link").exists()
    assert (target / "precious.txt").exists()
