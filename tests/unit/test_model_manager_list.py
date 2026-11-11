import json

import pytest
from fakes import file_only_validator

from zeroshot.domain.models import ModelStatus
from zeroshot.errors import UnknownModelError
from zeroshot.services.model_manager import ModelManager


@pytest.fixture
def manager(settings, catalogue, registry):
    return ModelManager(settings, catalogue, registry, artifact_validator=file_only_validator)


def test_empty_model_dir_lists_both_not_available(manager):
    models = manager.list_models()
    assert [m.definition.repo_id for m in models] == [
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "facebook/bart-large-mnli",
    ]
    assert {m.status for m in models} == {ModelStatus.NOT_AVAILABLE}
    assert manager.ready_models() == []


def test_ready_after_valid_marker(manager, catalogue, ready_model):
    path = ready_model(catalogue[0])
    inst = manager.get(catalogue[0].key)
    assert inst.status is ModelStatus.READY
    assert inst.local_path == path.resolve()
    assert inst.marker.revision == catalogue[0].revision
    assert [m.key for m in manager.ready_models()] == [catalogue[0].key]


def test_directory_without_marker_is_incomplete(manager, catalogue, settings):
    d = settings.model_dir / catalogue[0].slug / catalogue[0].revision
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    inst = manager.get(catalogue[0].key)
    assert inst.status is ModelStatus.INCOMPLETE
    assert inst.status_label == "Incomplete installation"


@pytest.mark.parametrize("field,value", [("revision", "0" * 40), ("model_id", "other/model")])
def test_mismatched_marker_is_incomplete(manager, catalogue, ready_model, field, value):
    path = ready_model(catalogue[0])
    marker = json.loads((path / ".complete.json").read_text())
    marker[field] = value
    (path / ".complete.json").write_text(json.dumps(marker))
    assert manager.get(catalogue[0].key).status is ModelStatus.INCOMPLETE


def test_marker_listing_missing_file_is_incomplete(manager, catalogue, ready_model):
    path = ready_model(catalogue[0])
    (path / "model.safetensors").unlink()
    assert manager.get(catalogue[0].key).status is ModelStatus.INCOMPLETE


def test_corrupt_marker_is_incomplete(manager, catalogue, ready_model):
    path = ready_model(catalogue[0])
    (path / ".complete.json").write_text("{not json")
    assert manager.get(catalogue[0].key).status is ModelStatus.INCOMPLETE


def test_in_use_flag(manager, catalogue, ready_model, registry):
    ready_model(catalogue[0])
    with registry.lease(catalogue[0].key):
        assert manager.get(catalogue[0].key).in_use is True
    assert manager.get(catalogue[0].key).in_use is False


def test_unknown_model(manager):
    with pytest.raises(UnknownModelError):
        manager.get("nope")


def test_empty_catalogue(settings, registry):
    assert ModelManager(settings, [], registry).list_models() == []


def test_paths(manager, settings):
    paths = manager.paths()
    assert paths.model_dir == settings.model_dir.resolve()
    assert paths.hf_home == settings.hf_home.resolve()
    assert paths.hf_cache_targets == [
        settings.hf_home.resolve() / "hub",
        settings.hf_home.resolve() / "xet",
    ]


def test_ready_installation_details(manager, catalogue, ready_model, settings):
    ready_model(catalogue[1])
    inst = manager.get(catalogue[1].key)
    assert inst.marker.downloaded_at.endswith("Z")
    assert inst.local_path.is_relative_to(settings.model_dir.resolve())
    assert inst.definition.license == "mit"
