import json
import logging
import re

import pytest
from fakes import FakePipeline, FakeStorage

from zeroshot.domain.predictions import OutcomeStatus
from zeroshot.errors import InternalError
from zeroshot.services.classifier import Classifier
from zeroshot.services.label_sets import LabelSetService
from zeroshot.services.model_manager import ModelManager
from zeroshot.services.prediction import PredictionService

TEXT = "The team won the championship after a very close game."
UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@pytest.fixture
def env(settings, catalogue, registry, ready_model):
    ready_model(catalogue[0])  # DeBERTa ready, BART not
    pipe = FakePipeline()
    storage = FakeStorage()
    mm = ModelManager(settings, catalogue, registry)
    labels = LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)
    clf = Classifier(settings, registry, pipeline_factory=lambda path, device: pipe)
    svc = PredictionService(settings, mm, labels, clf, storage, registry)
    return svc, storage, pipe, catalogue, registry


def test_success(env, settings):
    svc, storage, _, cat, _ = env
    out = svc.predict(TEXT, cat[0].key, "default")
    assert out.status is OutcomeStatus.SUCCEEDED
    assert out.object_key in storage.objects
    rec = out.record
    assert UUID4.fullmatch(rec.request_id) and out.request_id == rec.request_id
    stored = json.loads(storage.objects[out.object_key])
    assert stored["timestamp"].endswith("Z")
    assert stored["label_set"] == "default"
    assert stored["application_version"] == settings.app_version
    assert stored["model_revision"] == cat[0].revision
    assert stored["model_name"] == cat[0].name
    assert stored["inference_parameters"] == {
        "multi_label": False,
        "hypothesis_template": "This text is about {}",
        "device": "cpu",
    }
    assert stored["predicted_class"] == "sports"
    assert stored["inference_latency_ms"] >= 0


@pytest.mark.parametrize(
    "text,model,label_set",
    [
        ("", "deberta-v3-zeroshot-v2", "default"),
        ("x" * 5001, "deberta-v3-zeroshot-v2", "default"),
        (TEXT, "nope", "default"),
        (TEXT, "bart-large-mnli", "default"),  # not ready
        (TEXT, "deberta-v3-zeroshot-v2", "missing"),
    ],
)
def test_failures_store_nothing(env, text, model, label_set):
    svc, storage, _, _, _ = env
    out = svc.predict(text, model, label_set)
    assert out.status is OutcomeStatus.FAILED
    assert out.error is not None and out.error.user_message
    assert storage.objects == {}


def test_inference_error_is_failed(env):
    svc, storage, pipe, cat, _ = env
    pipe.fail = RuntimeError("kaboom")
    out = svc.predict(TEXT, cat[0].key, "default")
    assert out.status is OutcomeStatus.FAILED and storage.objects == {}


def test_storage_failure_is_computed_not_persisted(env):
    svc, storage, _, cat, _ = env
    storage.fail_on_save = True
    out = svc.predict(TEXT, cat[0].key, "default")
    assert out.status is OutcomeStatus.COMPUTED_NOT_PERSISTED
    assert out.result is not None and out.record is not None and out.object_key is None
    assert out.error.code == "storage_write_failed"


def test_bucket_init_failure_is_computed_not_persisted(env):
    svc, storage, _, cat, _ = env
    storage.fail_on_ensure = True
    assert svc.predict(TEXT, cat[0].key, "default").status is OutcomeStatus.COMPUTED_NOT_PERSISTED


def test_lease_held_during_inference_and_released(settings, catalogue, registry, ready_model):
    ready_model(catalogue[0])
    key = catalogue[0].key
    seen = []

    class SpyPipeline(FakePipeline):
        def __call__(self, *args, **kwargs):
            seen.append(registry.is_in_use(key))
            return super().__call__(*args, **kwargs)

    spy = SpyPipeline()
    svc = PredictionService(
        settings,
        ModelManager(settings, catalogue, registry),
        LabelSetService(settings.label_sets_file, settings.label_sets_seed_file),
        Classifier(settings, registry, pipeline_factory=lambda path, device: spy),
        FakeStorage(),
        registry,
    )
    svc.predict(TEXT, key, "default")
    assert seen == [True] and not registry.is_in_use(key)
    spy.fail = RuntimeError("x")
    assert svc.predict(TEXT, key, "default").status is OutcomeStatus.FAILED
    assert not registry.is_in_use(key)


def test_unexpected_exception_never_raises(env, monkeypatch):
    svc, storage, _, cat, _ = env
    monkeypatch.setattr(svc._classifier, "classify", lambda *a, **k: 1 / 0)
    out = svc.predict(TEXT, cat[0].key, "default")
    assert out.status is OutcomeStatus.FAILED
    assert isinstance(out.error, InternalError)
    assert storage.objects == {}


def test_logs_have_request_id_and_never_text(env, caplog):
    svc, _, _, cat, _ = env
    secret_text = "Confidential merger text 12345 " + TEXT
    with caplog.at_level(logging.DEBUG, logger="zeroshot"):
        out = svc.predict(secret_text, cat[0].key, "default")
    events = [r for r in caplog.records if getattr(r, "event", "").startswith("prediction.")]
    assert {r.event for r in events} >= {
        "prediction.started",
        "prediction.inferred",
        "prediction.persisted",
    }
    for r in events:
        assert r.zs_fields["request_id"] == out.request_id
    blob = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "Confidential merger" not in blob
