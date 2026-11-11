"""End-to-end persistence (Quality Gate 4, AC-14, AC-15, AC-18): real PredictionService +
real StorageService + real MinIO; a deterministic fake pipeline stands in for the model."""

import json
from pathlib import Path

import pytest
from fakes import FakePipeline
from jsonschema import Draft202012Validator

from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.domain.predictions import OutcomeStatus
from zeroshot.services.classifier import Classifier
from zeroshot.services.label_sets import LabelSetService
from zeroshot.services.model_manager import ModelManager
from zeroshot.services.prediction import PredictionService
from zeroshot.services.storage import StorageService

pytestmark = pytest.mark.integration

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "specs/001-zero-shot-platform/contracts/prediction-record.schema.json"
    ).read_text()
)
TEXT = "The team won the championship after a very close game."


def build(settings, catalogue, storage):
    registry = ModelUsageRegistry()
    mm = ModelManager(settings, catalogue, registry)
    labels = LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)
    clf = Classifier(settings, registry, pipeline_factory=lambda path, device: FakePipeline())
    return PredictionService(settings, mm, labels, clf, storage, registry)


def test_prediction_is_persisted(
    settings, catalogue, ready_model, minio_settings, minio_client, test_bucket
):
    ready_model(catalogue[0])
    s = settings.model_copy(
        update={
            "minio_endpoint": minio_settings.minio_endpoint,
            "minio_access_key": minio_settings.minio_access_key,
            "minio_secret_key": minio_settings.minio_secret_key,
            "minio_bucket": test_bucket,
        }
    )
    svc = build(s, catalogue, StorageService(s, client=minio_client))
    out = svc.predict(TEXT, "deberta-v3-zeroshot-v2", "default")
    assert out.status is OutcomeStatus.SUCCEEDED, out.error

    response = minio_client.get_object(test_bucket, out.object_key)
    try:
        payload = json.loads(response.read())
    finally:
        response.close()
        response.release_conn()
    assert list(Draft202012Validator(SCHEMA).iter_errors(payload)) == []
    assert out.object_key == (
        f"predictions/{out.record.timestamp:%Y/%m/%d}/{payload['request_id']}.json"
    )
    assert payload["model"] == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert payload["model_revision"] == "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"
    assert payload["inference_parameters"]["hypothesis_template"] == "This text is about {}"
    assert payload["inference_parameters"]["device"] == "cpu"
    assert payload["input_text"] == TEXT and payload["predicted_class"] == "sports"
    assert payload["candidate_labels"] == ["sports", "politics", "technology", "finance"]


def test_unreachable_minio_is_computed_not_persisted(settings, catalogue, ready_model):
    ready_model(catalogue[0])
    s = settings.model_copy(update={"minio_endpoint": "localhost:1"})
    svc = build(s, catalogue, StorageService(s))
    out = svc.predict(TEXT, "deberta-v3-zeroshot-v2", "default")
    assert out.status is OutcomeStatus.COMPUTED_NOT_PERSISTED
    assert out.result.predicted_class == "sports"
