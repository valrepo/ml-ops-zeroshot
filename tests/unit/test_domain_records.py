import json
from datetime import datetime, timedelta, timezone

import pytest
from fakes import sample_record_data as base
from pydantic import ValidationError

from zeroshot.domain.predictions import PredictionRecord


def test_valid_record():
    r = PredictionRecord(**base())
    payload = json.loads(r.to_json_bytes())
    assert payload["timestamp"] == "2026-10-31T12:00:00Z"
    assert payload["schema_version"] == "1.0"
    assert payload["inference_parameters"]["device"] == "cpu"


@pytest.mark.parametrize(
    "over",
    [
        {"confidence": 0.5},
        {"predicted_class": "politics"},
        {
            "ranked_predictions": [
                {"label": "technology", "score": 0.02},
                {"label": "sports", "score": 0.96},
                {"label": "finance", "score": 0.01},
                {"label": "politics", "score": 0.01},
            ]
        },
        {"candidate_labels": ["sports", "politics", "technology"]},
        {"candidate_labels": ["sports", "politics", "technology", "finance", "health"]},
        {
            "ranked_predictions": [
                {"label": "sports", "score": 0.96},
                {"label": "technology", "score": 0.5},
                {"label": "finance", "score": 0.01},
                {"label": "politics", "score": 0.01},
            ]
        },
        {"model_revision": "main"},
        {"request_id": "not-a-uuid"},
        {"extra_field": 1},
        {
            "inference_parameters": {
                "multi_label": False,
                "hypothesis_template": "This text is about {}",
            }
        },
        {
            "inference_parameters": {
                "multi_label": False,
                "hypothesis_template": "x {}",
                "device": "cuda",
            }
        },
        {
            "inference_parameters": {
                "multi_label": False,
                "hypothesis_template": "{} and {}",
                "device": "cpu",
            }
        },
    ],
)
def test_invariants_rejected(over):
    with pytest.raises(ValidationError):
        PredictionRecord(**base(**over))


def test_nan_rejected():
    ranked = base()["ranked_predictions"]
    ranked[3] = {"label": "politics", "score": float("nan")}
    with pytest.raises(ValidationError):
        PredictionRecord(**base(ranked_predictions=ranked))


def test_multi_label_does_not_require_sum_one():
    ranked = [
        {"label": "sports", "score": 0.9},
        {"label": "technology", "score": 0.8},
        {"label": "finance", "score": 0.3},
        {"label": "politics", "score": 0.1},
    ]
    params = {"multi_label": True, "hypothesis_template": "This text is about {}", "device": "cpu"}
    r = PredictionRecord(
        **base(ranked_predictions=ranked, confidence=0.9, inference_parameters=params)
    )
    assert r.confidence == 0.9


def test_non_utc_timestamp_normalized():
    ts = datetime(2026, 11, 10, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    r = PredictionRecord(**base(timestamp=ts))
    assert json.loads(r.to_json_bytes())["timestamp"] == "2026-11-11T04:30:00Z"


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        PredictionRecord(**base(timestamp=datetime(2026, 1, 1)))
