"""The pydantic PredictionRecord and the JSON Schema contract must not drift apart."""

import json
from pathlib import Path

from fakes import sample_record_data as base
from jsonschema import Draft202012Validator

from zeroshot.domain.predictions import PredictionRecord

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "specs/001-zero-shot-platform/contracts/prediction-record.schema.json"
    ).read_text()
)


def test_serialized_record_validates_against_schema():
    payload = json.loads(PredictionRecord(**base()).to_json_bytes())
    errors = [e.message for e in Draft202012Validator(SCHEMA).iter_errors(payload)]
    assert errors == []


def test_required_and_property_sets_match():
    fields = PredictionRecord.model_fields
    required_pyd = {n for n, f in fields.items() if f.is_required()} | {"schema_version"}
    assert set(SCHEMA["required"]) == required_pyd
    assert set(SCHEMA["properties"]) == set(fields)


def test_inference_parameters_match():
    from zeroshot.domain.predictions import InferenceParameters

    sub = SCHEMA["properties"]["inference_parameters"]
    assert set(sub["required"]) == set(InferenceParameters.model_fields)
