import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import sample_record_data as base
from jsonschema import Draft202012Validator

from zeroshot.domain.predictions import PredictionRecord
from zeroshot.errors import StorageUnavailableError
from zeroshot.services.storage import StorageService

pytestmark = pytest.mark.integration

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "specs/001-zero-shot-platform/contracts/prediction-record.schema.json"
    ).read_text()
)
KEY = re.compile(r"^predictions/\d{4}/\d{2}/\d{2}/[0-9a-f-]{36}\.json$")


@pytest.fixture
def svc(minio_settings, minio_client, test_bucket):
    return StorageService(
        minio_settings.model_copy(update={"minio_bucket": test_bucket}), client=minio_client
    )


def fresh_record():
    return PredictionRecord(**base(request_id=str(uuid.uuid4()), timestamp=datetime.now(UTC)))


def test_round_trip(svc):
    r = fresh_record()
    key = svc.save(r)
    assert KEY.fullmatch(key)
    assert svc.get(key) == r


def test_raw_object_matches_schema_and_content_type(svc, minio_client, test_bucket):
    key = svc.save(fresh_record())
    response = minio_client.get_object(test_bucket, key)
    try:
        payload = json.loads(response.read())
        content_type = response.headers.get("Content-Type", "")
    finally:
        response.close()
        response.release_conn()
    assert list(Draft202012Validator(SCHEMA).iter_errors(payload)) == []
    assert content_type.startswith("application/json")


def test_unavailable_endpoint_fails_fast(minio_settings):
    s = minio_settings.model_copy(update={"minio_endpoint": "localhost:1"})
    started = time.monotonic()
    with pytest.raises(StorageUnavailableError):
        StorageService(s).save(fresh_record())
    assert time.monotonic() - started < 10
