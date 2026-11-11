import io
import json
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from fakes import sample_record_data as base
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError, ProtocolError

from zeroshot.domain.predictions import PredictionRecord
from zeroshot.errors import BucketInitError, StorageUnavailableError, StorageWriteError
from zeroshot.services.storage import StorageService


def s3_error(code="AccessDenied"):
    return S3Error(None, code, "denied", "/b", "rid", "hid")


@pytest.fixture
def client():
    c = Mock()
    c.bucket_exists.return_value = True
    return c


@pytest.fixture
def svc(settings, client):
    return StorageService(settings, client=client)


def record(**over):
    return PredictionRecord(**base(**over))


def test_object_key_uses_utc_date(svc):
    ts = datetime(2026, 11, 10, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    r = record(timestamp=ts)
    assert svc.object_key(r) == f"predictions/2026/11/11/{r.request_id}.json"


def test_ensure_bucket_creates_only_when_missing_and_caches(settings):
    c = Mock()
    c.bucket_exists.return_value = False
    s = StorageService(settings, client=c)
    s.ensure_bucket()
    s.ensure_bucket()
    c.make_bucket.assert_called_once_with(settings.minio_bucket)
    assert c.bucket_exists.call_count == 1


def test_existing_bucket_reused(svc, client):
    svc.ensure_bucket()
    client.make_bucket.assert_not_called()


def test_save_puts_json(svc, client, settings):
    r = record()
    key = svc.save(r)
    args, kwargs = client.put_object.call_args
    assert args[0] == settings.minio_bucket and args[1] == key
    data: io.BytesIO = args[2]
    assert args[3] == len(data.getvalue())
    assert kwargs["content_type"] == "application/json; charset=utf-8"
    assert json.loads(data.getvalue())["request_id"] == r.request_id


def test_invalid_record_never_reaches_put(svc, client):
    bad = record().model_copy(update={"confidence": 0.1})  # bypasses validation
    with pytest.raises(StorageWriteError) as exc:
        svc.save(bad)
    assert exc.value.code == "invalid_record"
    client.put_object.assert_not_called()


def test_s3_error_in_ensure_is_bucket_init_error(settings):
    c = Mock()
    c.bucket_exists.side_effect = s3_error()
    with pytest.raises(BucketInitError):
        StorageService(settings, client=c).ensure_bucket()


def test_s3_error_in_save_is_write_error(svc, client):
    client.put_object.side_effect = s3_error()
    with pytest.raises(StorageWriteError) as exc:
        svc.save(record())
    assert exc.value.code == "storage_write_failed"


@pytest.mark.parametrize(
    "error",
    [
        MaxRetryError(None, "/", "boom"),
        ProtocolError("reset"),
        ConnectionError("refused"),
        TimeoutError("slow"),
    ],
)
@pytest.mark.parametrize("where", ["ensure", "save"])
def test_connection_errors_are_unavailable(settings, error, where):
    c = Mock()
    if where == "ensure":
        c.bucket_exists.side_effect = error
    else:
        c.bucket_exists.return_value = True
        c.put_object.side_effect = error
    s = StorageService(settings, client=c)
    with pytest.raises(StorageUnavailableError):
        s.ensure_bucket() if where == "ensure" else s.save(record())


def test_errors_contain_no_credentials(settings):
    c = Mock()
    c.bucket_exists.side_effect = s3_error()
    with pytest.raises(BucketInitError) as exc:
        StorageService(settings, client=c).ensure_bucket()
    assert settings.minio_secret_key.get_secret_value() not in exc.value.user_message
    assert settings.minio_access_key.get_secret_value() not in exc.value.user_message


def test_bucket_retried_after_failure(settings):
    c = Mock()
    c.bucket_exists.side_effect = [ConnectionError("down"), True]
    s = StorageService(settings, client=c)
    with pytest.raises(StorageUnavailableError):
        s.ensure_bucket()
    s.save(record())
    c.put_object.assert_called_once()


def test_now_timestamp_key(svc):
    r = record(timestamp=datetime.now(UTC))
    assert svc.object_key(r).startswith(f"predictions/{r.timestamp:%Y/%m/%d}/")
