"""Bucket behaviour against a real MinIO (AC-03) + reproducible evidence for ADR-0002."""

import pytest
from minio.error import S3Error

from zeroshot.config import load_settings
from zeroshot.errors import ConfigError
from zeroshot.services.storage import StorageService

pytestmark = pytest.mark.integration


def test_ensure_bucket_creates_and_is_idempotent(minio_settings, minio_client, test_bucket):
    s = minio_settings.model_copy(update={"minio_bucket": test_bucket})
    svc = StorageService(s, client=minio_client)
    assert not minio_client.bucket_exists(test_bucket)
    svc.ensure_bucket()
    assert minio_client.bucket_exists(test_bucket)
    StorageService(s, client=minio_client).ensure_bucket()  # second call: no error


def test_default_bucket_name_is_accepted(minio_client):
    name = "zeroshot-learning-output"
    existed = minio_client.bucket_exists(name)
    if not existed:
        minio_client.make_bucket(name)
    assert minio_client.bucket_exists(name)
    if not existed:
        minio_client.remove_bucket(name)


def test_required_underscore_name_is_rejected_by_server(minio_client):
    """ADR-0002 evidence: the assignment's bucket name is rejected at *both* levels."""
    # 1. The MinIO server refuses it (bucket_exists is passed through to the server).
    with pytest.raises(S3Error) as exc:
        minio_client.bucket_exists("zeroshot_learning_output")
    assert exc.value.code == "InvalidBucketName"
    # 2. The Python SDK refuses to even send make_bucket for it (client-side validation).
    with pytest.raises(ValueError, match="invalid bucket name"):
        minio_client.make_bucket("zeroshot_learning_output")


def test_settings_reject_underscore_name():
    with pytest.raises(ConfigError) as exc:
        load_settings(
            _env_file=None,
            minio_access_key="a",
            minio_secret_key="b",
            minio_bucket="zeroshot_learning_output",
        )
    assert exc.value.code == "invalid_bucket_name"
