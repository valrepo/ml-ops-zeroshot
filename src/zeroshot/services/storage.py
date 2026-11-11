"""MinIO persistence of prediction records — the system of record (Principle VI).

Object key: ``predictions/YYYY/MM/DD/{request_id}.json`` using the record's UTC timestamp.
Exception mapping is normative (contracts/services.md, "Exception mapping").
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Protocol

import urllib3
from minio import Minio
from minio.error import S3Error
from pydantic import ValidationError
from urllib3.exceptions import MaxRetryError, ProtocolError

from zeroshot.config import Settings
from zeroshot.domain.predictions import PredictionRecord
from zeroshot.errors import (
    BucketInitError,
    StorageError,
    StorageUnavailableError,
    StorageWriteError,
)
from zeroshot.logging import log_event

log = logging.getLogger(__name__)

CONTENT_TYPE = "application/json; charset=utf-8"
_CONNECTION_ERRORS = (MaxRetryError, ProtocolError, ConnectionError, TimeoutError)


class PredictionStore(Protocol):
    def ensure_bucket(self) -> None: ...
    def object_key(self, record: PredictionRecord) -> str: ...
    def save(self, record: PredictionRecord) -> str: ...
    def get(self, key: str) -> PredictionRecord: ...


def _unavailable(endpoint: str) -> StorageUnavailableError:
    return StorageUnavailableError(
        f"MinIO is not reachable at {endpoint}. Is it running? (docker compose up -d --wait minio)"
    )


class StorageService:
    def __init__(self, settings: Settings, client: Minio | None = None) -> None:
        self._endpoint = settings.minio_endpoint
        self._bucket = settings.minio_bucket
        self._client = client or Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key.get_secret_value(),
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=settings.minio_secure,
            http_client=urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=3, read=10),
                retries=urllib3.Retry(total=2, backoff_factor=0.3),
            ),
        )
        self._bucket_ready = False
        self._lock = threading.Lock()

    @property
    def bucket(self) -> str:
        return self._bucket

    def ensure_bucket(self) -> None:
        with self._lock:
            if self._bucket_ready:
                return
            try:
                if self._client.bucket_exists(self._bucket):
                    log_event(log, "storage.bucket.exists", bucket=self._bucket)
                else:
                    self._client.make_bucket(self._bucket)
                    log_event(log, "storage.bucket.created", bucket=self._bucket)
            except _CONNECTION_ERRORS:
                log_event(
                    log,
                    "storage.bucket.failed",
                    logging.ERROR,
                    bucket=self._bucket,
                    reason="unavailable",
                )
                raise _unavailable(self._endpoint) from None
            except S3Error as exc:
                log_event(
                    log,
                    "storage.bucket.failed",
                    logging.ERROR,
                    bucket=self._bucket,
                    reason=exc.code,
                )
                raise BucketInitError(
                    f"Could not create or access bucket '{self._bucket}' ({exc.code})."
                ) from None
            self._bucket_ready = True

    def object_key(self, record: PredictionRecord) -> str:
        ts = record.timestamp  # always UTC (validated by PredictionRecord)
        return f"predictions/{ts:%Y}/{ts:%m}/{ts:%d}/{record.request_id}.json"

    def save(self, record: PredictionRecord) -> str:
        try:
            # Re-validate: records built with model_construct() or mutated must never be written.
            record = PredictionRecord.model_validate(record.model_dump())
            data = record.to_json_bytes()
        except (ValidationError, ValueError) as exc:
            raise StorageWriteError(
                f"The prediction record is invalid and was not written: {exc}",
                code="invalid_record",
            ) from None
        self.ensure_bucket()
        key = self.object_key(record)
        try:
            self._client.put_object(
                self._bucket, key, io.BytesIO(data), len(data), content_type=CONTENT_TYPE
            )
        except _CONNECTION_ERRORS:
            raise _unavailable(self._endpoint) from None
        except S3Error as exc:
            raise StorageWriteError(
                f"MinIO refused to store the prediction ({exc.code})."
            ) from None
        return key

    def get(self, key: str) -> PredictionRecord:
        try:
            response = self._client.get_object(self._bucket, key)
            try:
                return PredictionRecord.model_validate_json(response.read())
            finally:
                response.close()
                response.release_conn()
        except _CONNECTION_ERRORS:
            raise _unavailable(self._endpoint) from None
        except S3Error as exc:
            raise StorageError(
                f"Could not read '{key}' ({exc.code}).", code="storage_read_failed"
            ) from None

    def latest_keys(self, limit: int = 1) -> list[str]:
        """Newest prediction keys (used by ``tools/show_latest``)."""
        try:
            objects = list(
                self._client.list_objects(self._bucket, prefix="predictions/", recursive=True)
            )
        except _CONNECTION_ERRORS:
            raise _unavailable(self._endpoint) from None
        except S3Error as exc:
            if exc.code == "NoSuchBucket":
                return []  # nothing has been stored yet
            raise StorageError(
                f"Could not list predictions ({exc.code}).", code="storage_read_failed"
            ) from None
        objects.sort(key=lambda o: o.last_modified, reverse=True)
        return [o.object_name for o in objects[:limit]]
