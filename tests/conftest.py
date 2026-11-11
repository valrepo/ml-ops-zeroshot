"""Shared, non-autouse fixtures for all test tiers."""

from __future__ import annotations

import json
import os
import shutil
import socket
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from zeroshot.config import Settings, load_model_catalogue
from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.domain.models import ModelDefinition

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    seed = tmp_path / "labels.yaml"
    shutil.copyfile(ROOT / "configs" / "labels.yaml", seed)
    return Settings(
        _env_file=None,
        model_dir=tmp_path / "models",
        hf_home=tmp_path / "hf-home",
        models_file=ROOT / "configs" / "models.yaml",
        label_sets_file=tmp_path / "data" / "label_sets.yaml",
        label_sets_seed_file=seed,
        minio_access_key="test-access",
        minio_secret_key="test-secret-value",
        minio_endpoint="localhost:1",
        app_version="0.1.0-test",
    )


@pytest.fixture
def catalogue() -> list[ModelDefinition]:
    return load_model_catalogue(ROOT / "configs" / "models.yaml")


@pytest.fixture
def registry() -> ModelUsageRegistry:
    return ModelUsageRegistry()


@pytest.fixture
def ready_model(settings):
    """Factory: create a valid, READY installation for ``definition`` under MODEL_DIR."""

    def _make(definition: ModelDefinition) -> Path:
        path = settings.model_dir / definition.slug / definition.revision
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["XForSequenceClassification"],
                    "id2label": {"0": "entailment", "1": "not_entailment"},
                }
            )
        )
        (path / "model.safetensors").write_bytes(b"\x00" * 8)
        files = ["config.json", "model.safetensors"]
        marker = {
            "marker_version": "1",
            "model_id": definition.repo_id,
            "revision": definition.revision,
            "downloaded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "files": files,
            "total_bytes": sum((path / f).stat().st_size for f in files),
        }
        (path / ".complete.json").write_text(json.dumps(marker))
        return path

    return _make


# ---- MinIO fixtures (used by tests/integration and tests/e2e) -------------------------


@pytest.fixture(scope="session")
def minio_settings() -> Settings:
    from zeroshot.config import load_settings

    return load_settings()


@pytest.fixture(scope="session")
def minio_client(minio_settings):
    host, _, port = minio_settings.minio_endpoint.partition(":")
    try:
        with socket.create_connection((host, int(port or 80)), timeout=2):
            pass
    except OSError:
        message = f"MinIO not reachable at {minio_settings.minio_endpoint}"
        if os.environ.get("REQUIRE_MINIO") == "1":
            pytest.fail(f"MinIO required (REQUIRE_MINIO=1) but unreachable: {message}")
        pytest.skip(message)
    from minio import Minio

    return Minio(
        minio_settings.minio_endpoint,
        access_key=minio_settings.minio_access_key.get_secret_value(),
        secret_key=minio_settings.minio_secret_key.get_secret_value(),
        secure=minio_settings.minio_secure,
    )


@pytest.fixture
def test_bucket(minio_client):
    name = f"zs-test-{uuid.uuid4().hex[:8]}"
    yield name
    if minio_client.bucket_exists(name):
        for obj in minio_client.list_objects(name, recursive=True):
            minio_client.remove_object(name, obj.object_name)
        minio_client.remove_bucket(name)
