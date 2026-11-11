"""Application settings and configuration-file loaders (contracts/configuration.md).

``load_settings`` never logs (logging is configured afterwards by ``build_services``);
non-fatal warnings are collected in ``Settings.warnings`` instead.
"""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from zeroshot import __version__
from zeroshot.domain.models import ModelDefinition
from zeroshot.errors import ConfigError

_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_SECRET_FIELDS = {"minio_access_key", "minio_secret_key", "hf_token"}


def validate_bucket_name(name: str) -> str:
    """S3 bucket naming rules, as enforced by MinIO (research R1, ADR-0002)."""
    is_ip = False
    try:
        ipaddress.IPv4Address(name)
        is_ip = True
    except ValueError:
        pass
    if not _BUCKET.fullmatch(name) or ".." in name or is_ip:
        raise ConfigError(
            f"MINIO_BUCKET '{name}' is not a valid S3 bucket name (lowercase letters, digits, "
            "'-' and '.' only; see docs/adr/0002-bucket-name-hyphens.md)",
            code="invalid_bucket_name",
        )
    return name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_version: str = __version__
    model_dir: Path = Path("./models")
    hf_home: Path = Path("./.cache/huggingface")
    hf_token: SecretStr | None = None
    models_file: Path = Path("./configs/models.yaml")
    label_sets_file: Path = Path("./data/label_sets.yaml")
    label_sets_seed_file: Path = Path("./configs/labels.yaml")
    max_input_chars: int = Field(default=5000, ge=1, le=100_000)
    max_loaded_models: int = Field(default=1, ge=1)
    inference_device: Literal["cpu", "mps"] = "cpu"
    minio_endpoint: str = "localhost:9010"
    minio_access_key: SecretStr
    minio_secret_key: SecretStr
    minio_secure: bool = False
    minio_bucket: str = "zeroshot-learning-output"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    _warnings: list[str] = PrivateAttr(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        """Non-fatal configuration warnings, logged by build_services (FR-10)."""
        return self._warnings

    @field_validator("hf_token", mode="before")
    @classmethod
    def _empty_token(cls, v: object) -> object:
        return None if v in ("", None) else v

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    @field_validator("minio_bucket")
    @classmethod
    def _bucket(cls, v: str) -> str:
        return validate_bucket_name(v)

    @field_validator("minio_endpoint")
    @classmethod
    def _endpoint(cls, v: str) -> str:
        if "://" in v or "/" in v or not v:
            raise ValueError("must be host:port without scheme or path, e.g. localhost:9010")
        return v

    @model_validator(mode="after")
    def _device_available(self) -> Settings:
        if self.inference_device == "mps":
            import torch  # lazy: only needed when MPS is requested

            if not torch.backends.mps.is_available():
                raise ConfigError(
                    "INFERENCE_DEVICE=mps was requested but MPS is not available on this "
                    "machine; use INFERENCE_DEVICE=cpu.",
                    code="device_unavailable",
                )
        return self

    def model_post_init(self, context: object, /) -> None:
        # Private attributes exist only from here on (after validation).
        super().model_post_init(context)
        if os.environ.get("TRANSFORMERS_CACHE"):
            self._warnings.append("TRANSFORMERS_CACHE is ignored; HF_HOME is used (ADR-0001)")


def load_settings(**overrides: object) -> Settings:
    """Load settings from the environment/.env, wrapping errors in a secret-free ConfigError."""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ConfigError:
        raise
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "settings"
            var = field.upper()
            if err["type"] == "missing":
                problems.append(f"{var} is required")
            elif field in _SECRET_FIELDS:
                problems.append(f"{var} is invalid")
            else:
                problems.append(f"{var}: {err['msg']}")
        code = "config_missing" if all(e["type"] == "missing" for e in exc.errors()) else None
        raise ConfigError(
            "Invalid configuration: " + "; ".join(problems), code=code or "config_invalid"
        ) from None


def _read_yaml(path: Path, what: str) -> object:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{what} file not found: {path}", code="config_missing") from None
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"{what} file {path} is not valid YAML: {exc}", code="catalogue_invalid"
        ) from None


def load_model_catalogue(path: Path) -> list[ModelDefinition]:
    data = _read_yaml(path, "Model catalogue") or {}
    entries = data.get("models") if isinstance(data, dict) else None
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise ConfigError(f"{path}: 'models' must be a list", code="catalogue_invalid")
    catalogue: list[ModelDefinition] = []
    keys: set[str] = set()
    repos: set[str] = set()
    for index, raw in enumerate(entries):
        try:
            definition = ModelDefinition.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"]) or "entry"
            raise ConfigError(
                f"{path}: model #{index + 1} field '{loc}': {first['msg']}",
                code="catalogue_invalid",
            ) from None
        if definition.key in keys:
            raise ConfigError(
                f"{path}: duplicate model key '{definition.key}'", code="catalogue_invalid"
            )
        if definition.repo_id in repos:
            raise ConfigError(
                f"{path}: duplicate repo_id '{definition.repo_id}'", code="catalogue_invalid"
            )
        keys.add(definition.key)
        repos.add(definition.repo_id)
        catalogue.append(definition)
    return catalogue
