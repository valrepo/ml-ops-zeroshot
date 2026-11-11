"""Model catalogue and installation entities (data-model.md §1–3)."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from zeroshot.core.paths import is_commit_sha, slug_for, validate_repo_id
from zeroshot.errors import PathSafetyError

DEFAULT_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]
DEFAULT_IGNORE_PATTERNS = ["onnx/*", "*.bin", "*.msgpack", "*.ot", "*.h5"]
_KEY = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


class DownloadSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    allow_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOW_PATTERNS))
    ignore_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))


class InferenceDefaults(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    multi_label: bool
    hypothesis_template: str

    @field_validator("hypothesis_template")
    @classmethod
    def _one_placeholder(cls, v: str) -> str:
        if v.count("{}") != 1 or v.replace("{}", "").count("{") or v.replace("{}", "").count("}"):
            raise ValueError("hypothesis_template must contain exactly one '{}'")
        return v


class ModelDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    name: str = Field(min_length=1, max_length=80)
    repo_id: str
    task: Literal["zero-shot-classification"]
    revision: str
    license: str | None = None
    inference_defaults: InferenceDefaults
    download: DownloadSpec = Field(default_factory=DownloadSpec)

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        if not _KEY.fullmatch(v):
            raise ValueError("key must match ^[a-z0-9][a-z0-9-]{1,62}$")
        return v

    @field_validator("repo_id")
    @classmethod
    def _repo(cls, v: str) -> str:
        try:
            return validate_repo_id(v)
        except PathSafetyError as exc:
            raise ValueError(exc.user_message) from exc

    @field_validator("revision")
    @classmethod
    def _revision(cls, v: str) -> str:
        if not is_commit_sha(v):
            raise ValueError(
                "revision must be a 40-char commit SHA, not a branch/tag such as 'main'"
            )
        return v

    @property
    def slug(self) -> str:
        return slug_for(self.repo_id)


class CompletionMarker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    marker_version: Literal["1"] = "1"
    model_id: str
    revision: str
    downloaded_at: str
    files: list[str]
    total_bytes: int = Field(ge=0)


class ModelStatus(StrEnum):
    NOT_AVAILABLE = "not_available"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


STATUS_LABELS = {
    ModelStatus.NOT_AVAILABLE: "Not available",
    ModelStatus.DOWNLOADING: "Downloading…",
    ModelStatus.READY: "Ready",
    ModelStatus.FAILED: "Failed",
    ModelStatus.INCOMPLETE: "Incomplete installation",
}


class ModelInstallation(BaseModel):
    model_config = ConfigDict(frozen=True)
    definition: ModelDefinition
    status: ModelStatus
    local_path: Path | None = None
    marker: CompletionMarker | None = None
    error: str | None = None
    in_use: bool = False

    @property
    def key(self) -> str:
        return self.definition.key

    @property
    def status_label(self) -> str:
        label = STATUS_LABELS[self.status]
        return (
            f"{label}: {self.error}" if self.status is ModelStatus.FAILED and self.error else label
        )


class DownloadOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["downloaded", "already_present", "failed"]
    error: str | None = None
    marker: CompletionMarker | None = None


class CacheCleanupReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["cleared", "nothing_to_clear"]
    removed_bytes: int = 0
    targets: list[Path] = Field(default_factory=list)


class StoragePaths(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_dir: Path
    hf_home: Path
    hf_cache_targets: list[Path]
