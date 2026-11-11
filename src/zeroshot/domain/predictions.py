"""Classification results and the persisted prediction record (data-model.md §5–9).

``PredictionRecord`` mirrors ``contracts/prediction-record.schema.json``; a unit test keeps
the two in sync. Cross-field invariants (``x-invariants``) are enforced here.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from zeroshot.errors import ZeroShotError

_REPO_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
_UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_SUM_TOLERANCE = 1e-6


class InferenceParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    multi_label: bool
    hypothesis_template: str = Field(pattern=r"^[^{}]*\{\}[^{}]*$")
    device: Literal["cpu", "mps"]


class LabelScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class ClassificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    ranked: list[LabelScore]
    model_id: str
    model_revision: str
    inference_parameters: InferenceParameters
    latency_ms: float

    @property
    def predicted_class(self) -> str:
        return self.ranked[0].label

    @property
    def confidence(self) -> float:
        return self.ranked[0].score


def utc_timestamp(value: datetime | None = None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class PredictionRecord(BaseModel):
    """One persisted classification (schema_version 1.0)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    timestamp: datetime
    input_text: str = Field(min_length=1)
    predicted_class: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    candidate_labels: list[str] = Field(min_length=2)
    ranked_predictions: list[LabelScore] = Field(min_length=2)
    model_name: str = Field(min_length=1)
    model: str = Field(pattern=_REPO_ID)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    inference_parameters: InferenceParameters
    label_set: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")
    application_version: str = Field(min_length=1)
    inference_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    session_id: str | None = None

    @field_validator("request_id", mode="before")
    @classmethod
    def _request_id(cls, v: Any) -> str:
        value = str(v) if isinstance(v, UUID) else v
        if not isinstance(value, str) or not _UUID4.fullmatch(value):
            raise ValueError("request_id must be a lowercase UUID4 string")
        return value

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return utc_timestamp(v)

    @field_validator("candidate_labels")
    @classmethod
    def _labels(cls, v: list[str]) -> list[str]:
        if any(not label or len(label) > 100 for label in v):
            raise ValueError("candidate labels must be 1–100 characters")
        if len(set(v)) != len(v):
            raise ValueError("candidate labels must be unique")
        return v

    @field_serializer("timestamp")
    def _ser_ts(self, v: datetime) -> str:
        return v.isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def _invariants(self) -> PredictionRecord:
        ranked = self.ranked_predictions
        if self.predicted_class != ranked[0].label:
            raise ValueError("predicted_class must equal ranked_predictions[0].label")
        if self.confidence != ranked[0].score:
            raise ValueError("confidence must equal ranked_predictions[0].score")
        scores = [p.score for p in ranked]
        if any(a < b for a, b in zip(scores, scores[1:], strict=False)):
            raise ValueError("ranked_predictions must be sorted by score, descending")
        if len(ranked) != len(self.candidate_labels) or sorted(p.label for p in ranked) != sorted(
            self.candidate_labels
        ):
            raise ValueError(
                "ranked_predictions must contain exactly one entry per candidate label"
            )
        if not self.inference_parameters.multi_label and not math.isclose(
            sum(scores), 1.0, abs_tol=_SUM_TOLERANCE
        ):
            raise ValueError("single-label scores must sum to 1")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_json_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")


class OutcomeStatus(StrEnum):
    FAILED = "failed"
    COMPUTED_NOT_PERSISTED = "computed_not_persisted"
    SUCCEEDED = "succeeded"


class PredictionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    status: OutcomeStatus
    request_id: str
    result: ClassificationResult | None = None
    record: PredictionRecord | None = None
    object_key: str | None = None
    error: ZeroShotError | None = None
