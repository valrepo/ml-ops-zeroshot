"""Candidate-label sets (data-model.md §4)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from zeroshot.errors import LabelSetValidationError

_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
MAX_LABEL_CHARS = 100
MAX_NAME_CHARS = 80


def normalize_labels(
    labels: Sequence[str], *, error_cls: type = LabelSetValidationError
) -> list[str]:
    """Trim labels and enforce: non-empty, <=100 chars, >=2, unique under ``casefold()``.

    Order and the user's text (after trimming) are preserved. ``error_cls`` lets the
    request validator reuse the same rules with ``InputValidationError``.
    """
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for position, raw in enumerate(labels, start=1):
        label = (raw or "").strip()
        if not label:
            raise error_cls(f"Label #{position} is empty.", code="empty_label")
        if len(label) > MAX_LABEL_CHARS and error_cls is LabelSetValidationError:
            raise error_cls(
                f"Label #{position} is longer than {MAX_LABEL_CHARS} characters.",
                code="label_too_long",
            )
        folded = label.casefold()
        if folded in seen:
            raise error_cls(
                f"Duplicate label {label!r} (same as label #{seen[folded]}).",
                code="duplicate_label",
            )
        seen[folded] = position
        cleaned.append(label)
    if len(cleaned) < 2:
        raise error_cls(
            "At least two distinct candidate labels are required.", code="too_few_labels"
        )
    return cleaned


class LabelSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    name: str
    labels: list[str]

    @field_validator("key", mode="before")
    @classmethod
    def _key(cls, v: object) -> str:
        key = str(v or "").strip()
        if not _KEY.fullmatch(key):
            raise LabelSetValidationError(
                "Key must be lowercase letters, digits, '-' or '_' (max 63 characters).",
                code="invalid_key",
            )
        return key

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v: object) -> str:
        name = str(v or "").strip()
        if not 1 <= len(name) <= MAX_NAME_CHARS:
            raise LabelSetValidationError(
                f"Name must be 1–{MAX_NAME_CHARS} characters.", code="invalid_name"
            )
        return name

    @field_validator("labels", mode="before")
    @classmethod
    def _labels(cls, v: object) -> list[str]:
        if not isinstance(v, (list, tuple)):
            raise LabelSetValidationError("Labels must be a list.", code="label_set_invalid")
        return normalize_labels([str(x) if x is not None else "" for x in v])
