"""Request validation — always before inference (FR-07, Principle VIII)."""

from __future__ import annotations

from collections.abc import Sequence

from zeroshot.domain.labels import normalize_labels
from zeroshot.errors import InputValidationError


def validate_request(text: str, labels: Sequence[str], max_chars: int) -> list[str]:
    """Validate input text and labels; return the normalized labels.

    The text itself is not modified (it is stored as submitted).
    """
    if not isinstance(text, str) or not text.strip():
        raise InputValidationError("Please enter some text to classify.", code="empty_text")
    if len(text) > max_chars:
        raise InputValidationError(
            f"Text exceeds the {max_chars}-character limit ({len(text)} characters).",
            code="text_too_long",
        )
    return normalize_labels(list(labels), error_cls=InputValidationError)
