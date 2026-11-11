"""Shared structured logging (research R10). Contains no business logic.

Usage::

    log = logging.getLogger(__name__)
    log_event(log, "prediction.started", request_id=rid, **text_fingerprint(text))

Never pass the submitted text, tokens or credentials as fields. A redaction filter masks
fields whose name looks secret anyway, as a second line of defence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

_SECRET_KEY = re.compile(r"(?i)token|secret|password|access_key|secret_key")
_FIELDS_ATTR = "zs_fields"


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    return dict(getattr(record, _FIELDS_ATTR, {}))


class RedactingFilter(logging.Filter):
    """Replace the value of any structured field whose key looks like a secret."""

    def filter(self, record: logging.LogRecord) -> bool:
        fields = getattr(record, _FIELDS_ATTR, None)
        if fields:
            setattr(
                record,
                _FIELDS_ATTR,
                {k: ("***" if _SECRET_KEY.search(k) else v) for k, v in fields.items()},
            )
        return True


def _ts(record: logging.LogRecord) -> str:
    return (
        datetime.fromtimestamp(record.created, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _ts(record),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(_fields(record))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(f"{k}={v}" for k, v in _fields(record).items())
        line = (
            f"{_ts(record)} {record.levelname:<7} {getattr(record, 'event', record.getMessage())}"
        )
        if fields:
            line = f"{line} {fields}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the root ``zeroshot`` logger once (idempotent)."""
    logger = logging.getLogger("zeroshot")
    logger.setLevel(level.upper())
    logger.propagate = True
    for handler in list(logger.handlers):
        if getattr(handler, "_zeroshot", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    handler.addFilter(RedactingFilter())
    handler._zeroshot = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    *,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Emit one structured event. ``fields`` must never contain raw input text or secrets."""
    clean = {k: ("***" if _SECRET_KEY.search(k) else v) for k, v in fields.items()}
    logger.log(level, event, extra={"event": event, _FIELDS_ATTR: clean}, exc_info=exc_info)


def text_fingerprint(text: str) -> dict[str, Any]:
    """Loggable, non-reversible description of submitted text (Principle X)."""
    return {
        "input_chars": len(text),
        "input_sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
    "RedactingFilter",
    "log_event",
    "setup_logging",
    "text_fingerprint",
]
