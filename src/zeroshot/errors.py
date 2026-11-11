"""Typed domain exceptions (contracts/services.md, "Errors" and "Error codes").

Every expected operational failure is a ``ZeroShotError`` carrying:

* ``user_message`` – safe to show in the UI (no secrets, no stack traces);
* ``code`` – a stable machine-readable identifier, asserted in tests.

Each class declares the complete set of codes it may carry (``CODES``); the first entry is
the default. Passing an unknown code raises ``ValueError`` so typos fail in tests.
"""

from __future__ import annotations

from typing import ClassVar


class ZeroShotError(Exception):
    CODES: ClassVar[tuple[str, ...]] = ("error",)

    def __init__(self, user_message: str, *, code: str | None = None) -> None:
        code = code or self.CODES[0]
        if code not in self.CODES:
            raise ValueError(f"{type(self).__name__} does not allow code {code!r}")
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code

    def __str__(self) -> str:
        return self.user_message


class ConfigError(ZeroShotError):
    CODES = (
        "config_invalid",
        "config_missing",
        "invalid_bucket_name",
        "catalogue_invalid",
        "device_unavailable",
    )


class InputValidationError(ZeroShotError):
    CODES = ("empty_text", "text_too_long", "too_few_labels", "empty_label", "duplicate_label")


class LabelSetError(ZeroShotError):
    CODES = ("label_set_error",)


class UnknownLabelSetError(LabelSetError):
    CODES = ("unknown_label_set",)


class LabelSetValidationError(LabelSetError):
    CODES = (
        "label_set_invalid",
        "empty_label",
        "duplicate_label",
        "too_few_labels",
        "label_too_long",
        "invalid_key",
        "invalid_name",
    )


class ModelError(ZeroShotError):
    CODES = ("model_error",)


class UnknownModelError(ModelError):
    CODES = ("unknown_model",)


class ModelNotReadyError(ModelError):
    CODES = ("model_not_ready",)


class ModelInUseError(ModelError):
    CODES = ("model_in_use", "model_being_deleted")


class ModelDownloadError(ModelError):
    CODES = (
        "download_failed",
        "repo_not_found",
        "revision_not_found",
        "network_error",
        "disk_error",
        "validation_failed",
    )


class ModelLoadError(ModelError):
    CODES = ("model_load_failed",)


class ModelDeleteError(ModelError):
    CODES = ("delete_failed",)


class PathSafetyError(ZeroShotError):
    CODES = ("unsafe_path",)


class CacheError(ZeroShotError):
    CODES = ("cache_error",)


class CacheInUseError(CacheError):
    CODES = ("cache_in_use",)


class CacheCleanupError(CacheError):
    CODES = ("cache_cleanup_failed",)


class InferenceError(ZeroShotError):
    CODES = ("inference_failed", "invalid_scores")


class StorageError(ZeroShotError):
    CODES = ("storage_read_failed",)


class BucketInitError(StorageError):
    CODES = ("bucket_init_failed",)


class StorageUnavailableError(StorageError):
    CODES = ("storage_unavailable",)


class StorageWriteError(StorageError):
    CODES = ("storage_write_failed", "invalid_record")


class InternalError(ZeroShotError):
    CODES = ("internal_error",)
