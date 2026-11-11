"""Model catalogue, downloads, local availability, deletion and HF cache cleanup.

Storage layout (research R6/R7)::

    MODEL_DIR/<owner>--<name>/<revision>/      # authoritative, app-managed copy
        .complete.json                          # written last; the only proof of "ready"
    MODEL_DIR/.staging/<slug>--<rev>--<uuid>/   # in-progress downloads (never "ready")
    HF_HOME/{hub,xet}/                          # library caches: target of cache cleanup

Never runs inference and never writes predictions (constitution Principle II).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from zeroshot.config import Settings
from zeroshot.core.paths import safe_child
from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.domain.models import (
    CacheCleanupReport,
    CompletionMarker,
    DownloadOutcome,
    ModelDefinition,
    ModelInstallation,
    ModelStatus,
    StoragePaths,
)
from zeroshot.errors import (
    CacheCleanupError,
    CacheInUseError,
    ModelDeleteError,
    ModelDownloadError,
    ModelInUseError,
    ModelNotReadyError,
    PathSafetyError,
    UnknownModelError,
    ZeroShotError,
)
from zeroshot.logging import log_event

log = logging.getLogger(__name__)

MARKER_NAME = ".complete.json"
STAGING_DIR = ".staging"
HF_CACHE_SUBDIRS = ("hub", "xet")

ArtifactValidator = Callable[[Path], list[str]]


class ModelManager:
    def __init__(
        self,
        settings: Settings,
        catalogue: Sequence[ModelDefinition],
        registry: ModelUsageRegistry,
        hub: ModuleType | Any | None = None,
        artifact_validator: ArtifactValidator | None = None,
    ) -> None:
        self._settings = settings
        self._catalogue = {d.key: d for d in catalogue}
        self._registry = registry
        self._hub = hub
        self._validate_artifacts = artifact_validator or validate_artifacts
        self._model_dir = Path(settings.model_dir).resolve()
        self._hf_home = Path(settings.hf_home).resolve()
        self._state: dict[str, tuple[ModelStatus, str | None]] = {}
        self._state_lock = threading.Lock()
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_staging()

    # -- lookup -------------------------------------------------------------------------
    def _definition(self, model_key: str) -> ModelDefinition:
        try:
            return self._catalogue[model_key]
        except KeyError:
            raise UnknownModelError(f"Unknown model '{model_key}'.") from None

    def _install_dir(self, definition: ModelDefinition) -> Path:
        return safe_child(self._model_dir, definition.slug, definition.revision)

    def _read_marker(
        self, definition: ModelDefinition, install_dir: Path
    ) -> CompletionMarker | None:
        marker_path = install_dir / MARKER_NAME
        try:
            marker = CompletionMarker.model_validate_json(marker_path.read_text(encoding="utf-8"))
        except OSError, ValidationError, json.JSONDecodeError, ValueError:
            return None
        if marker.model_id != definition.repo_id or marker.revision != definition.revision:
            return None
        if not all((install_dir / f).is_file() for f in marker.files):
            return None
        return marker

    def _installation(self, definition: ModelDefinition) -> ModelInstallation:
        in_use = self._registry.is_in_use(definition.key)
        with self._state_lock:
            transient = self._state.get(definition.key)
        if transient and transient[0] is ModelStatus.DOWNLOADING:
            return ModelInstallation(
                definition=definition, status=ModelStatus.DOWNLOADING, in_use=in_use
            )
        try:
            install_dir = self._install_dir(definition)
        except PathSafetyError as exc:
            return ModelInstallation(
                definition=definition, status=ModelStatus.FAILED, error=exc.user_message
            )
        if install_dir.is_dir():
            marker = self._read_marker(definition, install_dir)
            if marker:
                return ModelInstallation(
                    definition=definition,
                    status=ModelStatus.READY,
                    local_path=install_dir,
                    marker=marker,
                    in_use=in_use,
                )
            return ModelInstallation(
                definition=definition, status=ModelStatus.INCOMPLETE, in_use=in_use
            )
        if transient and transient[0] is ModelStatus.FAILED:
            return ModelInstallation(
                definition=definition, status=ModelStatus.FAILED, error=transient[1]
            )
        return ModelInstallation(definition=definition, status=ModelStatus.NOT_AVAILABLE)

    # -- public read API -------------------------------------------------------------------
    def list_models(self) -> list[ModelInstallation]:
        return [self._installation(d) for d in self._catalogue.values()]

    def get(self, model_key: str) -> ModelInstallation:
        return self._installation(self._definition(model_key))

    def ready_models(self) -> list[ModelInstallation]:
        return [m for m in self.list_models() if m.status is ModelStatus.READY]

    def paths(self) -> StoragePaths:
        return StoragePaths(
            model_dir=self._model_dir,
            hf_home=self._hf_home,
            hf_cache_targets=[self._hf_home / name for name in HF_CACHE_SUBDIRS],
        )

    # -- downloads (research R5, R7) -------------------------------------------------------
    @property
    def _staging_root(self) -> Path:
        return self._model_dir / STAGING_DIR

    def _cleanup_stale_staging(self) -> None:
        if not self._staging_root.is_dir():
            return
        cutoff = time.time() - STALE_STAGING_SECONDS
        for entry in self._staging_root.iterdir():
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue

    def _set_state(self, key: str, status: ModelStatus | None, error: str | None = None) -> None:
        with self._state_lock:
            if status is None:
                self._state.pop(key, None)
            else:
                self._state[key] = (status, error)

    def download(
        self,
        model_keys: Sequence[str],
        on_progress: Callable[[str, str], None] | None = None,
    ) -> dict[str, DownloadOutcome]:
        """Download each model sequentially; one failure never affects the others."""
        report = on_progress or (lambda _key, _status: None)
        outcomes: dict[str, DownloadOutcome] = {}
        for key in model_keys:
            try:
                definition = self._definition(key)
            except UnknownModelError as exc:
                outcomes[key] = DownloadOutcome(status="failed", error=exc.user_message)
                report(key, "failed")
                continue
            current = self._installation(definition)
            if current.status is ModelStatus.READY:
                log_event(
                    log,
                    "model.download.skipped",
                    model=definition.repo_id,
                    revision=definition.revision,
                    outcome="already_present",
                )
                outcomes[key] = DownloadOutcome(status="already_present", marker=current.marker)
                report(key, "already_present")
                continue
            report(key, "started")
            try:
                marker = self._download_one(definition)
            except ZeroShotError as exc:
                self._set_state(key, ModelStatus.FAILED, exc.user_message)
                outcomes[key] = DownloadOutcome(status="failed", error=exc.user_message)
                report(key, "failed")
            else:
                self._set_state(key, None)
                outcomes[key] = DownloadOutcome(status="downloaded", marker=marker)
                report(key, "downloaded")
        return outcomes

    def _download_one(self, definition: ModelDefinition) -> CompletionMarker:
        final_dir = self._install_dir(definition)
        staging = safe_child(
            self._model_dir, STAGING_DIR, f"{definition.slug}--{definition.revision}--{uuid4().hex}"
        )
        started = time.perf_counter()
        log_event(
            log, "model.download.started", model=definition.repo_id, revision=definition.revision
        )
        try:
            with self._registry.downloading(definition.key):
                self._set_state(definition.key, ModelStatus.DOWNLOADING)
                staging.mkdir(parents=True)
                self._fetch(definition, staging)
                # snapshot_download(local_dir=...) keeps its metadata here; not model files.
                shutil.rmtree(staging / ".cache", ignore_errors=True)
                problems = self._validate_artifacts(staging)
                if problems:
                    raise ModelDownloadError(
                        "The downloaded files are not a usable zero-shot model: "
                        + "; ".join(problems),
                        code="validation_failed",
                    )
                marker = _write_marker(definition, staging)
                self._promote(staging, final_dir)
        except ZeroShotError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            log_event(
                log,
                "model.download.failed",
                logging.WARNING,
                model=definition.repo_id,
                revision=definition.revision,
                code=exc.code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to a user-safe error below
            shutil.rmtree(staging, ignore_errors=True)
            error = _map_download_error(exc)
            log_event(
                log,
                "model.download.failed",
                logging.WARNING,
                exc_info=True,
                model=definition.repo_id,
                revision=definition.revision,
                code=error.code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise error from None
        log_event(
            log,
            "model.download.succeeded",
            model=definition.repo_id,
            revision=definition.revision,
            total_bytes=marker.total_bytes,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return marker

    def _fetch(self, definition: ModelDefinition, staging: Path) -> None:
        hub = self._hub
        if hub is None:
            import huggingface_hub as hub  # heavy; imported lazily
        token = self._settings.hf_token.get_secret_value() if self._settings.hf_token else None
        hub.snapshot_download(
            repo_id=definition.repo_id,
            revision=definition.revision,
            local_dir=staging,
            allow_patterns=list(definition.download.allow_patterns),
            ignore_patterns=list(definition.download.ignore_patterns),
            token=token,
        )

    # -- deletion (FR-05, US-04) ---------------------------------------------------------
    def delete(self, model_key: str) -> None:
        """Delete one managed installation. Refused while the model is in use."""
        definition = self._definition(model_key)
        install_dir = self._install_dir(definition)  # PathSafetyError if it escapes MODEL_DIR
        if not install_dir.is_dir():
            raise ModelNotReadyError(f"'{definition.name}' is not downloaded; nothing to delete.")
        try:
            with self._registry.exclusive(model_key):
                shutil.rmtree(install_dir)
                parent = install_dir.parent
                if parent != self._model_dir and not any(parent.iterdir()):
                    parent.rmdir()
        except ModelInUseError as exc:
            log_event(
                log,
                "model.delete.refused",
                logging.WARNING,
                model=definition.repo_id,
                code=exc.code,
            )
            raise
        except OSError as exc:
            log_event(
                log,
                "model.delete.refused",
                logging.ERROR,
                exc_info=True,
                model=definition.repo_id,
                code="delete_failed",
            )
            raise ModelDeleteError(
                f"Could not delete '{definition.name}' ({exc.strerror or type(exc).__name__})."
            ) from None
        self._set_state(model_key, None)
        self._registry.notify_removed(model_key)  # classifier evicts its cached pipeline
        log_event(
            log,
            "model.deleted",
            model=definition.repo_id,
            revision=definition.revision,
            path=str(install_dir),
        )

    # -- HF cache cleanup (FR-05, US-05) ---------------------------------------------------
    def clear_hf_cache(self) -> CacheCleanupReport:
        """Empty HF_HOME/hub and HF_HOME/xet. Never touches MODEL_DIR or HF_HOME/token."""
        targets = self.paths().hf_cache_targets
        removed = 0
        found = False
        try:
            with self._registry.global_exclusive():
                for name in HF_CACHE_SUBDIRS:
                    target = self._hf_home / name
                    if not target.is_dir() or target.is_symlink():
                        continue
                    safe_child(self._hf_home, name)
                    for child in list(target.iterdir()):
                        found = True
                        if child.is_symlink() or not child.is_dir():
                            removed += child.lstat().st_size
                            child.unlink()
                        else:
                            removed += _tree_size(child)
                            shutil.rmtree(child)
        except CacheInUseError:
            raise
        except (OSError, PathSafetyError) as exc:
            log_event(
                log,
                "cache.cleanup.failed",
                logging.ERROR,
                exc_info=True,
                hf_home=str(self._hf_home),
            )
            message = getattr(exc, "strerror", None) or type(exc).__name__
            raise CacheCleanupError(f"Hugging Face cache cleanup failed ({message}).") from None
        status = "cleared" if found else "nothing_to_clear"
        log_event(
            log, "cache.cleared", hf_home=str(self._hf_home), removed_bytes=removed, outcome=status
        )
        return CacheCleanupReport(status=status, removed_bytes=removed, targets=targets)

    def _promote(self, staging: Path, final_dir: Path) -> None:
        """Atomically move a validated staging directory to its final location."""
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        old = None
        if final_dir.exists():  # an INCOMPLETE installation: move it aside first
            old = staging.with_name(staging.name + "-old")
            os.replace(final_dir, old)
        os.replace(staging, final_dir)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)


STALE_STAGING_SECONDS = 3600


def _tree_size(root: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                total += (Path(dirpath) / name).lstat().st_size
            except OSError:
                continue
    return total


def _write_marker(definition: ModelDefinition, directory: Path) -> CompletionMarker:
    files = sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())
    marker = CompletionMarker(
        model_id=definition.repo_id,
        revision=definition.revision,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        files=files,
        total_bytes=sum((directory / f).stat().st_size for f in files),
    )
    (directory / MARKER_NAME).write_text(marker.model_dump_json(indent=2), encoding="utf-8")
    return marker


def _map_download_error(exc: Exception) -> ModelDownloadError:
    """Translate library errors to user-safe messages (never includes tokens or URLs)."""
    from huggingface_hub import errors as hub_errors

    if isinstance(exc, getattr(hub_errors, "RepositoryNotFoundError", ())):
        return ModelDownloadError(
            "The model repository was not found or requires access (check HF_TOKEN).",
            code="repo_not_found",
        )
    if isinstance(exc, getattr(hub_errors, "RevisionNotFoundError", ())):
        return ModelDownloadError(
            "The configured revision does not exist in the model repository.",
            code="revision_not_found",
        )
    if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "LocalEntryNotFoundError",
    }:
        return ModelDownloadError(
            "Network error while downloading the model; check your connection and retry.",
            code="network_error",
        )
    if isinstance(exc, getattr(hub_errors, "HfHubHTTPError", ())):
        return ModelDownloadError(
            "Hugging Face returned an error while downloading; retry later.",
            code="download_failed",
        )
    if isinstance(exc, OSError):
        return ModelDownloadError(
            f"Could not write model files ({exc.strerror or type(exc).__name__}); "
            "check free disk space and permissions.",
            code="disk_error",
        )
    return ModelDownloadError("The model download failed unexpectedly.", code="download_failed")


def validate_artifacts(path: Path) -> list[str]:
    """Check a downloaded model without running inference (research R7 step 3)."""
    problems: list[str] = []
    if not (path / "config.json").is_file():
        problems.append("config.json missing")
    if not list(path.glob("*.safetensors")):
        problems.append("no *.safetensors weights")
    if not any((path / n).is_file() for n in ("tokenizer.json", "spm.model", "vocab.json")):
        problems.append("no tokenizer files")
    if problems:
        return problems
    from transformers import AutoConfig, AutoTokenizer  # heavy; imported lazily

    try:
        config = AutoConfig.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any loader failure means "invalid"
        return [f"config.json could not be loaded: {type(exc).__name__}"]
    architectures = getattr(config, "architectures", None) or []
    if not any(a.endswith("ForSequenceClassification") for a in architectures):
        problems.append(f"unsupported architecture {architectures!r}")
    labels = {str(v).lower() for v in (getattr(config, "id2label", None) or {}).values()}
    if not any(label.startswith("entail") for label in labels):
        problems.append("id2label has no 'entailment' label (not an NLI zero-shot model)")
    try:
        AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"tokenizer could not be loaded: {type(exc).__name__}")
    return problems
