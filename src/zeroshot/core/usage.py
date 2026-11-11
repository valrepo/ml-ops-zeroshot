"""In-process coordination of model use (research R8).

The model manager and the classifier both depend on this registry and never on each other
(constitution Principle II). It protects a single application process; running several
app processes against one ``MODEL_DIR`` is unsupported and documented in the README.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from zeroshot.errors import CacheInUseError, ModelInUseError


class ModelUsageRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._leases: Counter[str] = Counter()
        self._downloads: Counter[str] = Counter()
        self._exclusive: set[str] = set()
        self._global_exclusive = False
        self._removed_callbacks: list[Callable[[str], None]] = []

    # -- inference ----------------------------------------------------------------------
    @contextmanager
    def lease(self, model_key: str) -> Iterator[None]:
        """Mark ``model_key`` as in use for the duration of the block."""
        with self._lock:
            if self._global_exclusive:
                raise CacheInUseError("The model cache is being cleaned up; try again shortly.")
            if model_key in self._exclusive:
                raise ModelInUseError(
                    "This model is being deleted and cannot be used.", code="model_being_deleted"
                )
            self._leases[model_key] += 1
        try:
            yield
        finally:
            with self._lock:
                self._leases[model_key] -= 1
                if self._leases[model_key] <= 0:
                    del self._leases[model_key]

    # -- downloads ----------------------------------------------------------------------
    @contextmanager
    def downloading(self, model_key: str) -> Iterator[None]:
        with self._lock:
            if self._global_exclusive:
                raise CacheInUseError("The model cache is being cleaned up; try again shortly.")
            self._downloads[model_key] += 1
        try:
            yield
        finally:
            with self._lock:
                self._downloads[model_key] -= 1
                if self._downloads[model_key] <= 0:
                    del self._downloads[model_key]

    # -- destructive operations ------------------------------------------------------------
    @contextmanager
    def exclusive(self, model_key: str) -> Iterator[None]:
        """Exclusive access for deleting one model; refused while it is in use."""
        with self._lock:
            if self._leases.get(model_key, 0) > 0:
                raise ModelInUseError(
                    "This model is currently used by a classification; try again when it finishes."
                )
            if self._downloads.get(model_key, 0) > 0 or model_key in self._exclusive:
                raise ModelInUseError("This model is being downloaded or deleted right now.")
            self._exclusive.add(model_key)
        try:
            yield
        finally:
            with self._lock:
                self._exclusive.discard(model_key)

    @contextmanager
    def global_exclusive(self) -> Iterator[None]:
        """Exclusive access for cache cleanup; refused while anything is active."""
        with self._lock:
            if self._global_exclusive or self._exclusive or self._active_locked():
                raise CacheInUseError(
                    "Cache cleanup is not possible while a classification or download is running."
                )
            self._global_exclusive = True
        try:
            yield
        finally:
            with self._lock:
                self._global_exclusive = False

    # -- queries ------------------------------------------------------------------------
    def is_in_use(self, model_key: str) -> bool:
        with self._lock:
            return self._leases.get(model_key, 0) > 0

    def any_active(self) -> bool:
        with self._lock:
            return self._active_locked()

    def _active_locked(self) -> bool:
        return bool(+self._leases) or bool(+self._downloads)

    # -- eviction notifications ----------------------------------------------------------
    def on_removed(self, callback: Callable[[str], None]) -> None:
        self._removed_callbacks.append(callback)

    def notify_removed(self, model_key: str) -> None:
        for callback in list(self._removed_callbacks):
            callback(model_key)
