"""YAML-backed candidate-label sets (research R9, contracts/configuration.md §3).

* Seed file (``configs/labels.yaml``) – tracked in git, never written by the app.
* Runtime file (``LABEL_SETS_FILE``, default ``data/label_sets.yaml``) – created from the
  seed on first start; holds the administrator's edits.

Invalid entries in the runtime file are skipped and reported via ``load_warnings()``;
valid entries stay usable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path

import yaml
from filelock import FileLock
from pydantic import ValidationError

from zeroshot.domain.labels import LabelSet
from zeroshot.errors import ConfigError, LabelSetError, UnknownLabelSetError


class LabelSetService:
    def __init__(self, runtime_file: Path, seed_file: Path) -> None:
        self._file = Path(runtime_file)
        self._seed = Path(seed_file)
        self._lock = threading.Lock()
        self._sets: dict[str, LabelSet] = {}
        self._warnings: list[str] = []
        if not self._file.exists():
            if not self._seed.exists():
                raise ConfigError(
                    f"Label-set seed file not found: {self._seed}", code="config_missing"
                )
            self._file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._seed, self._file)
        self._reload()

    # -- reading ------------------------------------------------------------------------
    def _reload(self) -> None:
        try:
            data = yaml.safe_load(self._file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Label-set file {self._file} is not valid YAML: {exc}", code="config_invalid"
            ) from None
        entries = data.get("label_sets", []) if isinstance(data, dict) else []
        sets: dict[str, LabelSet] = {}
        warnings: list[str] = []
        for index, raw in enumerate(entries if isinstance(entries, list) else [], start=1):
            try:
                label_set = LabelSet.model_validate(raw)
            except (LabelSetError, ValidationError) as exc:
                message = exc.user_message if isinstance(exc, LabelSetError) else str(exc)
                warnings.append(f"Label set #{index} skipped: {message}")
                continue
            if label_set.key in sets:
                warnings.append(f"Label set #{index} skipped: duplicate key '{label_set.key}'")
                continue
            sets[label_set.key] = label_set
        self._sets, self._warnings = sets, warnings

    def list(self) -> list[LabelSet]:
        with self._lock:
            return list(self._sets.values())

    def get(self, key: str) -> LabelSet:
        with self._lock:
            try:
                return self._sets[key]
            except KeyError:
                raise UnknownLabelSetError(f"Unknown label set '{key}'.") from None

    def load_warnings(self) -> list[str]:
        with self._lock:
            return list(self._warnings)

    # -- writing (research R9: file lock + atomic replace) --------------------------------
    def save(self, label_set: LabelSet) -> LabelSet:
        """Create or replace a label set. ``LabelSet`` itself has already validated it."""
        with self._lock, FileLock(f"{self._file}.lock"):
            sets = dict(self._sets)
            sets[label_set.key] = label_set
            self._write(list(sets.values()))
            self._sets = sets
        return label_set

    def delete(self, key: str) -> None:
        with self._lock, FileLock(f"{self._file}.lock"):
            if key not in self._sets:
                raise UnknownLabelSetError(f"Unknown label set '{key}'.")
            sets = {k: v for k, v in self._sets.items() if k != key}
            self._write(list(sets.values()))
            self._sets = sets

    def _write(self, sets: list[LabelSet]) -> None:
        payload = {"label_sets": [s.model_dump() for s in sets]}
        fd, tmp = tempfile.mkstemp(dir=self._file.parent, prefix=".label_sets.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
            os.replace(tmp, self._file)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
