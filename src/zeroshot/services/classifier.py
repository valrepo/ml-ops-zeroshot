"""Zero-shot inference with Hugging Face Transformers on PyTorch (FR-07, research R8/R13/R14).

* Never downloads models and never touches MinIO (Principle II).
* The compute device comes from ``INFERENCE_DEVICE`` and is never auto-detected.
* Loaded pipelines are kept in a small LRU (``MAX_LOADED_MODELS``, default 1) and evicted
  when the model manager reports a deletion through the shared usage registry.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from zeroshot.config import Settings
from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.core.validation import validate_request
from zeroshot.domain.models import ModelInstallation, ModelStatus
from zeroshot.domain.predictions import ClassificationResult, InferenceParameters, LabelScore
from zeroshot.errors import InferenceError, ModelLoadError, ModelNotReadyError
from zeroshot.logging import log_event

log = logging.getLogger(__name__)


class ZeroShotPipeline(Protocol):
    def __call__(
        self, text: str, candidate_labels: list[str], hypothesis_template: str, multi_label: bool
    ) -> dict[str, Any]: ...


PipelineFactory = Callable[[Path, str], ZeroShotPipeline]


def transformers_pipeline_factory(path: Path, device: str) -> ZeroShotPipeline:
    from transformers import pipeline  # heavy; imported lazily

    return pipeline("zero-shot-classification", model=str(path), device=device)


class Classifier:
    def __init__(
        self,
        settings: Settings,
        registry: ModelUsageRegistry,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._settings = settings
        self._device = settings.inference_device
        self._factory = pipeline_factory or transformers_pipeline_factory
        self._max_loaded = settings.max_loaded_models
        self._pipelines: OrderedDict[str, tuple[str, ZeroShotPipeline]] = OrderedDict()
        self._lock = threading.Lock()
        registry.on_removed(self.evict)

    # -- pipeline cache ------------------------------------------------------------------
    def _pipeline_for(self, model: ModelInstallation) -> ZeroShotPipeline:
        key = model.key
        revision = model.marker.revision if model.marker else ""
        with self._lock:
            cached = self._pipelines.get(key)
            if cached and cached[0] == revision:
                self._pipelines.move_to_end(key)
                return cached[1]
            while len(self._pipelines) >= self._max_loaded:
                evicted, _ = self._pipelines.popitem(last=False)
                log_event(log, "pipeline.evicted", model_key=evicted, reason="lru")
            started = time.perf_counter()
            try:
                pipe = self._factory(model.local_path, self._device)
            except Exception:
                log_event(
                    log,
                    "pipeline.load_failed",
                    logging.ERROR,
                    exc_info=True,
                    model=model.definition.repo_id,
                    revision=revision,
                    device=self._device,
                )
                raise ModelLoadError(
                    f"The model '{model.definition.name}' could not be loaded. Check the logs, "
                    "or delete and re-download it on the Administration page."
                ) from None
            self._pipelines[key] = (revision, pipe)
            log_event(
                log,
                "pipeline.loaded",
                model=model.definition.repo_id,
                revision=revision,
                device=self._device,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return pipe

    def evict(self, model_key: str) -> None:
        with self._lock:
            if self._pipelines.pop(model_key, None) is not None:
                log_event(log, "pipeline.evicted", model_key=model_key, reason="removed")

    # -- inference ------------------------------------------------------------------------
    def classify(
        self,
        text: str,
        labels: Sequence[str],
        model: ModelInstallation,
        params: InferenceParameters,
    ) -> ClassificationResult:
        candidates = validate_request(text, labels, self._settings.max_input_chars)
        if model.status is not ModelStatus.READY or model.local_path is None or not model.marker:
            raise ModelNotReadyError(
                f"The model '{model.definition.name}' is not downloaded. An administrator must "
                "download it on the Administration page."
            )
        effective = params.model_copy(update={"device": self._device})
        pipe = self._pipeline_for(model)
        started = time.perf_counter()
        try:
            output = pipe(
                text,
                candidate_labels=candidates,
                hypothesis_template=effective.hypothesis_template,
                multi_label=effective.multi_label,
            )
        except Exception:
            log_event(
                log,
                "prediction.failed",
                logging.ERROR,
                exc_info=True,
                model=model.definition.repo_id,
                stage="inference",
            )
            raise InferenceError("Classification failed while running the model.") from None
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = _rank(candidates, output)
        return ClassificationResult(
            ranked=ranked,
            model_id=model.definition.repo_id,
            model_revision=model.marker.revision,
            inference_parameters=effective,
            latency_ms=round(latency_ms, 1),
        )


def _rank(candidates: list[str], output: dict[str, Any]) -> list[LabelScore]:
    """Pair labels with scores and sort by (-score, candidate position) — deterministic ties."""
    scores = dict(zip(output.get("labels", []), output.get("scores", []), strict=False))
    if set(scores) != set(candidates):
        raise InferenceError(
            "The model did not return a score for every label.", code="invalid_scores"
        )
    position = {label: i for i, label in enumerate(candidates)}
    pairs = []
    for label in candidates:
        score = float(scores[label])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise InferenceError("The model returned an invalid score.", code="invalid_scores")
        pairs.append((label, score))
    pairs.sort(key=lambda p: (-p[1], position[p[0]]))
    return [LabelScore(label=label, score=score) for label, score in pairs]
