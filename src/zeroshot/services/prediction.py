"""Prediction workflow orchestration (FR-12): the only path from the UI to inference.

validate -> lease model -> classify -> build record -> validate record -> persist.
Always returns exactly one ``PredictionOutcome``; never raises.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from zeroshot.config import Settings
from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.core.validation import validate_request
from zeroshot.domain.models import ModelStatus
from zeroshot.domain.predictions import (
    InferenceParameters,
    OutcomeStatus,
    PredictionOutcome,
    PredictionRecord,
)
from zeroshot.errors import InternalError, ModelNotReadyError, StorageError, ZeroShotError
from zeroshot.logging import log_event, text_fingerprint
from zeroshot.services.classifier import Classifier
from zeroshot.services.label_sets import LabelSetService
from zeroshot.services.model_manager import ModelManager
from zeroshot.services.storage import PredictionStore

log = logging.getLogger(__name__)


class PredictionService:
    def __init__(
        self,
        settings: Settings,
        model_manager: ModelManager,
        label_sets: LabelSetService,
        classifier: Classifier,
        storage: PredictionStore,
        registry: ModelUsageRegistry,
    ) -> None:
        self._settings = settings
        self._models = model_manager
        self._label_sets = label_sets
        self._classifier = classifier
        self._storage = storage
        self._registry = registry

    def predict(
        self, text: str, model_key: str, label_set_key: str, session_id: str | None = None
    ) -> PredictionOutcome:
        request_id = str(uuid.uuid4())
        ctx = {"request_id": request_id, "label_set": label_set_key}
        log_event(log, "prediction.started", model_key=model_key, **ctx, **text_fingerprint(text))
        try:
            return self._run(request_id, text, model_key, label_set_key, session_id, ctx)
        except ZeroShotError as exc:
            log_event(
                log,
                "prediction.failed",
                logging.WARNING,
                code=exc.code,
                outcome=OutcomeStatus.FAILED.value,
                **ctx,
            )
            return PredictionOutcome(status=OutcomeStatus.FAILED, request_id=request_id, error=exc)
        except Exception:
            log_event(
                log,
                "prediction.failed",
                logging.ERROR,
                exc_info=True,
                code="internal_error",
                outcome=OutcomeStatus.FAILED.value,
                **ctx,
            )
            return PredictionOutcome(
                status=OutcomeStatus.FAILED,
                request_id=request_id,
                error=InternalError(
                    f"An unexpected error occurred (request {request_id}); see the logs."
                ),
            )

    def _run(
        self,
        request_id: str,
        text: str,
        model_key: str,
        label_set_key: str,
        session_id: str | None,
        ctx: dict[str, str],
    ) -> PredictionOutcome:
        label_set = self._label_sets.get(label_set_key)
        validate_request(text, label_set.labels, self._settings.max_input_chars)
        model = self._models.get(model_key)
        if model.status is not ModelStatus.READY:
            raise ModelNotReadyError(
                f"The model '{model.definition.name}' is not downloaded. An administrator must "
                "download it on the Administration page."
            )
        ctx |= {"model": model.definition.repo_id, "revision": model.definition.revision}
        defaults = model.definition.inference_defaults
        params = InferenceParameters(
            multi_label=defaults.multi_label,
            hypothesis_template=defaults.hypothesis_template,
            device=self._settings.inference_device,
        )
        with self._registry.lease(model_key):
            model = self._models.get(model_key)  # re-read under the lease
            result = self._classifier.classify(text, label_set.labels, model, params)
        log_event(
            log,
            "prediction.inferred",
            duration_ms=result.latency_ms,
            predicted_class=result.predicted_class,
            **ctx,
        )

        record = PredictionRecord(
            request_id=request_id,
            timestamp=datetime.now(UTC),
            input_text=text,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            candidate_labels=list(label_set.labels),
            ranked_predictions=result.ranked,
            model_name=model.definition.name,
            model=result.model_id,
            model_revision=result.model_revision,
            inference_parameters=result.inference_parameters,
            label_set=label_set.key,
            application_version=self._settings.app_version,
            inference_latency_ms=result.latency_ms,
            session_id=session_id,
        )
        try:
            key = self._storage.save(record)
        except StorageError as exc:
            log_event(
                log,
                "prediction.persist_failed",
                logging.ERROR,
                code=exc.code,
                outcome=OutcomeStatus.COMPUTED_NOT_PERSISTED.value,
                **ctx,
            )
            return PredictionOutcome(
                status=OutcomeStatus.COMPUTED_NOT_PERSISTED,
                request_id=request_id,
                result=result,
                record=record,
                error=exc,
            )
        log_event(
            log,
            "prediction.persisted",
            object_key=key,
            outcome=OutcomeStatus.SUCCEEDED.value,
            **ctx,
        )
        return PredictionOutcome(
            status=OutcomeStatus.SUCCEEDED,
            request_id=request_id,
            result=result,
            record=record,
            object_key=key,
        )
