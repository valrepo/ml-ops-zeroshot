"""Wire the services once per process (cached by the UI with ``st.cache_resource``)."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field

from zeroshot.config import Settings, load_model_catalogue, load_settings
from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.errors import ConfigError, StorageError
from zeroshot.logging import log_event, setup_logging
from zeroshot.services.classifier import Classifier
from zeroshot.services.label_sets import LabelSetService
from zeroshot.services.model_manager import ModelManager
from zeroshot.services.prediction import PredictionService
from zeroshot.services.storage import StorageService

log = logging.getLogger("zeroshot.bootstrap")


@dataclass
class Services:
    settings: Settings
    registry: ModelUsageRegistry
    label_sets: LabelSetService
    model_manager: ModelManager
    classifier: Classifier
    storage: StorageService
    prediction: PredictionService
    startup_errors: list[str] = field(default_factory=list)


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or load_settings()
    setup_logging(settings.log_level, settings.log_format)
    log_event(
        log,
        "app.startup",
        app_version=settings.app_version,
        python=platform.python_version(),
        inference_device=settings.inference_device,
    )
    log_event(
        log,
        "config.validated",
        logging.WARNING if settings.warnings else logging.INFO,
        warnings=settings.warnings,
    )
    startup_errors: list[str] = []
    registry = ModelUsageRegistry()
    label_sets = LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)

    # A broken model catalogue must not crash the app; it is reported on the admin page.
    try:
        catalogue = load_model_catalogue(settings.models_file)
    except ConfigError as exc:
        startup_errors.append(exc.user_message)
        catalogue = []
    model_manager = ModelManager(settings, catalogue, registry)
    classifier = Classifier(settings, registry)

    # The bucket is ensured at startup; a failure is reported (not fatal) and retried
    # before the first write, so the app recovers once MinIO is back.
    storage = StorageService(settings)
    try:
        storage.ensure_bucket()
    except StorageError as exc:
        startup_errors.append(f"Storage initialization failed: {exc.user_message}")

    return Services(
        settings=settings,
        registry=registry,
        label_sets=label_sets,
        model_manager=model_manager,
        classifier=classifier,
        storage=storage,
        prediction=PredictionService(
            settings, model_manager, label_sets, classifier, storage, registry
        ),
        startup_errors=startup_errors,
    )
