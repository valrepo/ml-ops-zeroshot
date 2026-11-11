"""Opt-in smoke test with the real DeBERTa model (SC-006).

Run with ``uv run pytest -m model tests/model -s``.

Skipped unless the model is READY in the real MODEL_DIR. Not part of the default suite.
"""

import pytest

from zeroshot.config import load_model_catalogue, load_settings
from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.domain.models import ModelStatus
from zeroshot.domain.predictions import InferenceParameters
from zeroshot.services.classifier import Classifier
from zeroshot.services.model_manager import ModelManager

pytestmark = pytest.mark.model

TEXT = "The team won the championship after a very close game."
LABELS = ["sports", "politics", "technology", "finance"]


def test_real_deberta_classifies_sports():
    settings = load_settings()
    registry = ModelUsageRegistry()
    mm = ModelManager(settings, load_model_catalogue(settings.models_file), registry)
    inst = mm.get("deberta-v3-zeroshot-v2")
    if inst.status is not ModelStatus.READY:
        pytest.skip("DeBERTa is not downloaded; download it on the Administration page first")
    d = inst.definition.inference_defaults
    params = InferenceParameters(
        multi_label=d.multi_label,
        hypothesis_template=d.hypothesis_template,
        device=settings.inference_device,
    )
    clf = Classifier(settings, registry)
    clf.classify(TEXT, LABELS, inst, params)  # warm-up: loads the model (not timed for SC-006)
    result = clf.classify(TEXT, LABELS, inst, params)
    print(
        f"\npredicted={result.predicted_class} confidence={result.confidence:.3f} "
        f"latency_ms={result.latency_ms:.0f} device={settings.inference_device}"
    )
    assert result.predicted_class == "sports"
    assert result.latency_ms < 10_000  # SC-006
