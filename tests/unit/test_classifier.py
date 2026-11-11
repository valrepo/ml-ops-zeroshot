import logging
import math

import pytest
from fakes import FakePipeline

from zeroshot.domain.models import ModelStatus
from zeroshot.domain.predictions import InferenceParameters
from zeroshot.errors import InferenceError, InputValidationError, ModelLoadError, ModelNotReadyError
from zeroshot.services.classifier import Classifier
from zeroshot.services.model_manager import ModelManager

LABELS = ["sports", "politics", "technology", "finance"]
TEXT = "The team won the championship after a very close game."


class Factory:
    def __init__(self, pipeline=None, fail=None):
        self.pipeline = pipeline or FakePipeline()
        self.fail = fail
        self.calls = []

    def __call__(self, path, device):
        self.calls.append((path, device))
        if self.fail:
            raise self.fail
        return self.pipeline


@pytest.fixture
def installs(settings, catalogue, registry, ready_model):
    for d in catalogue:
        ready_model(d)
    mm = ModelManager(settings, catalogue, registry)
    return [mm.get(d.key) for d in catalogue]


def params(inst, multi=False):
    d = inst.definition.inference_defaults
    return InferenceParameters(
        multi_label=multi, hypothesis_template=d.hypothesis_template, device="cpu"
    )


def test_scores_every_label_sorted(settings, registry, installs):
    clf = Classifier(settings, registry, pipeline_factory=Factory())
    r = clf.classify(TEXT, LABELS, installs[0], params(installs[0]))
    assert sorted(p.label for p in r.ranked) == sorted(LABELS)
    scores = [p.score for p in r.ranked]
    assert scores == sorted(scores, reverse=True)
    assert r.predicted_class == "sports" and r.confidence == r.ranked[0].score
    assert math.isclose(sum(scores), 1.0, abs_tol=1e-6)
    assert r.model_revision == installs[0].marker.revision
    assert r.model_id == installs[0].definition.repo_id
    assert r.latency_ms >= 0


def test_ties_keep_candidate_order(settings, registry, installs):
    pipe = FakePipeline(
        scores={"sports": 0.25, "politics": 0.25, "technology": 0.25, "finance": 0.25}
    )
    clf = Classifier(settings, registry, pipeline_factory=Factory(pipe))
    r = clf.classify(
        TEXT, ["finance", "sports", "politics", "technology"], installs[0], params(installs[0])
    )
    assert [p.label for p in r.ranked] == ["finance", "sports", "politics", "technology"]


def test_parameters_passed_to_pipeline(settings, registry, installs):
    pipe = FakePipeline()
    clf = Classifier(settings, registry, pipeline_factory=Factory(pipe))
    clf.classify(TEXT, LABELS, installs[0], params(installs[0], multi=True))
    assert pipe.calls[0]["hypothesis_template"] == "This text is about {}"
    assert pipe.calls[0]["multi_label"] is True


def test_device_from_settings(settings, registry, installs):
    factory = Factory()
    clf = Classifier(settings, registry, pipeline_factory=factory)
    r = clf.classify(TEXT, LABELS, installs[0], params(installs[0]))
    assert factory.calls[0][1] == settings.inference_device == "cpu"
    assert r.inference_parameters.device == "cpu"


def test_nan_score_is_inference_error(settings, registry, installs):
    pipe = FakePipeline(
        scores={"sports": float("nan"), "politics": 0.5, "technology": 0.3, "finance": 0.2}
    )
    clf = Classifier(settings, registry, pipeline_factory=Factory(pipe))
    with pytest.raises(InferenceError) as exc:
        clf.classify(TEXT, LABELS, installs[0], params(installs[0]))
    assert exc.value.code == "invalid_scores"


def test_pipeline_exception_is_inference_error(settings, registry, installs):
    clf = Classifier(
        settings, registry, pipeline_factory=Factory(FakePipeline(fail=RuntimeError("boom")))
    )
    with pytest.raises(InferenceError):
        clf.classify(TEXT, LABELS, installs[0], params(installs[0]))


def test_factory_exception_is_load_error_and_logged(settings, registry, installs, caplog):
    clf = Classifier(settings, registry, pipeline_factory=Factory(fail=OSError("corrupt")))
    with caplog.at_level(logging.ERROR, logger="zeroshot"), pytest.raises(ModelLoadError) as exc:
        clf.classify(TEXT, LABELS, installs[0], params(installs[0]))
    assert "Traceback" not in exc.value.user_message
    rec = [r for r in caplog.records if getattr(r, "event", "") == "pipeline.load_failed"][-1]
    assert rec.exc_info is not None


def test_not_ready_model_rejected(settings, registry, catalogue):
    mm = ModelManager(settings, catalogue, registry)
    inst = mm.get(catalogue[0].key)
    assert inst.status is ModelStatus.NOT_AVAILABLE
    clf = Classifier(settings, registry, pipeline_factory=Factory())
    with pytest.raises(ModelNotReadyError):
        clf.classify(TEXT, LABELS, inst, params(inst))


def test_input_validation(settings, registry, installs):
    clf = Classifier(settings, registry, pipeline_factory=Factory())
    with pytest.raises(InputValidationError):
        clf.classify("   ", LABELS, installs[0], params(installs[0]))


def test_lru_keeps_one_model(settings, registry, installs, caplog):
    factory = Factory()
    clf = Classifier(settings, registry, pipeline_factory=factory)
    a, b = installs
    with caplog.at_level(logging.INFO, logger="zeroshot"):
        clf.classify(TEXT, LABELS, a, params(a))
        clf.classify(TEXT, LABELS, a, params(a))
        assert len(factory.calls) == 1  # reused
        clf.classify(TEXT, LABELS, b, params(b))  # evicts a
        clf.classify(TEXT, LABELS, a, params(a))  # reload a
    assert len(factory.calls) == 3
    events = [getattr(r, "event", "") for r in caplog.records]
    assert "pipeline.loaded" in events and "pipeline.evicted" in events


def test_registry_removal_evicts(settings, registry, installs):
    factory = Factory()
    clf = Classifier(settings, registry, pipeline_factory=factory)
    clf.classify(TEXT, LABELS, installs[0], params(installs[0]))
    registry.notify_removed(installs[0].key)
    clf.classify(TEXT, LABELS, installs[0], params(installs[0]))
    assert len(factory.calls) == 2
