"""Principle XIV: a compatible model is added by configuration only."""

from pathlib import Path

from fakes import FakeHub, FakePipeline, file_only_validator

from zeroshot.config import load_model_catalogue
from zeroshot.domain.models import ModelStatus
from zeroshot.domain.predictions import InferenceParameters
from zeroshot.services.classifier import Classifier
from zeroshot.services.model_manager import ModelManager

ROOT = Path(__file__).resolve().parents[2]
THIRD = """
  - key: third-model
    name: Third Model
    repo_id: example-org/nli-third-model
    task: zero-shot-classification
    revision: 0123456789abcdef0123456789abcdef01234567
    inference_defaults: {multi_label: true, hypothesis_template: "It is about {}."}
"""


def test_third_model_works_without_code_changes(settings, registry, tmp_path):
    models_file = tmp_path / "models.yaml"
    models_file.write_text((ROOT / "configs" / "models.yaml").read_text() + THIRD)
    catalogue = load_model_catalogue(models_file)
    assert [m.key for m in catalogue][-1] == "third-model"

    mm = ModelManager(
        settings, catalogue, registry, hub=FakeHub(), artifact_validator=file_only_validator
    )
    assert mm.download(["third-model"])["third-model"].status == "downloaded"
    inst = mm.get("third-model")
    assert inst.status is ModelStatus.READY

    clf = Classifier(settings, registry, pipeline_factory=lambda path, device: FakePipeline())
    d = inst.definition.inference_defaults
    result = clf.classify(
        "Great match yesterday",
        ["sports", "finance"],
        inst,
        InferenceParameters(
            multi_label=d.multi_label, hypothesis_template=d.hypothesis_template, device="cpu"
        ),
    )
    assert result.predicted_class == "sports"
    assert result.inference_parameters.multi_label is True


def test_no_hard_coded_model_ids_in_source():
    ids = [
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "facebook/bart-large-mnli",
        "deberta-v3-zeroshot-v2",
        "bart-large-mnli",
    ]
    offenders = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*.py")
        for model_id in ids
        if model_id in p.read_text()
    ]
    assert offenders == []
