"""Lightweight test doubles: no network, no model downloads (Principle VII)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zeroshot.domain.predictions import PredictionRecord
from zeroshot.errors import BucketInitError, StorageWriteError


class FakePipeline:
    """Mimics transformers' zero-shot pipeline output shape."""

    def __init__(self, scores: dict[str, float] | None = None, fail: Exception | None = None):
        self.scores = scores
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text, candidate_labels, hypothesis_template, multi_label):  # noqa: ANN001
        self.calls.append(
            {
                "text": text,
                "candidate_labels": list(candidate_labels),
                "hypothesis_template": hypothesis_template,
                "multi_label": multi_label,
            }
        )
        if self.fail:
            raise self.fail
        labels = list(candidate_labels)
        if self.scores is not None:
            raw = [self.scores[label] for label in labels]
        else:
            logits = [3.0 if "sport" in label.lower() else 0.0 for label in labels]
            if multi_label:
                raw = [1 / (1 + math.exp(-x)) for x in logits]
            else:
                exp = [math.exp(x) for x in logits]
                raw = [e / sum(exp) for e in exp]
        # transformers returns labels sorted by score (descending); emulate that.
        order = sorted(range(len(labels)), key=lambda i: -raw[i])
        return {
            "sequence": text,
            "labels": [labels[i] for i in order],
            "scores": [raw[i] for i in order],
        }


class FakeStorage:
    """In-memory PredictionStore."""

    def __init__(self, fail_on_save: bool = False, fail_on_ensure: bool = False):
        self.fail_on_save = fail_on_save
        self.fail_on_ensure = fail_on_ensure
        self.objects: dict[str, bytes] = {}

    def ensure_bucket(self) -> None:
        if self.fail_on_ensure:
            raise BucketInitError("Storage initialization failed (fake).")

    def object_key(self, record: PredictionRecord) -> str:
        ts = record.timestamp
        return f"predictions/{ts:%Y}/{ts:%m}/{ts:%d}/{record.request_id}.json"

    def save(self, record: PredictionRecord) -> str:
        self.ensure_bucket()
        if self.fail_on_save:
            raise StorageWriteError("Could not write the prediction (fake).")
        key = self.object_key(record)
        self.objects[key] = record.to_json_bytes()
        return key

    def get(self, key: str) -> PredictionRecord:
        return PredictionRecord.model_validate_json(self.objects[key])


class FakeHub:
    """Stand-in for the ``huggingface_hub`` module's ``snapshot_download``."""

    def __init__(self, mode: str = "ok", fail_for: set[str] | None = None):
        self.mode = mode  # ok | network | missing_weights | no_entailment
        self.fail_for = fail_for
        self.calls: list[dict[str, Any]] = []

    def snapshot_download(
        self,
        repo_id,
        revision,
        local_dir,
        allow_patterns=None,  # noqa: ANN001
        ignore_patterns=None,
        token=None,
        **_: Any,
    ) -> str:
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": Path(local_dir),
                "allow_patterns": allow_patterns,
                "ignore_patterns": ignore_patterns,
                "token": token,
            }
        )
        mode = self.mode if (self.fail_for is None or repo_id in self.fail_for) else "ok"
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        id2label = {"0": "entailment", "1": "not_entailment"}
        if mode == "no_entailment":
            id2label = {"0": "positive", "1": "negative"}
        (target / "config.json").write_text(
            json.dumps({"architectures": ["XForSequenceClassification"], "id2label": id2label})
        )
        if mode == "network":
            raise ConnectionError("simulated network failure")
        if mode != "missing_weights":
            (target / "model.safetensors").write_bytes(b"\x00" * 8)
        (target / "tokenizer.json").write_text("{}")
        return str(target)


def file_only_validator(path: Path) -> list[str]:
    """Artifact validator for tests: checks files and config without loading transformers."""
    problems = []
    if not (path / "config.json").exists():
        problems.append("config.json missing")
    if not list(path.glob("*.safetensors")):
        problems.append("no *.safetensors weights")
    if not any((path / name).exists() for name in ("tokenizer.json", "spm.model", "vocab.json")):
        problems.append("no tokenizer files")
    if (path / "config.json").exists():
        config = json.loads((path / "config.json").read_text())
        labels = {str(v).lower() for v in config.get("id2label", {}).values()}
        if "entailment" not in labels:
            problems.append("id2label has no 'entailment' label")
    return problems


SAMPLE_REVISION = "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"


def sample_record_data(**over):
    """Brief §3.5 example values, completed per the contract (real DeBERTa revision)."""
    data = {
        "request_id": "6a6d6a85-9dbb-4f3e-9f89-5d8e4e8c1b2a",
        "timestamp": datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
        "input_text": "The team won the championship after a very close game.",
        "predicted_class": "sports",
        "confidence": 0.96,
        "candidate_labels": ["sports", "politics", "technology", "finance"],
        "ranked_predictions": [
            {"label": "sports", "score": 0.96},
            {"label": "technology", "score": 0.02},
            {"label": "finance", "score": 0.01},
            {"label": "politics", "score": 0.01},
        ],
        "model_name": "DeBERTa Zero-Shot v2",
        "model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "model_revision": SAMPLE_REVISION,
        "inference_parameters": {
            "multi_label": False,
            "hypothesis_template": "This text is about {}",
            "device": "cpu",
        },
        "label_set": "default",
        "application_version": "0.1.0",
    }
    data.update(over)
    return data
