"""Unit-test-only guard: no network, no real downloads, no real pipelines (SC-005).

Lives here — not in the root conftest — so tests/model/ can still load a real model.
"""

import pytest


def _forbidden(*_args, **_kwargs):
    raise AssertionError("network/model access in unit test")


@pytest.fixture(autouse=True)
def _offline_guard(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.snapshot_download", _forbidden)
    monkeypatch.setattr("transformers.pipeline", _forbidden)
