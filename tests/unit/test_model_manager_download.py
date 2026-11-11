import json
import os
import time

import pytest
from fakes import FakeHub, file_only_validator

from zeroshot.domain.models import DEFAULT_ALLOW_PATTERNS, DEFAULT_IGNORE_PATTERNS, ModelStatus
from zeroshot.services.model_manager import ModelManager


def make(settings, catalogue, registry, hub, **kw):
    return ModelManager(
        settings, catalogue, registry, hub=hub, artifact_validator=file_only_validator, **kw
    )


def test_successful_download_writes_marker_and_is_ready(settings, catalogue, registry):
    hub = FakeHub()
    mm = make(settings, catalogue, registry, hub)
    out = mm.download([catalogue[0].key])
    assert out[catalogue[0].key].status == "downloaded"
    inst = mm.get(catalogue[0].key)
    assert inst.status is ModelStatus.READY
    marker = json.loads((inst.local_path / ".complete.json").read_text())
    assert marker["model_id"] == catalogue[0].repo_id
    assert marker["revision"] == catalogue[0].revision
    assert marker["downloaded_at"].endswith("Z")
    assert "model.safetensors" in marker["files"] and marker["total_bytes"] > 0
    assert not list((settings.model_dir / ".staging").glob("*"))


def test_snapshot_download_arguments(settings, catalogue, registry):
    hub = FakeHub()
    make(settings, catalogue, registry, hub).download([catalogue[0].key])
    call = hub.calls[0]
    assert call["repo_id"] == catalogue[0].repo_id
    assert call["revision"] == catalogue[0].revision
    staging = call["local_dir"]
    assert staging.parent == (settings.model_dir / ".staging").resolve()
    assert staging.name.startswith(f"{catalogue[0].slug}--{catalogue[0].revision}--")
    assert call["allow_patterns"] == DEFAULT_ALLOW_PATTERNS
    assert call["ignore_patterns"] == DEFAULT_IGNORE_PATTERNS
    assert call["token"] is None


def test_token_passed_but_never_in_errors(settings, catalogue, registry):
    s = settings.model_copy(update={"hf_token": settings.minio_secret_key.__class__("hf_SECRET")})
    hub = FakeHub(mode="network")
    out = make(s, catalogue, registry, hub).download([catalogue[0].key])
    assert hub.calls[0]["token"] == "hf_SECRET"
    assert "hf_SECRET" not in (out[catalogue[0].key].error or "")


def test_already_present_skips_hub(settings, catalogue, registry, ready_model):
    ready_model(catalogue[0])
    hub = FakeHub()
    out = make(settings, catalogue, registry, hub).download([catalogue[0].key])
    assert out[catalogue[0].key].status == "already_present"
    assert hub.calls == []


@pytest.mark.parametrize("mode", ["network", "missing_weights", "no_entailment"])
def test_failures_never_ready_and_leave_no_staging(settings, catalogue, registry, mode):
    mm = make(settings, catalogue, registry, FakeHub(mode=mode))
    out = mm.download([catalogue[0].key])
    outcome = out[catalogue[0].key]
    assert outcome.status == "failed" and outcome.error
    inst = mm.get(catalogue[0].key)
    assert inst.status is ModelStatus.FAILED
    assert not (settings.model_dir / catalogue[0].slug / catalogue[0].revision).exists()
    assert not list((settings.model_dir / ".staging").glob("*"))


def test_one_failure_does_not_stop_others(settings, catalogue, registry):
    hub = FakeHub(mode="network", fail_for={catalogue[0].repo_id})
    mm = make(settings, catalogue, registry, hub)
    out = mm.download([catalogue[0].key, catalogue[1].key])
    assert out[catalogue[0].key].status == "failed"
    assert out[catalogue[1].key].status == "downloaded"
    assert mm.get(catalogue[1].key).status is ModelStatus.READY


def test_progress_callback(settings, catalogue, registry, ready_model):
    ready_model(catalogue[1])
    events = []
    hub = FakeHub(mode="network", fail_for={catalogue[0].repo_id})
    make(settings, catalogue, registry, hub).download(
        [catalogue[0].key, catalogue[1].key], on_progress=lambda k, s: events.append((k, s))
    )
    assert events == [
        (catalogue[0].key, "started"),
        (catalogue[0].key, "failed"),
        (catalogue[1].key, "already_present"),
    ]


def test_stale_staging_removed_at_init(settings, catalogue, registry):
    staging = settings.model_dir / ".staging"
    old, new = staging / "old", staging / "new"
    old.mkdir(parents=True)
    new.mkdir()
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    make(settings, catalogue, registry, FakeHub())
    assert not old.exists() and new.exists()


def test_download_marks_registry(settings, catalogue, registry):
    seen = []

    class SpyHub(FakeHub):
        def snapshot_download(self, *a, **kw):
            seen.append(registry.any_active())
            return super().snapshot_download(*a, **kw)

    make(settings, catalogue, registry, SpyHub()).download([catalogue[0].key])
    assert seen == [True]
    assert not registry.any_active()


def test_incomplete_installation_is_replaced(settings, catalogue, registry):
    d = settings.model_dir / catalogue[0].slug / catalogue[0].revision
    d.mkdir(parents=True)
    (d / "junk.txt").write_text("x")
    mm = make(settings, catalogue, registry, FakeHub())
    assert mm.get(catalogue[0].key).status is ModelStatus.INCOMPLETE
    assert mm.download([catalogue[0].key])[catalogue[0].key].status == "downloaded"
    assert mm.get(catalogue[0].key).status is ModelStatus.READY
    assert not (d / "junk.txt").exists()
