from pathlib import Path

import pytest

from zeroshot.config import Settings, load_model_catalogue, load_settings
from zeroshot.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
CREDS = {"minio_access_key": "a", "minio_secret_key": "b"}


def make(**kw):
    return load_settings(_env_file=None, **{**CREDS, **kw})


def test_defaults(monkeypatch):
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    s = make()
    assert s.max_input_chars == 5000
    assert s.minio_bucket == "zeroshot-learning-output"
    assert s.inference_device == "cpu"
    assert s.hf_home == Path(".cache/huggingface")
    assert s.max_loaded_models == 1
    assert s.warnings == []


def test_missing_secret_is_reported_without_value(monkeypatch):
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    with pytest.raises(ConfigError) as exc:
        load_settings(_env_file=None, minio_access_key="super-secret-access")
    assert "MINIO_SECRET_KEY is required" in exc.value.user_message
    assert "super-secret-access" not in exc.value.user_message
    assert exc.value.code == "config_missing"


@pytest.mark.parametrize("value", [0, "abc", 200_000])
def test_max_input_chars_rejected(value):
    with pytest.raises(ConfigError) as exc:
        make(max_input_chars=value)
    assert "MAX_INPUT_CHARS" in exc.value.user_message


def test_underscore_bucket_rejected_with_adr_hint():
    with pytest.raises(ConfigError) as exc:
        make(minio_bucket="zeroshot_learning_output")
    assert exc.value.code == "invalid_bucket_name"
    assert "0002-bucket-name-hyphens" in exc.value.user_message
    assert make(minio_bucket="zeroshot-learning-output").minio_bucket == "zeroshot-learning-output"


@pytest.mark.parametrize("bucket", ["ab", "UPPER", "a..b", "192.168.1.1", "-lead"])
def test_other_invalid_buckets(bucket):
    with pytest.raises(ConfigError):
        make(minio_bucket=bucket)


def test_transformers_cache_warning(monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/tmp/x")
    assert any("TRANSFORMERS_CACHE is ignored" in w for w in make().warnings)


def test_device_validation(monkeypatch):
    assert make().inference_device == "cpu"
    with pytest.raises(ConfigError):
        make(inference_device="cuda")
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(ConfigError) as exc:
        make(inference_device="mps")
    assert exc.value.code == "device_unavailable"


def test_endpoint_with_scheme_rejected():
    with pytest.raises(ConfigError):
        make(minio_endpoint="http://localhost:9010")


# ---- model catalogue -----------------------------------------------------------------

GOOD = """
models:
  - key: m1
    name: M1
    repo_id: org/m1
    task: zero-shot-classification
    revision: 8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8
    inference_defaults: {multi_label: false, hypothesis_template: "This is {}."}
"""


def write(tmp_path, text):
    p = tmp_path / "models.yaml"
    p.write_text(text)
    return p


def test_real_catalogue_has_two_models():
    cat = load_model_catalogue(ROOT / "configs" / "models.yaml")
    assert [m.repo_id for m in cat] == [
        "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "facebook/bart-large-mnli",
    ]
    assert all(len(m.revision) == 40 for m in cat)


def test_branch_revision_rejected(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_model_catalogue(
            write(tmp_path, GOOD.replace("8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8", "main"))
        )
    assert "commit SHA" in exc.value.user_message


def test_duplicate_key_rejected(tmp_path):
    second = GOOD.split("models:\n", 1)[1].replace("org/m1", "org/m2")
    with pytest.raises(ConfigError, match="duplicate model key"):
        load_model_catalogue(write(tmp_path, GOOD + second))


def test_template_without_placeholder_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_model_catalogue(write(tmp_path, GOOD.replace("This is {}.", "No placeholder")))


def test_malformed_yaml(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_model_catalogue(write(tmp_path, "models: [unclosed"))
    assert exc.value.code == "catalogue_invalid"


def test_empty_catalogue_returns_empty_list(tmp_path):
    assert load_model_catalogue(write(tmp_path, "models: []\n")) == []


def test_settings_class_is_importable_without_streamlit():
    assert Settings.__name__ == "Settings"
