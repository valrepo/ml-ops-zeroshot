import json
import logging

from zeroshot.logging import JsonFormatter, RedactingFilter, log_event, text_fingerprint


def _record(caplog, **fields):
    logger = logging.getLogger("zeroshot.test")
    with caplog.at_level(logging.INFO, logger="zeroshot"):
        log_event(logger, "unit.event", **fields)
    return caplog.records[-1]


def test_json_output(caplog):
    rec = _record(caplog, request_id="r1", outcome="ok")
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["event"] == "unit.event"
    assert payload["ts"].endswith("Z")
    assert payload["request_id"] == "r1"


def test_secret_fields_redacted(caplog):
    rec = _record(caplog, hf_token="hf_abc", secret_key="s3cr3t", password="pw", model="m")
    RedactingFilter().filter(rec)
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["hf_token"] == payload["secret_key"] == payload["password"] == "***"
    assert payload["model"] == "m"
    assert "hf_abc" not in JsonFormatter().format(rec)


def test_text_fingerprint_never_contains_text():
    text = "Confidential: the merger closes on Friday."
    fp = text_fingerprint(text)
    assert fp["input_chars"] == len(text)
    assert len(fp["input_sha256_12"]) == 12
    int(fp["input_sha256_12"], 16)
    assert text not in json.dumps(fp)
