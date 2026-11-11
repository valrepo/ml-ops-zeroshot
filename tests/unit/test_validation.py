import pytest

from zeroshot.core.validation import validate_request
from zeroshot.errors import InputValidationError

LABELS = ["sports", "politics"]


@pytest.mark.parametrize(
    "text,labels,code",
    [
        ("", LABELS, "empty_text"),
        ("  \n\t", LABELS, "empty_text"),
        ("x" * 5001, LABELS, "text_too_long"),
        ("ok", ["sports"], "too_few_labels"),
        ("ok", [], "too_few_labels"),
        ("ok", ["sports", "  "], "empty_label"),
        ("ok", ["Sports", "sports"], "duplicate_label"),
    ],
)
def test_rejections(text, labels, code):
    with pytest.raises(InputValidationError) as exc:
        validate_request(text, labels, max_chars=5000)
    assert exc.value.code == code


def test_limit_is_named_in_message():
    with pytest.raises(InputValidationError) as exc:
        validate_request("x" * 5001, LABELS, max_chars=5000)
    assert "5000" in exc.value.user_message


def test_exactly_max_chars_passes():
    validate_request("x" * 5000, LABELS, max_chars=5000)
