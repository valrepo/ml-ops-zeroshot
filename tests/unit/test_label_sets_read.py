import pytest

from zeroshot.errors import UnknownLabelSetError
from zeroshot.services.label_sets import LabelSetService


def test_seed_copied_on_first_start(settings):
    assert not settings.label_sets_file.exists()
    svc = LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)
    assert settings.label_sets_file.exists()
    assert [s.key for s in svc.list()] == ["default", "extended"]
    assert svc.get("default").labels == ["sports", "politics", "technology", "finance"]
    assert len(svc.get("extended").labels) == 6


def test_unknown_key(settings):
    svc = LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)
    with pytest.raises(UnknownLabelSetError):
        svc.get("nope")


def test_invalid_entry_skipped_with_warning(settings):
    settings.label_sets_file.parent.mkdir(parents=True)
    settings.label_sets_file.write_text(
        "label_sets:\n"
        "  - {key: good, name: Good, labels: [a, b]}\n"
        "  - {key: bad, name: Bad, labels: [Sports, sports]}\n"
    )
    svc = LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)
    assert [s.key for s in svc.list()] == ["good"]
    warnings = svc.load_warnings()
    assert len(warnings) == 1 and "Duplicate label" in warnings[0]
