import pytest

from zeroshot.domain.labels import LabelSet
from zeroshot.errors import LabelSetValidationError, UnknownLabelSetError
from zeroshot.services.label_sets import LabelSetService


@pytest.fixture
def svc(settings):
    return LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)


def test_create_and_survives_restart(svc, settings):
    svc.save(LabelSet(key="news", name="News", labels=["sports", "politics", "science"]))
    again = LabelSetService(settings.label_sets_file, settings.label_sets_seed_file)
    assert again.get("news").labels == ["sports", "politics", "science"]


def test_replace_existing(svc):
    svc.save(LabelSet(key="default", name="Default topics", labels=["a", "b", "c"]))
    assert svc.get("default").labels == ["a", "b", "c"]


def test_delete(svc):
    svc.delete("extended")
    with pytest.raises(UnknownLabelSetError):
        svc.get("extended")
    with pytest.raises(UnknownLabelSetError):
        svc.delete("extended")


@pytest.mark.parametrize(
    "data,code",
    [
        ({"key": "k", "name": "N", "labels": ["a", " "]}, "empty_label"),
        ({"key": "k", "name": "N", "labels": ["Sports", "sports"]}, "duplicate_label"),
        ({"key": "k", "name": "N", "labels": ["only"]}, "too_few_labels"),
        ({"key": "k", "name": "N", "labels": ["a", "x" * 101]}, "label_too_long"),
        ({"key": "Bad Key", "name": "N", "labels": ["a", "b"]}, "invalid_key"),
        ({"key": "k", "name": "  ", "labels": ["a", "b"]}, "invalid_name"),
    ],
)
def test_validation_errors_leave_file_unchanged(svc, settings, data, code):
    before = settings.label_sets_file.read_bytes()
    with pytest.raises(LabelSetValidationError) as exc:
        svc.save(LabelSet(**data))
    assert exc.value.code == code
    assert settings.label_sets_file.read_bytes() == before


def test_labels_are_trimmed_but_text_preserved(svc):
    saved = svc.save(LabelSet(key="t", name=" Trim ", labels=["  Sci-Fi ", "Drama"]))
    assert saved.labels == ["Sci-Fi", "Drama"] and saved.name == "Trim"


def test_atomic_write_keeps_original_on_failure(svc, settings, monkeypatch):
    before = settings.label_sets_file.read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("zeroshot.services.label_sets.os.replace", boom)
    with pytest.raises(OSError):
        svc.save(LabelSet(key="news", name="News", labels=["a", "b"]))
    assert settings.label_sets_file.read_bytes() == before
    assert [p.name for p in settings.label_sets_file.parent.glob("*.tmp")] == []
    assert "news" not in {s.key for s in svc.list()}


def test_file_lock_used(svc, settings, monkeypatch):
    acquired = []
    import filelock

    real = filelock.FileLock

    class Spy(real):
        def acquire(self, *a, **k):
            acquired.append(self.lock_file)
            return super().acquire(*a, **k)

    monkeypatch.setattr("zeroshot.services.label_sets.FileLock", Spy)
    svc.save(LabelSet(key="news", name="News", labels=["a", "b"]))
    assert acquired and acquired[0].endswith("label_sets.yaml.lock")
