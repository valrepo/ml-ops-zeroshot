import threading

import pytest

from zeroshot.core.usage import ModelUsageRegistry
from zeroshot.errors import CacheInUseError, ModelInUseError


def test_lease_blocks_exclusive():
    r = ModelUsageRegistry()
    with r.lease("m"), pytest.raises(ModelInUseError) as exc, r.exclusive("m"):
        pass
    assert exc.value.code == "model_in_use"


def test_exclusive_blocks_lease():
    r = ModelUsageRegistry()
    with r.exclusive("m"), pytest.raises(ModelInUseError) as exc, r.lease("m"):
        pass
    assert exc.value.code == "model_being_deleted"


def test_lease_or_download_blocks_global_exclusive():
    r = ModelUsageRegistry()
    with r.lease("m"), pytest.raises(CacheInUseError), r.global_exclusive():
        pass
    with r.downloading("m"), pytest.raises(CacheInUseError), r.global_exclusive():
        pass
    with r.global_exclusive(), pytest.raises(CacheInUseError), r.lease("m"):
        pass


def test_download_blocks_exclusive():
    r = ModelUsageRegistry()
    with r.downloading("m"), pytest.raises(ModelInUseError), r.exclusive("m"):
        pass


def test_lease_released_on_exception():
    r = ModelUsageRegistry()
    with pytest.raises(RuntimeError), r.lease("m"):
        assert r.is_in_use("m")
        raise RuntimeError("boom")
    assert not r.is_in_use("m")
    assert not r.any_active()


def test_concurrent_leases_balance():
    r = ModelUsageRegistry()
    barrier = threading.Barrier(8)

    def worker():
        for _ in range(200):
            with r.lease("m"):
                pass
        barrier.wait()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not r.is_in_use("m")
    assert not r.any_active()


def test_notify_removed_calls_callbacks():
    r = ModelUsageRegistry()
    seen = []
    r.on_removed(seen.append)
    r.notify_removed("m")
    assert seen == ["m"]
