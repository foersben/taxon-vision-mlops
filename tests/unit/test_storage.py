# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for the ephemeral in-memory image cache and its configured singleton accessor."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from taxon_vision.service.storage import (
    EphemeralImageCache,
    ImageCacheProtocol,
    MemoryImageCache,
    RedisImageCache,
    get_image_cache,
)


def test_hash_is_deterministic_sha256() -> None:
    """Identical bytes always map to the same 64-character hex digest."""
    first = EphemeralImageCache.hash_bytes(b"observation")
    assert first == EphemeralImageCache.hash_bytes(b"observation")
    assert len(first) == 64
    assert first != EphemeralImageCache.hash_bytes(b"other")


def test_store_and_get_round_trip() -> None:
    """Stored bytes are returned unchanged under the hash returned by `store`."""
    cache = EphemeralImageCache(max_items=4, ttl_seconds=60.0)
    image_hash = cache.store(b"payload")
    assert cache.get(image_hash) == b"payload"


def test_unknown_hash_returns_none() -> None:
    """Looking up a hash that was never stored yields None."""
    assert EphemeralImageCache(max_items=2, ttl_seconds=60.0).get("0" * 64) is None


def test_duplicate_store_does_not_grow_cache() -> None:
    """Re-storing identical bytes refreshes the entry instead of adding a second one."""
    cache = EphemeralImageCache(max_items=4, ttl_seconds=60.0)
    assert cache.store(b"same") == cache.store(b"same")
    assert len(cache) == 1


def test_lru_eviction_at_capacity() -> None:
    """The least recently used entry is evicted when capacity is exceeded."""
    cache = EphemeralImageCache(max_items=2, ttl_seconds=60.0)
    first = cache.store(b"a")
    second = cache.store(b"b")
    assert cache.get(first) == b"a"  # touch "a" so "b" becomes least recently used
    third = cache.store(b"c")
    assert cache.get(second) is None
    assert cache.get(first) == b"a"
    assert cache.get(third) == b"c"
    assert len(cache) == 2


def test_capacity_of_one_hundred_evicts_oldest_on_101st_insert() -> None:
    """Storing 101 distinct images into a 100-item cache evicts exactly the first one."""
    cache = EphemeralImageCache(max_items=100, ttl_seconds=60.0)
    hashes = [cache.store(str(i).encode()) for i in range(101)]
    assert len(cache) == 100
    assert cache.get(hashes[0]) is None
    assert cache.get(hashes[-1]) == b"100"


def test_expired_entries_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries older than the TTL are unreachable and removed on access."""
    clock = {"now": 1000.0}
    monkeypatch.setattr("taxon_vision.service.storage.time.monotonic", lambda: clock["now"])
    cache = EphemeralImageCache(max_items=4, ttl_seconds=10.0)
    image_hash = cache.store(b"short-lived")
    clock["now"] += 5.0
    assert cache.get(image_hash) == b"short-lived"
    clock["now"] += 10.1
    assert cache.get(image_hash) is None
    assert len(cache) == 0


def test_store_prunes_expired_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Insertion purges stale entries so they do not occupy capacity."""
    clock = {"now": 0.0}
    monkeypatch.setattr("taxon_vision.service.storage.time.monotonic", lambda: clock["now"])
    cache = EphemeralImageCache(max_items=4, ttl_seconds=10.0)
    cache.store(b"old")
    clock["now"] = 20.0
    cache.store(b"new")
    assert len(cache) == 1


def test_clear_empties_cache() -> None:
    """`clear` removes all entries."""
    cache = EphemeralImageCache(max_items=2, ttl_seconds=60.0)
    image_hash = cache.store(b"x")
    cache.clear()
    assert cache.get(image_hash) is None


@pytest.mark.parametrize(("max_items", "ttl"), [(0, 1.0), (-1, 1.0), (1, 0.0), (1, -5.0)])
def test_invalid_configuration_is_rejected(max_items: int, ttl: float) -> None:
    """Non-positive capacity or TTL raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        EphemeralImageCache(max_items=max_items, ttl_seconds=ttl)


def test_concurrent_stores_respect_capacity() -> None:
    """Parallel insertions never exceed capacity or corrupt the cache."""
    cache = EphemeralImageCache(max_items=8, ttl_seconds=60.0)

    def worker(offset: int) -> None:
        for i in range(50):
            cache.store(f"{offset}-{i}".encode())

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(cache) == 8


def test_singleton_is_configured_from_settings() -> None:
    """`get_image_cache` returns one shared instance built from `service` settings."""
    get_image_cache.cache_clear()
    cache = get_image_cache()
    assert cache is get_image_cache()
    assert isinstance(cache, ImageCacheProtocol)
    assert isinstance(cache, MemoryImageCache)
    assert cache.max_items > 0
    assert cache.ttl_seconds > 0


def test_image_cache_protocols_conformance() -> None:
    """Verify both MemoryImageCache and RedisImageCache implement ImageCacheProtocol."""
    mem_cache = MemoryImageCache(max_items=10, ttl_seconds=60.0)
    assert isinstance(mem_cache, ImageCacheProtocol)


def test_redis_image_cache_round_trip() -> None:
    """Bytes stored in RedisImageCache are returned under the computed hash."""
    import fakeredis

    client = fakeredis.FakeRedis()
    cache = RedisImageCache("redis://localhost:6379/0", ttl_seconds=60.0, redis_client=client)
    assert isinstance(cache, ImageCacheProtocol)
    assert cache.ping() is True

    payload = b"sample_observation_bytes"
    img_hash = cache.store(payload)
    assert len(img_hash) == 64
    assert cache.get(img_hash) == payload


def test_redis_image_cache_unknown_hash_returns_none() -> None:
    """Looking up a non-existent key in RedisImageCache yields None."""
    import fakeredis

    client = fakeredis.FakeRedis()
    cache = RedisImageCache("redis://localhost:6379/0", ttl_seconds=60.0, redis_client=client)
    assert cache.get("0" * 64) is None


def test_redis_image_cache_clear_and_len() -> None:
    """`clear` deletes only keys within its prefix and len counts active keys."""
    import fakeredis

    client = fakeredis.FakeRedis()
    cache = RedisImageCache(
        "redis://localhost:6379/0",
        ttl_seconds=60.0,
        key_prefix="test:taxon:",
        redis_client=client,
    )
    # External unmanaged key
    client.set("other:key", b"do_not_touch")

    assert len(cache) == 0
    h1 = cache.store(b"img1")
    h2 = cache.store(b"img2")
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0
    assert cache.get(h1) is None
    assert cache.get(h2) is None
    assert client.get("other:key") == b"do_not_touch"


def test_redis_image_cache_ttl_expiry() -> None:
    """Redis sets expiration accurately on stored items."""
    import fakeredis

    client = fakeredis.FakeRedis()
    cache = RedisImageCache("redis://localhost:6379/0", ttl_seconds=60.0, redis_client=client)
    img_hash = cache.store(b"expiring_soon")
    ttl = client.ttl(f"taxon:img:{img_hash}")
    assert 0 < ttl <= 60


def test_redis_invalid_configuration_is_rejected() -> None:
    """Non-positive TTL for RedisImageCache raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        RedisImageCache("redis://localhost:6379/0", ttl_seconds=0.0)


def test_redis_image_cache_ping_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """ping returns False on ConnectionError."""
    import redis

    cache = RedisImageCache("redis://invalid-host:6379/0", ttl_seconds=60.0)

    def mock_ping() -> bool:
        raise redis.ConnectionError("Connection refused")

    monkeypatch.setattr(cache._redis, "ping", mock_ping)
    assert cache.ping() is False


def test_get_image_cache_with_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_image_cache instantiates RedisImageCache when redis_url is set and reachable."""
    import fakeredis

    from taxon_vision.config import get_settings

    client = fakeredis.FakeRedis()

    def mock_init(self: Any, url: str = "", ttl_seconds: float = 60.0, **_kw: Any) -> None:
        _ = url
        self._redis = client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = "taxon:img:"

    monkeypatch.setattr(RedisImageCache, "__init__", mock_init)

    settings = get_settings()
    settings.service.redis_url = "redis://redis-ephemeral:6379/0"
    get_image_cache.cache_clear()
    try:
        cache = get_image_cache()
        assert isinstance(cache, RedisImageCache)
    finally:
        settings.service.redis_url = None
        get_image_cache.cache_clear()


def test_get_image_cache_fallback_on_unreachable_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_image_cache falls back to MemoryImageCache when redis_url is unreachable."""
    from taxon_vision.config import get_settings

    monkeypatch.setattr(RedisImageCache, "ping", lambda _self: False)

    settings = get_settings()
    settings.service.redis_url = "redis://unreachable:6379/0"
    get_image_cache.cache_clear()
    try:
        cache = get_image_cache()
        assert isinstance(cache, MemoryImageCache)
    finally:
        settings.service.redis_url = None
        get_image_cache.cache_clear()
