import asyncio

import pytest

from app.cache import TTLCache


@pytest.mark.asyncio
async def test_ttl_cache_expires_items():
    cache = TTLCache[str, int](ttl_seconds=0.05)
    await cache.set("k", 42)
    assert await cache.get("k") == 42
    await asyncio.sleep(0.06)
    assert await cache.get("k") is None
