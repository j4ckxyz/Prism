from __future__ import annotations

import asyncio
import time
from typing import Callable, Dict, Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    def __init__(self, ttl_seconds: float, max_items: int = 2048):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: Dict[K, tuple[float, V]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: K) -> V | None:
        async with self._lock:
            entry = self._items.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                self._items.pop(key, None)
                return None
            return value

    async def set(self, key: K, value: V) -> None:
        async with self._lock:
            if len(self._items) >= self.max_items:
                self._prune_expired_or_oldest_locked()
            self._items[key] = (time.time() + self.ttl_seconds, value)

    async def get_or_set(self, key: K, factory: Callable[[], "V"]):
        value = await self.get(key)
        if value is not None:
            return value
        value = factory()
        await self.set(key, value)
        return value

    def _prune_expired_or_oldest_locked(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._items.items() if exp < now]
        for k in expired:
            self._items.pop(k, None)
        if len(self._items) < self.max_items:
            return
        oldest_key = min(self._items, key=lambda k: self._items[k][0])
        self._items.pop(oldest_key, None)
