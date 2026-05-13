from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
import copy
import httpx
import time

from .cache import TTLCache


class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 5.0):
        self.min_interval = 1.0 / max(calls_per_second, 0.01)
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.time()


class BlueskyClient:
    def __init__(
        self,
        pds_url: str,
        appview_url: str,
        handle: str,
        app_password: str,
        cache_ttl_resolve_seconds: int = 3600,
        cache_ttl_profile_seconds: int = 300,
        cache_ttl_follows_seconds: int = 45,
        cache_ttl_author_feed_seconds: int = 20,
        cache_ttl_likes_seconds: int = 60,
        local_mode: bool = False,
    ):
        self.pds_url = pds_url.rstrip("/")
        self.appview_url = appview_url.rstrip("/")
        self.handle = handle
        self.app_password = app_password
        self.access_jwt: Optional[str] = None
        self._http = httpx.AsyncClient(timeout=20.0)
        self.local_mode = local_mode

        # In local mode, use very long cache TTLs to avoid re-fetching
        resolve_ttl = 36000 if local_mode else cache_ttl_resolve_seconds
        profile_ttl = 3600 if local_mode else cache_ttl_profile_seconds
        follows_ttl = 300 if local_mode else cache_ttl_follows_seconds
        feed_ttl = 600 if local_mode else cache_ttl_author_feed_seconds
        likes_ttl = 300 if local_mode else cache_ttl_likes_seconds

        self._resolve_cache = TTLCache[str, str](resolve_ttl, max_items=4096)
        self._profile_cache = TTLCache[str, Dict[str, Any]](profile_ttl, max_items=4096)
        self._follows_cache = TTLCache[tuple[str, int], List[Dict[str, Any]]](follows_ttl, max_items=4096)
        self._author_feed_cache = TTLCache[tuple[str, int], List[Dict[str, Any]]](feed_ttl, max_items=8192)
        self._likes_cache = TTLCache[str, Tuple[Set[str], Dict[str, int]]](likes_ttl, max_items=2048)

        self._rate_limiter = None if local_mode else RateLimiter(calls_per_second=8.0)
        self._api_call_count = 0

    async def close(self) -> None:
        await self._http.aclose()

    async def login(self) -> None:
        if not self.handle or not self.app_password:
            return
        payload = {"identifier": self.handle, "password": self.app_password}
        r = await self._http.post(
            f"{self.pds_url}/xrpc/com.atproto.server.createSession", json=payload
        )
        r.raise_for_status()
        data = r.json()
        self.access_jwt = data.get("accessJwt")

    def _headers(self) -> Dict[str, str]:
        if self.access_jwt:
            return {"Authorization": f"Bearer {self.access_jwt}"}
        return {}

    async def _get(self, url: str, **kwargs) -> httpx.Response:
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        self._api_call_count += 1
        return await self._http.get(url, **kwargs)

    async def resolve_handle(self, handle: str) -> str:
        cached = await self._resolve_cache.get(handle)
        if cached:
            return cached

        r = await self._get(
            f"{self.appview_url}/xrpc/com.atproto.identity.resolveHandle",
            params={"handle": handle},
            headers=self._headers(),
        )
        r.raise_for_status()
        did = r.json()["did"]
        await self._resolve_cache.set(handle, did)
        return did

    async def get_profile(self, actor: str) -> Dict[str, Any]:
        cached = await self._profile_cache.get(actor)
        if cached:
            return copy.deepcopy(cached)

        r = await self._get(
            f"{self.appview_url}/xrpc/app.bsky.actor.getProfile",
            params={"actor": actor},
            headers=self._headers(),
        )
        r.raise_for_status()
        data = r.json()
        await self._profile_cache.set(actor, data)
        return copy.deepcopy(data)

    async def get_author_feed(self, actor: str, limit: int = 10) -> List[Dict[str, Any]]:
        key = (actor, limit)
        cached = await self._author_feed_cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)

        r = await self._get(
            f"{self.appview_url}/xrpc/app.bsky.feed.getAuthorFeed",
            params={"actor": actor, "limit": limit, "filter": "posts_no_replies"},
            headers=self._headers(),
        )
        r.raise_for_status()
        data = r.json().get("feed", [])
        await self._author_feed_cache.set(key, data)
        return copy.deepcopy(data)

    async def get_follows(self, actor: str, limit: int = 100) -> List[Dict[str, Any]]:
        key = (actor, limit)
        cached = await self._follows_cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)

        out: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while len(out) < limit:
            page_limit = min(100, limit - len(out))
            params = {"actor": actor, "limit": page_limit}
            if cursor:
                params["cursor"] = cursor
            r = await self._get(
                f"{self.appview_url}/xrpc/app.bsky.graph.getFollows",
                params=params,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("follows", []))
            cursor = data.get("cursor")
            if not cursor:
                break

        await self._follows_cache.set(key, out)
        return copy.deepcopy(out)

    async def get_actor_likes(self, actor: str, limit: int = 200) -> Tuple[Set[str], Dict[str, int]]:
        """Return (liked_uris, liked_author_frequency)."""
        cached = await self._likes_cache.get(actor)
        if cached is not None:
            return cached

        liked_uris: Set[str] = set()
        author_freq: Dict[str, int] = {}
        cursor: Optional[str] = None
        fetched = 0
        while fetched < limit:
            page_limit = min(100, limit - fetched)
            params = {"actor": actor, "limit": page_limit}
            if cursor:
                params["cursor"] = cursor
            try:
                r = await self._get(
                    f"{self.appview_url}/xrpc/app.bsky.feed.getActorLikes",
                    params=params,
                    headers=self._headers(),
                )
                r.raise_for_status()
                data = r.json()
                for item in data.get("feed", []):
                    post = item.get("post", {})
                    uri = post.get("uri")
                    author_did = post.get("author", {}).get("did")
                    if uri:
                        liked_uris.add(uri)
                    if author_did:
                        author_freq[author_did] = author_freq.get(author_did, 0) + 1
                cursor = data.get("cursor")
                fetched += page_limit
                if not cursor:
                    break
            except Exception:
                break

        result = (liked_uris, author_freq)
        await self._likes_cache.set(actor, result)
        return result

    async def get_viewer_recent_posts(self, actor: str, limit: int = 8) -> List[str]:
        feed = await self.get_author_feed(actor, limit=limit)
        texts: List[str] = []
        for item in feed:
            post = item.get("post", {})
            record = post.get("record", {})
            text = record.get("text", "")
            if text:
                texts.append(text)
        return texts

    def reset_api_call_count(self) -> int:
        count = self._api_call_count
        self._api_call_count = 0
        return count
