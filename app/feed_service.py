from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .bluesky_client import BlueskyClient
from .embedding import OllamaEmbedder, cosine_similarity
from .models import CandidatePost, parse_bsky_time
from .preferences import PreferenceStore
from .ranker import Ranker


@dataclass
class PoolMeta:
    last_access: float = 0.0
    last_refresh: float = 0.0
    access_count: int = 0
    total_refreshes: int = 0
    api_calls: int = 0
    pool_size: int = 0
    fresh_count: int = 0


class FeedService:
    def __init__(self, client: BlueskyClient, ranker: Ranker, embedder: OllamaEmbedder, settings):
        self.client = client
        self.ranker = ranker
        self.embedder = embedder
        self.settings = settings
        self.pref_store = PreferenceStore(settings.preference_store_path)
        self._candidate_pools: Dict[str, List[CandidatePost]] = {}
        self._pool_meta: Dict[str, PoolMeta] = {}
        self._building: Set[str] = set()
        self._refresh_task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Background pool warming
    # ------------------------------------------------------------------
    def start_background_refresh(self) -> None:
        if self.settings.local_mode:
            return
        self._running = True
        self._refresh_task = asyncio.create_task(self._background_loop())

    def stop_background_refresh(self) -> None:
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()

    async def _background_loop(self) -> None:
        while self._running:
            try:
                await self._run_background_batch()
            except Exception:
                pass
            try:
                await asyncio.sleep(self.settings.background_refresh_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _run_background_batch(self) -> None:
        now = time.time()
        stale: List[tuple[float, str]] = []

        for handle, meta in list(self._pool_meta.items()):
            if handle not in self._candidate_pools or handle in self._building:
                continue
            interval = self._target_interval(now - meta.last_access)
            if interval == float("inf"):
                continue
            if now - meta.last_refresh >= interval:
                stale.append((-meta.last_access, handle))

        stale.sort()

        tasks = []
        for _, handle in stale[: self.settings.max_concurrent_refreshes]:
            tasks.append(asyncio.create_task(self._refresh_pool(handle)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _target_interval(time_since_access: float) -> float:
        if time_since_access < 600:
            return 120
        elif time_since_access < 3600:
            return 600
        elif time_since_access < 86400:
            return 3600
        elif time_since_access < 604800:
            return 21600
        else:
            return float("inf")

    async def _refresh_pool(self, handle: str) -> None:
        if handle in self._building:
            return
        self._building.add(handle)
        try:
            pool = await self._build_candidate_pool(handle)
            self._candidate_pools[handle] = pool
            meta = self._pool_meta.get(handle)
            if meta:
                meta.last_refresh = time.time()
                meta.total_refreshes += 1
                meta.api_calls = self.client.reset_api_call_count()
                meta.pool_size = len(pool)
            self._evict_if_needed()
        except Exception:
            pass
        finally:
            self._building.discard(handle)

    def _evict_if_needed(self) -> None:
        max_pools = getattr(self.settings, "max_pools_in_memory", 200)
        if len(self._candidate_pools) <= max_pools:
            return

        sorted_handles = sorted(
            self._candidate_pools.keys(),
            key=lambda h: self._pool_meta.get(h, PoolMeta()).last_access,
        )
        to_evict = max(1, len(sorted_handles) - max_pools)
        now = time.time()
        for h in sorted_handles[:to_evict]:
            self._candidate_pools.pop(h, None)
            meta = self._pool_meta.get(h)
            if meta and now - meta.last_access > 604800:
                self._pool_meta.pop(h, None)

    def get_pool_meta(self, handle: str) -> PoolMeta | None:
        return self._pool_meta.get(handle)

    # ------------------------------------------------------------------
    # Liked-author analysis
    # ------------------------------------------------------------------
    def _get_liked_author_boosts(self, viewer_handle: str, viewer_did: str) -> Dict[str, float]:
        cached = self.pref_store.get_liked_author_boosts(viewer_handle)
        if cached:
            return cached
        return {}

    async def _analyze_liked_authors(self, viewer_handle: str, viewer_did: str) -> Dict[str, float]:
        liked_uris, author_freq = await self.client.get_actor_likes(viewer_did, limit=self.settings.max_likes_fetch)
        if not author_freq:
            return {}
        boosts = self.pref_store.record_liked_authors(viewer_handle, author_freq)
        return boosts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def build_feed(self, viewer_handle: str, limit: int | None = None, cycle: bool = False) -> List[CandidatePost]:
        now = time.time()
        meta = self._pool_meta.setdefault(viewer_handle, PoolMeta(last_access=now, last_refresh=now))
        meta.last_access = now
        meta.access_count += 1

        # Prune old shown entries periodically
        if meta.access_count % 10 == 0:
            self.pref_store.prune_shown(viewer_handle)

        out_limit = limit or self.settings.output_size

        # Ensure we have a pool
        if viewer_handle not in self._candidate_pools:
            while viewer_handle in self._building:
                await asyncio.sleep(0.05)
            if viewer_handle not in self._candidate_pools:
                self._building.add(viewer_handle)
                try:
                    pool = await self._build_candidate_pool(viewer_handle)
                    self._candidate_pools[viewer_handle] = pool
                    meta.last_refresh = now
                    meta.total_refreshes += 1
                    meta.api_calls = self.client.reset_api_call_count()
                    meta.pool_size = len(pool)
                    self._evict_if_needed()
                finally:
                    self._building.discard(viewer_handle)

        pool = self._candidate_pools[viewer_handle]

        # Filter to fresh (unshown) posts
        shown_info = self.pref_store.get_shown_info(viewer_handle)
        fresh = self._filter_fresh(pool, shown_info)
        meta.fresh_count = len(fresh)

        # If fresh content is low, rebuild with expanded limits
        if len(fresh) < out_limit * 2:
            while viewer_handle in self._building:
                await asyncio.sleep(0.05)
            self._building.add(viewer_handle)
            try:
                new_pool = await self._build_candidate_pool(
                    viewer_handle,
                    expand_limits=True,
                )
                self._candidate_pools[viewer_handle] = new_pool
                pool = new_pool
                meta.last_refresh = time.time()
                meta.total_refreshes += 1
                meta.api_calls = self.client.reset_api_call_count()
                meta.pool_size = len(pool)
                fresh = self._filter_fresh(pool, self.pref_store.get_shown_info(viewer_handle))
                meta.fresh_count = len(fresh)
            finally:
                self._building.discard(viewer_handle)

        # If STILL not enough fresh content, allow one repeat of top-scoring posts
        if len(fresh) < out_limit:
            score_threshold = self._compute_repeat_threshold(pool)
            repeat_candidates = [
                p for p in pool
                if p.uri not in [f.uri for f in fresh]
                and self.pref_store.can_show_uri(viewer_handle, p.uri, p.score, score_threshold)
            ]
            # Mark repeats
            for p in repeat_candidates:
                p.is_repeat = True
            fresh = fresh + repeat_candidates

        out = fresh[:out_limit]

        # Record shown URIs persistently
        self.pref_store.record_shown(viewer_handle, [p.uri for p in out])
        return out

    def _filter_fresh(self, pool: List[CandidatePost], shown_info: Dict[str, Dict]) -> List[CandidatePost]:
        fresh: List[CandidatePost] = []
        for p in pool:
            entry = shown_info.get(p.uri)
            if not entry:
                fresh.append(p)
                continue
            count = entry.get("count", 0)
            if count == 0:
                fresh.append(p)
        return fresh

    def _compute_repeat_threshold(self, pool: List[CandidatePost]) -> float:
        if not pool:
            return float("inf")
        scores = [p.score for p in pool]
        scores.sort(reverse=True)
        # Top 15% of posts are eligible for one repeat
        idx = max(0, int(len(scores) * 0.15) - 1)
        return scores[idx]

    def record_preference(self, viewer_handle: str, author_handle: str, post_text: str, direction: int) -> None:
        self.pref_store.adjust(viewer_handle, author_handle, post_text, direction)
        self._apply_preferences_to_pool(viewer_handle)

    def _apply_preferences_to_pool(self, viewer_handle: str) -> None:
        pool = self._candidate_pools.get(viewer_handle)
        if not pool:
            return
        for p in pool:
            adj = self.pref_store.get_adjustment(viewer_handle, p.author_handle, p.text)
            p.score += adj
            if p.debug_info:
                p.debug_info["preference_adjustment"] = round(adj, 3)
        pool.sort(key=lambda p: p.score, reverse=True)

    # ------------------------------------------------------------------
    # Embed helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_embed(embed: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not embed:
            return None
        etype = embed.get("$type", "")
        out: Dict[str, Any] = {}

        if "images#view" in etype:
            out["images"] = embed.get("images", [])
        elif "external#view" in etype:
            out["external"] = embed.get("external", {})
        elif "recordWithMedia#view" in etype:
            media = embed.get("media", {})
            if "images#view" in media.get("$type", ""):
                out["images"] = media.get("images", [])
            record = embed.get("record", {})
            if record:
                quote = FeedService._normalize_quote(record.get("record", record))
                if quote:
                    out["quote"] = quote
        elif "record#view" in etype:
            quote = FeedService._normalize_quote(embed.get("record", {}))
            if quote:
                out["quote"] = quote

        return out if out else None

    @staticmethod
    def _normalize_quote(record: Dict[str, Any]) -> Dict[str, Any] | None:
        if not record:
            return None
        author = record.get("author", {})
        value = record.get("value", {})
        return {
            "author_handle": author.get("handle", ""),
            "author_display_name": author.get("displayName", ""),
            "author_avatar": author.get("avatar", ""),
            "text": value.get("text", ""),
            "indexed_at": record.get("indexedAt", ""),
        }

    async def _post_from_feed_item(self, item: Dict[str, Any], in_network: bool) -> CandidatePost | None:
        post = item.get("post", {})
        author = post.get("author", {})
        record = post.get("record", {})
        text = record.get("text", "")
        if not text:
            return None

        created_at_raw = record.get("createdAt", "")
        candidate = CandidatePost(
            uri=post.get("uri", ""),
            cid=post.get("cid", ""),
            text=text,
            author_did=author.get("did", ""),
            author_handle=author.get("handle", ""),
            author_avatar=author.get("avatar", ""),
            author_display_name=author.get("displayName", ""),
            indexed_at=parse_bsky_time(post.get("indexedAt", "")),
            created_at=parse_bsky_time(created_at_raw) if created_at_raw else None,
            like_count=post.get("likeCount", 0),
            repost_count=post.get("repostCount", 0),
            reply_count=post.get("replyCount", 0),
            quote_count=post.get("quoteCount", 0),
            in_network=in_network,
            embed=self._normalize_embed(post.get("embed")),
        )

        max_age_seconds = self.settings.max_post_age_hours * 3600
        age_seconds = (datetime.now(timezone.utc) - candidate.indexed_at).total_seconds()
        if age_seconds > max_age_seconds:
            return None

        return candidate

    async def _collect_posts(self, author_ids: List[str], per_author: int, in_network: bool) -> List[CandidatePost]:
        out: List[CandidatePost] = []
        for actor in author_ids:
            try:
                feed = await self.client.get_author_feed(actor, limit=per_author)
            except Exception:
                continue
            for item in feed:
                post = await self._post_from_feed_item(item, in_network=in_network)
                if post:
                    out.append(post)
        return out

    async def _build_oon_authors(self, follows: List[Dict[str, Any]], in_network: Set[str]) -> List[str]:
        seeds = [f.get("did") for f in follows[: self.settings.max_oon_seed_authors] if f.get("did")]
        oon: List[str] = []
        for seed in seeds:
            try:
                secondary = await self.client.get_follows(seed, limit=10)
            except Exception:
                continue
            for f in secondary:
                did = f.get("did")
                if did and did not in in_network and did not in oon:
                    oon.append(did)
                    if len(oon) >= self.settings.max_oon_authors:
                        return oon
        return oon

    async def _build_candidate_pool(self, viewer_handle: str, expand_limits: bool = False) -> List[CandidatePost]:
        viewer_did = await self.client.resolve_handle(viewer_handle)
        profile = await self.client.get_profile(viewer_did)
        follows = await self.client.get_follows(viewer_did, limit=self.settings.max_following_scan)

        liked_uris, author_freq = await self.client.get_actor_likes(viewer_did, limit=self.settings.max_likes_fetch)

        # Cache liked-author boosts
        if author_freq:
            self.pref_store.record_liked_authors(viewer_handle, author_freq)
        liked_author_boosts = self.pref_store.get_liked_author_boosts(viewer_handle)

        in_network_authors = [f.get("did") for f in follows if f.get("did")]
        in_network_set = set(in_network_authors)

        # Expand fetch limits if we're trying to get more fresh content
        in_per_author = self.settings.posts_per_author_in_network * (2 if expand_limits else 1)
        oon_per_author = self.settings.posts_per_author_oon * (2 if expand_limits else 1)
        max_pool = self.settings.max_candidate_pool_size * (2 if expand_limits else 1)

        in_posts = await self._collect_posts(
            in_network_authors,
            per_author=in_per_author,
            in_network=True,
        )

        oon_authors = await self._build_oon_authors(follows, in_network_set)
        oon_posts = await self._collect_posts(
            oon_authors,
            per_author=oon_per_author,
            in_network=False,
        )

        in_posts.sort(key=lambda p: p.indexed_at, reverse=True)
        oon_posts.sort(key=lambda p: p.indexed_at, reverse=True)

        candidates: List[CandidatePost] = []
        seen_uri: Set[str] = set()
        for p in in_posts + oon_posts:
            if not p.uri or p.uri in seen_uri:
                continue
            if p.uri in liked_uris:
                continue
            if p.author_did == viewer_did:
                continue
            seen_uri.add(p.uri)
            p.liked_author_boost = liked_author_boosts.get(p.author_did, 0.0)
            candidates.append(p)
            if len(candidates) >= max_pool:
                break

        interest_parts = [profile.get("description", "")]
        interest_parts.extend(await self.client.get_viewer_recent_posts(viewer_did, limit=8))
        user_interest = "\n".join(x for x in interest_parts if x).strip() or viewer_handle

        max_embed = min(self.settings.max_embed_candidates * (2 if expand_limits else 1), len(candidates))
        embed_subset = candidates[:max_embed]
        vectors = await self.embedder.embed([user_interest] + [p.text for p in embed_subset])
        user_vec = vectors[0]
        for p, vec in zip(embed_subset, vectors[1:]):
            p.semantic_similarity = cosine_similarity(user_vec, vec)

        # Rank with liked-author boosts
        if self.settings.debug_ui:
            ranked = self.ranker.rank_debug(candidates, liked_author_boosts=liked_author_boosts)
        else:
            ranked = self.ranker.rank(candidates, liked_author_boosts=liked_author_boosts)

        # Apply stored topic/preferences
        for p in ranked:
            adj = self.pref_store.get_adjustment(viewer_handle, p.author_handle, p.text)
            p.score += adj
            if p.debug_info:
                p.debug_info["preference_adjustment"] = round(adj, 3)

        ranked.sort(key=lambda p: p.score, reverse=True)
        return ranked
