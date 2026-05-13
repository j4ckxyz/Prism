import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.feed_service import FeedService
from app.ranker import Ranker


class FakeClient:
    async def resolve_handle(self, handle: str) -> str:
        return "did:viewer"

    async def get_profile(self, actor: str):
        return {"description": "tech, ai, design"}

    async def get_follows(self, actor: str, limit: int = 100):
        if actor == "did:viewer":
            return [{"did": "did:in1"}, {"did": "did:in2"}]
        return [{"did": "did:oon1"}, {"did": "did:oon2"}]

    async def get_author_feed(self, actor: str, limit: int = 10):
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return [
            {
                "post": {
                    "uri": f"at://{actor}/post1",
                    "cid": "cid1",
                    "indexedAt": now,
                    "author": {"did": actor, "handle": f"{actor}.bsky.social"},
                    "record": {"text": f"post by {actor}"},
                    "likeCount": 5,
                    "replyCount": 1,
                    "repostCount": 2,
                    "quoteCount": 0,
                }
            }
        ]

    async def get_viewer_recent_posts(self, actor: str, limit: int = 8):
        return ["ai systems", "ranking"]

    async def get_actor_likes(self, actor: str, limit: int = 200):
        return (set(), {})

    def reset_api_call_count(self) -> int:
        return 0


class FakeClientWithLikes(FakeClient):
    async def get_actor_likes(self, actor: str, limit: int = 200):
        return ({"at://did:in1/post1"}, {"did:in1": 1})


class FakeClientWithOldPosts(FakeClient):
    async def get_author_feed(self, actor: str, limit: int = 10):
        recent = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        old = (datetime.now(timezone.utc) - timedelta(hours=120)).isoformat().replace("+00:00", "Z")
        return [
            {
                "post": {
                    "uri": f"at://{actor}/recent",
                    "cid": "cid-recent",
                    "indexedAt": recent,
                    "author": {"did": actor, "handle": f"{actor}.bsky.social"},
                    "record": {"text": f"recent by {actor}"},
                    "likeCount": 5,
                    "replyCount": 1,
                    "repostCount": 2,
                    "quoteCount": 0,
                }
            },
            {
                "post": {
                    "uri": f"at://{actor}/old",
                    "cid": "cid-old",
                    "indexedAt": old,
                    "author": {"did": actor, "handle": f"{actor}.bsky.social"},
                    "record": {"text": f"old by {actor}"},
                    "likeCount": 100,
                    "replyCount": 10,
                    "repostCount": 10,
                    "quoteCount": 4,
                }
            },
        ]


class FakeEmbedder:
    async def embed(self, texts):
        return [[1.0, 0.0]] + [[1.0, 0.0] for _ in texts[1:]]


@dataclass
class FakeSettings:
    max_following_scan: int = 10
    posts_per_author_in_network: int = 2
    posts_per_author_oon: int = 1
    max_oon_seed_authors: int = 2
    max_oon_authors: int = 2
    output_size: int = 10
    max_post_age_hours: int = 72
    max_candidate_pool_size: int = 100
    max_embed_candidates: int = 20
    preference_store_path: str = "/tmp/test_prefs.json"
    max_likes_fetch: int = 200
    background_refresh_interval_seconds: int = 30
    max_concurrent_refreshes: int = 3
    max_pools_in_memory: int = 200
    local_mode: bool = False
    debug_ui: bool = False


def _make_settings():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return FakeSettings(preference_store_path=path)


@pytest.mark.asyncio
async def test_build_feed_returns_ranked_posts():
    settings = _make_settings()
    service = FeedService(
        client=FakeClient(),
        ranker=Ranker(),
        embedder=FakeEmbedder(),
        settings=settings,
    )
    posts = await service.build_feed("viewer.bsky.social", limit=5)
    assert len(posts) > 0
    assert all(p.text for p in posts)
    assert posts == sorted(posts, key=lambda p: p.score, reverse=True)
    os.remove(settings.preference_store_path)


@pytest.mark.asyncio
async def test_second_call_serves_different_fresh_posts():
    call_count = 0

    class CountingClient(FakeClient):
        async def get_author_feed(self, actor, limit=10):
            nonlocal call_count
            call_count += 1
            return await super().get_author_feed(actor, limit)

    settings = _make_settings()
    service = FeedService(
        client=CountingClient(),
        ranker=Ranker(),
        embedder=FakeEmbedder(),
        settings=settings,
    )
    posts1 = await service.build_feed("viewer.bsky.social", limit=2)
    uris1 = {p.uri for p in posts1}
    calls_after_first = call_count

    # Second call: pool is low on fresh content, so it triggers an expanded rebuild
    posts2 = await service.build_feed("viewer.bsky.social", limit=2)
    uris2 = {p.uri for p in posts2}
    calls_after_second = call_count

    assert len(posts1) > 0
    assert len(posts2) > 0
    # Expanded rebuild makes extra API calls to find fresh content
    assert calls_after_second > calls_after_first
    # Posts should be different (fresh tracking)
    assert len(uris1 & uris2) == 0 or len(posts1) + len(posts2) > len(set(p.uri for p in service._candidate_pools["viewer.bsky.social"]))
    os.remove(settings.preference_store_path)


@pytest.mark.asyncio
async def test_liked_posts_are_filtered_out():
    settings = _make_settings()
    service = FeedService(
        client=FakeClientWithLikes(),
        ranker=Ranker(),
        embedder=FakeEmbedder(),
        settings=settings,
    )
    posts = await service.build_feed("viewer.bsky.social", limit=20)
    assert len(posts) > 0
    assert all("post by did:in1" not in p.text for p in posts)
    os.remove(settings.preference_store_path)


@pytest.mark.asyncio
async def test_old_posts_are_filtered_by_age():
    settings = _make_settings()
    service = FeedService(
        client=FakeClientWithOldPosts(),
        ranker=Ranker(),
        embedder=FakeEmbedder(),
        settings=settings,
    )
    posts = await service.build_feed("viewer.bsky.social", limit=20)
    assert len(posts) > 0
    assert all("old by" not in p.text for p in posts)
    os.remove(settings.preference_store_path)


@pytest.mark.asyncio
async def test_liked_author_boost_applied():
    settings = _make_settings()
    # FakeClientWithLikes says did:in1 has 1 liked post
    service = FeedService(
        client=FakeClientWithLikes(),
        ranker=Ranker(),
        embedder=FakeEmbedder(),
        settings=settings,
    )
    posts = await service.build_feed("viewer.bsky.social", limit=20)
    # did:in1's post was filtered because it was liked, so we can't see the boost directly
    # but we can verify the liked_author_boosts were stored
    boosts = service.pref_store.get_liked_author_boosts("viewer.bsky.social")
    assert "did:in1" in boosts
    assert boosts["did:in1"] > 0
    os.remove(settings.preference_store_path)
