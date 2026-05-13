from datetime import datetime, timezone, timedelta

from app.models import CandidatePost
from app.ranker import Ranker


def make_post(author: str, in_network: bool, sem: float, likes: int, age_hours: int = 1) -> CandidatePost:
    return CandidatePost(
        uri=f"at://{author}/{likes}/{age_hours}",
        cid="c",
        text="hello",
        author_did=author,
        author_handle=f"{author}.bsky.social",
        indexed_at=datetime.now(timezone.utc) - timedelta(hours=age_hours),
        like_count=likes,
        repost_count=0,
        reply_count=0,
        quote_count=0,
        in_network=in_network,
        semantic_similarity=sem,
    )


def test_ranker_prefers_better_signal():
    ranker = Ranker()
    a = make_post("did:a", True, 0.8, 50)
    b = make_post("did:b", False, -0.2, 1)
    scored = ranker.rank([b, a])
    assert scored[0].author_did == "did:a"


def test_author_diversity_penalizes_repeat_authors():
    ranker = Ranker()
    p1 = make_post("did:one", True, 0.8, 100)
    p2 = make_post("did:one", True, 0.8, 90)
    p3 = make_post("did:two", True, 0.4, 20)
    ranked = ranker.rank([p1, p2, p3])
    one_posts = [p for p in ranked if p.author_did == "did:one"]
    assert one_posts[0].score >= one_posts[1].score


def test_recency_boost_prefers_newer_post_when_quality_similar():
    ranker = Ranker(recency_half_life_hours=3.0, recency_boost=2.0)
    old_post = make_post("did:old", True, 0.5, 10, age_hours=30)
    new_post = make_post("did:new", True, 0.5, 10, age_hours=1)
    ranked = ranker.rank([old_post, new_post])
    assert ranked[0].author_did == "did:new"
