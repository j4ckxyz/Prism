from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, List

from .models import CandidatePost


@dataclass
class Weights:
    favorite: float = 1.0
    reply: float = 2.0
    repost: float = 2.5
    quote: float = 2.0
    click: float = 0.5
    follow_author: float = 3.0
    dwell: float = 0.25
    not_interested: float = -3.0
    block_author: float = -6.0
    mute_author: float = -4.0
    report: float = -8.0


class Ranker:
    def __init__(
        self,
        weights: Weights | None = None,
        oon_factor: float = 0.92,
        recency_half_life_hours: float = 6.0,
        recency_boost: float = 1.25,
    ):
        self.weights = weights or Weights()
        self.oon_factor = oon_factor
        self.recency_half_life_hours = max(recency_half_life_hours, 0.5)
        self.recency_boost = max(recency_boost, 0.0)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _recency(self, post: CandidatePost) -> float:
        now = datetime.now(timezone.utc)
        age_hours = max((now - post.indexed_at).total_seconds() / 3600.0, 0.0)
        return 2.0 ** (-age_hours / self.recency_half_life_hours)

    def _engagement_signal(self, post: CandidatePost) -> float:
        raw = post.like_count + 2.0 * post.repost_count + 1.2 * post.reply_count + 1.5 * post.quote_count
        return math.log1p(max(raw, 0.0)) / 4.0

    def score_one(self, post: CandidatePost, liked_author_boost: float = 0.0) -> float:
        affinity = 1.0 if post.in_network else 0.0
        semantic = (post.semantic_similarity + 1.0) / 2.0
        recency = self._recency(post)
        engagement = self._engagement_signal(post)

        p_favorite = self._sigmoid(-1.0 + 1.4 * engagement + 0.7 * affinity + 0.6 * semantic + 0.4 * recency)
        p_reply = self._sigmoid(-1.6 + 1.2 * engagement + 0.4 * affinity + 0.6 * semantic)
        p_repost = self._sigmoid(-1.4 + 1.3 * engagement + 0.5 * affinity + 0.5 * semantic)
        p_quote = self._sigmoid(-1.8 + 1.1 * engagement + 0.5 * semantic)
        p_click = self._sigmoid(-1.0 + 0.8 * semantic + 0.4 * recency)
        p_follow_author = self._sigmoid(-2.6 + 0.8 * semantic + (0.0 if affinity else 0.7))
        p_dwell = self._sigmoid(-0.8 + 0.9 * semantic + 0.3 * recency)

        p_not_interested = self._sigmoid(-2.2 + 1.4 * (1 - semantic) + 0.6 * (1 - affinity))
        p_block_author = self._sigmoid(-4.0 + 1.8 * (1 - semantic))
        p_mute_author = self._sigmoid(-3.5 + 1.5 * (1 - semantic))
        p_report = self._sigmoid(-5.0 + 1.7 * (1 - semantic))

        w = self.weights
        s = (
            w.favorite * p_favorite
            + w.reply * p_reply
            + w.repost * p_repost
            + w.quote * p_quote
            + w.click * p_click
            + w.follow_author * p_follow_author
            + w.dwell * p_dwell
            + w.not_interested * p_not_interested
            + w.block_author * p_block_author
            + w.mute_author * p_mute_author
            + w.report * p_report
        )
        s *= (1.0 + self.recency_boost * recency)

        if not post.in_network:
            s *= self.oon_factor

        # Boost posts from authors the viewer frequently likes
        s += liked_author_boost

        return s

    def score_one_debug(self, post: CandidatePost, liked_author_boost: float = 0.0) -> Dict[str, Any]:
        """Return full scoring breakdown for debugging."""
        affinity = 1.0 if post.in_network else 0.0
        semantic = (post.semantic_similarity + 1.0) / 2.0
        recency = self._recency(post)
        engagement = self._engagement_signal(post)

        p_favorite = self._sigmoid(-1.0 + 1.4 * engagement + 0.7 * affinity + 0.6 * semantic + 0.4 * recency)
        p_reply = self._sigmoid(-1.6 + 1.2 * engagement + 0.4 * affinity + 0.6 * semantic)
        p_repost = self._sigmoid(-1.4 + 1.3 * engagement + 0.5 * affinity + 0.5 * semantic)
        p_quote = self._sigmoid(-1.8 + 1.1 * engagement + 0.5 * semantic)
        p_click = self._sigmoid(-1.0 + 0.8 * semantic + 0.4 * recency)
        p_follow_author = self._sigmoid(-2.6 + 0.8 * semantic + (0.0 if affinity else 0.7))
        p_dwell = self._sigmoid(-0.8 + 0.9 * semantic + 0.3 * recency)

        p_not_interested = self._sigmoid(-2.2 + 1.4 * (1 - semantic) + 0.6 * (1 - affinity))
        p_block_author = self._sigmoid(-4.0 + 1.8 * (1 - semantic))
        p_mute_author = self._sigmoid(-3.5 + 1.5 * (1 - semantic))
        p_report = self._sigmoid(-5.0 + 1.7 * (1 - semantic))

        w = self.weights
        raw_score = (
            w.favorite * p_favorite
            + w.reply * p_reply
            + w.repost * p_repost
            + w.quote * p_quote
            + w.click * p_click
            + w.follow_author * p_follow_author
            + w.dwell * p_dwell
            + w.not_interested * p_not_interested
            + w.block_author * p_block_author
            + w.mute_author * p_mute_author
            + w.report * p_report
        )
        recency_boosted = raw_score * (1.0 + self.recency_boost * recency)
        after_oon = recency_boosted * (1.0 if post.in_network else self.oon_factor)
        final_score = after_oon + liked_author_boost

        return {
            "inputs": {
                "affinity": round(affinity, 3),
                "semantic_similarity": round(post.semantic_similarity, 3),
                "semantic_normalized": round(semantic, 3),
                "recency": round(recency, 3),
                "engagement": round(engagement, 3),
                "liked_author_boost": round(liked_author_boost, 3),
                "likes": post.like_count,
                "reposts": post.repost_count,
                "replies": post.reply_count,
                "quotes": post.quote_count,
            },
            "probabilities": {
                "favorite": round(p_favorite, 3),
                "reply": round(p_reply, 3),
                "repost": round(p_repost, 3),
                "quote": round(p_quote, 3),
                "click": round(p_click, 3),
                "follow_author": round(p_follow_author, 3),
                "dwell": round(p_dwell, 3),
                "not_interested": round(p_not_interested, 3),
                "block_author": round(p_block_author, 3),
                "mute_author": round(p_mute_author, 3),
                "report": round(p_report, 3),
            },
            "score_breakdown": {
                "raw_score": round(raw_score, 3),
                "recency_boost": round(1.0 + self.recency_boost * recency, 3),
                "recency_boosted": round(recency_boosted, 3),
                "oon_factor": 1.0 if post.in_network else self.oon_factor,
                "after_oon": round(after_oon, 3),
                "liked_author_boost": round(liked_author_boost, 3),
                "final_score": round(final_score, 3),
            },
        }

    def apply_author_diversity(self, posts: List[CandidatePost], decay: float = 0.85, floor: float = 0.6) -> None:
        posts.sort(key=lambda p: p.score, reverse=True)
        seen: Dict[str, int] = defaultdict(int)
        for p in posts:
            k = seen[p.author_did]
            seen[p.author_did] += 1
            mult = (1.0 - floor) * (decay ** k) + floor
            p.score *= mult
            if p.debug_info:
                p.debug_info["diversity_penalty"] = round(mult, 3)

    def rank(self, posts: Iterable[CandidatePost], liked_author_boosts: Dict[str, float] | None = None) -> List[CandidatePost]:
        boosts = liked_author_boosts or {}
        ranked = list(posts)
        for p in ranked:
            p.score = self.score_one(p, liked_author_boost=boosts.get(p.author_did, 0.0))
        self.apply_author_diversity(ranked)
        ranked.sort(key=lambda p: p.score, reverse=True)
        return ranked

    def rank_debug(self, posts: Iterable[CandidatePost], liked_author_boosts: Dict[str, float] | None = None) -> List[CandidatePost]:
        boosts = liked_author_boosts or {}
        ranked = list(posts)
        for p in ranked:
            boost = boosts.get(p.author_did, 0.0)
            p.score = self.score_one(p, liked_author_boost=boost)
            p.debug_info = self.score_one_debug(p, liked_author_boost=boost)
        self.apply_author_diversity(ranked)
        ranked.sort(key=lambda p: p.score, reverse=True)
        return ranked
