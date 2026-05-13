from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

_STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "don", "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves", "com", "https", "http",
    "www", "like", "get", "one", "two", "way", "new", "time", "want", "make",
    "think", "know", "see", "go", "come", "take", "use", "work", "good", "back",
    "year", "look", "say", "said", "man", "day", "even", "right", "old", "too",
    "still", "well", "say", "much", "little", "long", "great", "world", "hand",
    "part", "place", "made", "live", "where", "after", "again", "around", "every",
    "here", "there", "through", "when", "where", "being", "every", "having",
    "will", "shall", "might", "must", "need", "used", "upon", "such", "only",
    "over", "many", "then", "them", "these", "than", "well", "were",
}


def _extract_topics(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[a-z]{4,}", text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(5)]


class PreferenceStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data: Dict = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    # ------------------------------------------------------------------
    # Topic / author preference adjustments
    # ------------------------------------------------------------------
    def adjust(self, viewer_handle: str, author_handle: str, post_text: str, direction: int) -> None:
        topics = _extract_topics(post_text)
        viewer_prefs = self.data.setdefault(viewer_handle, {})
        author_prefs = viewer_prefs.setdefault("authors", {}).setdefault(author_handle, {})
        global_prefs = viewer_prefs.setdefault("topics", {})

        delta = direction * 0.5
        for topic in topics:
            author_prefs[topic] = max(-3.0, min(3.0, author_prefs.get(topic, 0.0) + delta))
            global_prefs[topic] = max(-3.0, min(3.0, global_prefs.get(topic, 0.0) + delta))
        self.save()

    def get_adjustment(self, viewer_handle: str, author_handle: str, post_text: str) -> float:
        topics = _extract_topics(post_text)
        viewer_prefs = self.data.get(viewer_handle, {})
        author_prefs = viewer_prefs.get("authors", {}).get(author_handle, {})
        global_prefs = viewer_prefs.get("topics", {})

        adjustment = 0.0
        for topic in topics:
            adjustment += author_prefs.get(topic, 0.0) + global_prefs.get(topic, 0.0)
        return adjustment

    def get_summary(self, viewer_handle: str) -> Dict:
        viewer_prefs = self.data.get(viewer_handle, {})
        return {
            "topics": viewer_prefs.get("topics", {}),
            "authors": viewer_prefs.get("authors", {}),
        }

    # ------------------------------------------------------------------
    # Shown-post tracking (prevents repeats)
    # ------------------------------------------------------------------
    def record_shown(self, viewer_handle: str, uris: List[str]) -> None:
        viewer_prefs = self.data.setdefault(viewer_handle, {})
        shown = viewer_prefs.setdefault("shown_uris", {})
        now = time.time()
        for uri in uris:
            entry = shown.get(uri, {"count": 0, "first_shown": now})
            entry["count"] = entry.get("count", 0) + 1
            entry["last_shown"] = now
            shown[uri] = entry
        self.save()

    def get_shown_info(self, viewer_handle: str) -> Dict[str, Dict]:
        viewer_prefs = self.data.get(viewer_handle, {})
        return viewer_prefs.get("shown_uris", {})

    def prune_shown(self, viewer_handle: str, max_age_days: float = 7.0, max_entries: int = 5000) -> int:
        """Remove old shown entries. Returns number pruned."""
        viewer_prefs = self.data.get(viewer_handle, {})
        shown = viewer_prefs.get("shown_uris", {})
        if not shown:
            return 0

        cutoff = time.time() - (max_age_days * 86400)
        to_remove = [uri for uri, entry in shown.items() if entry.get("last_shown", 0) < cutoff]

        # If still too many, remove oldest by first_shown
        if len(shown) - len(to_remove) > max_entries:
            remaining = [(uri, entry) for uri, entry in shown.items() if uri not in to_remove]
            remaining.sort(key=lambda x: x[1].get("first_shown", 0))
            excess = len(remaining) - max_entries
            to_remove.extend(uri for uri, _ in remaining[:excess])

        for uri in to_remove:
            shown.pop(uri, None)

        if to_remove:
            self.save()
        return len(to_remove)

    def can_show_uri(self, viewer_handle: str, uri: str, score: float, score_threshold_for_repeat: float) -> bool:
        """Returns True if this URI can be shown. Top-scoring posts get max 2 shows, everything else max 1."""
        shown = self.get_shown_info(viewer_handle)
        entry = shown.get(uri)
        if not entry:
            return True
        count = entry.get("count", 0)
        if count >= 1 and score >= score_threshold_for_repeat:
            return count < 2  # top posts get 2 max
        return count < 1  # everything else gets 1 max

    # ------------------------------------------------------------------
    # Liked-author affinity (boost posts from authors the viewer likes)
    # ------------------------------------------------------------------
    def record_liked_authors(self, viewer_handle: str, author_freq: Dict[str, int]) -> Dict[str, float]:
        """Cache liked-author frequency map and return boost values."""
        viewer_prefs = self.data.setdefault(viewer_handle, {})
        boosts: Dict[str, float] = {}
        for author_did, count in author_freq.items():
            # Log-scaled boost: 3 likes → ~0.5 boost, 10 likes → ~0.9 boost
            boost = min(1.5, 0.25 * (count ** 0.5))
            boosts[author_did] = boost
        viewer_prefs["liked_author_boosts"] = boosts
        viewer_prefs["liked_author_freq"] = author_freq
        viewer_prefs["liked_authors_updated"] = time.time()
        self.save()
        return boosts

    def get_liked_author_boosts(self, viewer_handle: str) -> Dict[str, float]:
        viewer_prefs = self.data.get(viewer_handle, {})
        return viewer_prefs.get("liked_author_boosts", {})

    def get_liked_author_freq(self, viewer_handle: str) -> Dict[str, int]:
        viewer_prefs = self.data.get(viewer_handle, {})
        return viewer_prefs.get("liked_author_freq", {})
