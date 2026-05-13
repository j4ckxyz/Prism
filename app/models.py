from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class CandidatePost:
    uri: str
    cid: str
    text: str
    author_did: str
    author_handle: str
    indexed_at: datetime
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    in_network: bool = False
    semantic_similarity: float = 0.0
    score: float = 0.0
    author_avatar: str = ""
    author_display_name: str = ""
    created_at: Optional[datetime] = None
    embed: Optional[Dict[str, Any]] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
    liked_author_boost: float = 0.0
    is_repeat: bool = False


def parse_bsky_time(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
