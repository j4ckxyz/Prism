from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass
class Settings:
    bsky_handle: str
    bsky_app_password: str
    pds_url: str
    appview_url: str
    ollama_url: str
    ollama_model: str

    max_following_scan: int = 20
    posts_per_author_in_network: int = 8
    posts_per_author_oon: int = 3
    max_oon_seed_authors: int = 5
    max_oon_authors: int = 20
    output_size: int = 50

    # realtime tuning (Thunder-like recency behavior)
    max_post_age_hours: int = 72
    recency_half_life_hours: float = 6.0
    recency_boost: float = 1.25

    # efficiency/caching
    max_candidate_pool_size: int = 800
    max_embed_candidates: int = 80
    cache_ttl_resolve_seconds: int = 3600
    cache_ttl_profile_seconds: int = 300
    cache_ttl_follows_seconds: int = 45
    cache_ttl_author_feed_seconds: int = 20
    cache_ttl_likes_seconds: int = 60
    candidate_pool_ttl_seconds: int = 120

    preference_store_path: str = "prism/data/preferences.json"
    max_likes_fetch: int = 200

    # background pool warming
    background_refresh_interval_seconds: int = 30
    max_concurrent_refreshes: int = 3
    max_pools_in_memory: int = 200

    # local dev mode: disables rate limits, enables debug UI, longer caches
    local_mode: bool = False
    debug_ui: bool = False



def load_settings() -> Settings:
    load_dotenv()
    local_mode = os.getenv("LOCAL_MODE", "").lower() in ("1", "true", "yes", "on")
    debug_ui = os.getenv("DEBUG_UI", "").lower() in ("1", "true", "yes", "on") or local_mode

    return Settings(
        bsky_handle=os.getenv("BSKY_HANDLE", ""),
        bsky_app_password=os.getenv("BSKY_APP_PASSWORD", ""),
        pds_url=os.getenv("PDS_URL", "https://bsky.social"),
        appview_url=os.getenv("APPVIEW_URL", "https://public.api.bsky.app"),
        ollama_url=os.getenv("OLLAMA_URL", "http://100.97.0.44:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3-embedding:0.6b"),
        max_following_scan=int(os.getenv("MAX_FOLLOWING_SCAN", "20")),
        posts_per_author_in_network=int(os.getenv("POSTS_PER_AUTHOR_IN_NETWORK", "8")),
        posts_per_author_oon=int(os.getenv("POSTS_PER_AUTHOR_OON", "3")),
        max_oon_seed_authors=int(os.getenv("MAX_OON_SEED_AUTHORS", "5")),
        max_oon_authors=int(os.getenv("MAX_OON_AUTHORS", "20")),
        output_size=int(os.getenv("OUTPUT_SIZE", "50")),
        max_post_age_hours=int(os.getenv("MAX_POST_AGE_HOURS", "72")),
        recency_half_life_hours=float(os.getenv("RECENCY_HALF_LIFE_HOURS", "6")),
        recency_boost=float(os.getenv("RECENCY_BOOST", "1.25")),
        max_candidate_pool_size=int(os.getenv("MAX_CANDIDATE_POOL_SIZE", "800")),
        max_embed_candidates=int(os.getenv("MAX_EMBED_CANDIDATES", "80")),
        cache_ttl_resolve_seconds=int(os.getenv("CACHE_TTL_RESOLVE_SECONDS", "3600")),
        cache_ttl_profile_seconds=int(os.getenv("CACHE_TTL_PROFILE_SECONDS", "300")),
        cache_ttl_follows_seconds=int(os.getenv("CACHE_TTL_FOLLOWS_SECONDS", "45")),
        cache_ttl_author_feed_seconds=int(os.getenv("CACHE_TTL_AUTHOR_FEED_SECONDS", "20")),
        cache_ttl_likes_seconds=int(os.getenv("CACHE_TTL_LIKES_SECONDS", "60")),
        candidate_pool_ttl_seconds=int(os.getenv("CANDIDATE_POOL_TTL_SECONDS", "120")),
        preference_store_path=os.getenv("PREFERENCE_STORE_PATH", "prism/data/preferences.json"),
        max_likes_fetch=int(os.getenv("MAX_LIKES_FETCH", "200")),
        background_refresh_interval_seconds=int(os.getenv("BACKGROUND_REFRESH_INTERVAL_SECONDS", "30")),
        max_concurrent_refreshes=int(os.getenv("MAX_CONCURRENT_REFRESHES", "3")),
        max_pools_in_memory=int(os.getenv("MAX_POOLS_IN_MEMORY", "200")),
        local_mode=local_mode,
        debug_ui=debug_ui,
    )
