from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .bluesky_client import BlueskyClient
from .config import load_settings
from .embedding import OllamaEmbedder
from .feed_service import FeedService
from .ranker import Ranker


settings = load_settings()
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _relative_time(dt: datetime | None) -> str:
    if not dt:
        return ""
    now = datetime.now(timezone.utc)
    diff = now - dt
    seconds = diff.total_seconds()
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h"
    if seconds < 604800:
        return f"{int(seconds / 86400)}d"
    return dt.strftime("%b %d")


templates.env.filters["relative_time"] = _relative_time


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = BlueskyClient(
        pds_url=settings.pds_url,
        appview_url=settings.appview_url,
        handle=settings.bsky_handle,
        app_password=settings.bsky_app_password,
        cache_ttl_resolve_seconds=settings.cache_ttl_resolve_seconds,
        cache_ttl_profile_seconds=settings.cache_ttl_profile_seconds,
        cache_ttl_follows_seconds=settings.cache_ttl_follows_seconds,
        cache_ttl_author_feed_seconds=settings.cache_ttl_author_feed_seconds,
        cache_ttl_likes_seconds=settings.cache_ttl_likes_seconds,
        local_mode=settings.local_mode,
    )
    await client.login()
    embedder = OllamaEmbedder(settings.ollama_url, settings.ollama_model)
    ranker = Ranker(
        recency_half_life_hours=settings.recency_half_life_hours,
        recency_boost=settings.recency_boost,
    )
    service = FeedService(client=client, ranker=ranker, embedder=embedder, settings=settings)
    service.start_background_refresh()

    app.state.client = client
    app.state.embedder = embedder
    app.state.service = service
    yield

    service.stop_background_refresh()
    await embedder.close()
    await client.close()


app = FastAPI(title="Prism", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "posts": [], "handle": ""},
    )


@app.post("/feed", response_class=HTMLResponse)
async def render_feed(
    request: Request,
    handle: str = Form(...),
    limit: int = Form(30),
    refresh: str = Form(""),
):
    posts = []
    error = ""
    cycle = refresh in ("1", "on", "true")
    try:
        posts = await request.app.state.service.build_feed(handle, limit=limit, cycle=cycle)
    except Exception as e:
        error = str(e)

    prefs = request.app.state.service.pref_store.get_summary(handle)
    liked_authors = request.app.state.service.pref_store.get_liked_author_freq(handle)
    pool_meta = request.app.state.service.get_pool_meta(handle)
    pool_age = None
    api_calls = None
    pool_size = None
    fresh_count = None
    if pool_meta:
        if pool_meta.last_refresh:
            pool_age = _relative_time(datetime.fromtimestamp(pool_meta.last_refresh, tz=timezone.utc))
        api_calls = pool_meta.api_calls
        pool_size = pool_meta.pool_size
        fresh_count = pool_meta.fresh_count

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "posts": posts,
            "handle": handle,
            "limit": limit,
            "error": error,
            "prefs": prefs,
            "liked_authors": liked_authors,
            "pool_age": pool_age,
            "api_calls": api_calls,
            "pool_size": pool_size,
            "fresh_count": fresh_count,
            "debug_ui": request.app.state.service.settings.debug_ui,
            "local_mode": request.app.state.service.settings.local_mode,
        },
    )


@app.post("/preference")
async def record_preference(
    request: Request,
    viewer_handle: str = Form(...),
    author_handle: str = Form(...),
    post_text: str = Form(""),
    direction: int = Form(...),
):
    try:
        request.app.state.service.record_preference(viewer_handle, author_handle, post_text, direction)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
