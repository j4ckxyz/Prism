# Prism

A port of X/Twitter's "For You" algorithm to Bluesky, built as a local webapp for experimenting with personalized feed ranking. This project adapts the scoring framework from [xai-org/x-algorithm](https://github.com/xai-org/x-algorithm) to work with Bluesky's AT Protocol and public AppView APIs.

> **Status:** Personal research tool. Not a production AT Protocol feed generator (yet).

---

## What this is

X's "For You" feed ranks posts by predicting a set of user actions (favorite, reply, repost, quote, click, dwell, follow author, not interested, block, mute, report) and weighting them. Prism ports that same philosophy to Bluesky:

1. **Candidate generation** — fetch posts from your follows (in-network) and their follows (out-of-network)
2. **Embedding** — compute semantic similarity between your profile/recent posts and each candidate via Ollama
3. **Ranking** — score each post using a probabilistic model tuned to Bluesky's engagement signals
4. **Filtering** — remove already-liked posts, apply author diversity, topic preferences, and freshness rules

The result is a ranked feed that surfaces posts you're likely to engage with, mixing in-network and out-of-network content.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  BlueskyClient  │────▶│  CandidatePool   │────▶│  OllamaEmbedder │
│  (AT Protocol)  │     │  (collect+filter)│     │  (semantic sim) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
   resolve_handle          author feeds              cosine_similarity
   get_profile             follows graph             user vs post vectors
   get_follows             liked posts filtering
   get_author_feed
   get_actor_likes
```

The **Ranker** then scores each candidate using the probabilistic engagement model below.

---

## Ranking model

The scorer predicts the probability of each engagement type using sigmoid functions, then combines them with weights.

### Input features per post

| Feature | Description |
|---------|-------------|
| `affinity` | 1.0 if author is in-network (you follow them), 0.0 otherwise |
| `semantic_similarity` | Cosine similarity between your interest vector and post text vector (-1 to 1) |
| `recency` | Half-life decay: `2^(-age_hours / recency_half_life_hours)` |
| `engagement` | Log-scaled signal: `log1p(likes + 2·reposts + 1.2·replies + 1.5·quotes) / 4` |

### Engagement probabilities (sigmoid)

```
p(favorite)      = σ(-1.0 + 1.4·engagement + 0.7·affinity + 0.6·semantic + 0.4·recency)
p(reply)         = σ(-1.6 + 1.2·engagement + 0.4·affinity + 0.6·semantic)
p(repost)        = σ(-1.4 + 1.3·engagement + 0.5·affinity + 0.5·semantic)
p(quote)         = σ(-1.8 + 1.1·engagement + 0.5·semantic)
p(click)         = σ(-1.0 + 0.8·semantic + 0.4·recency)
p(follow_author) = σ(-2.6 + 0.8·semantic + (0.0 if affinity else 0.7))
p(dwell)         = σ(-0.8 + 0.9·semantic + 0.3·recency)

p(not_interested)= σ(-2.2 + 1.4·(1-semantic) + 0.6·(1-affinity))
p(block_author)  = σ(-4.0 + 1.8·(1-semantic))
p(mute_author)   = σ(-3.5 + 1.5·(1-semantic))
p(report)        = σ(-5.0 + 1.7·(1-semantic))
```

### Weighted score

```
raw_score = Σ weight_i × p(action_i)
          = 1.0·p(fav) + 2.0·p(reply) + 2.5·p(repost) + 2.0·p(quote)
            + 0.5·p(click) + 3.0·p(follow) + 0.25·p(dwell)
            - 3.0·p(not_interested) - 6.0·p(block) - 4.0·p(mute) - 8.0·p(report)

score = raw_score × (1 + recency_boost × recency)

if out-of-network:
    score *= oon_factor  (default 0.92)
```

### Author diversity penalty

After initial scoring, repeat authors from the same feed are penalized multiplicatively:

```
penalty_k = (1 - floor) × decay^k + floor
# k = 0 for first post by author, 1 for second, etc.
# default decay=0.85, floor=0.6
```

### Liked-author affinity boost

The system analyzes which authors you like most frequently and adds a direct score boost to their posts:

```
boost = min(1.5, 0.25 × √(like_count_to_author))
```

This is cached per-user and refreshed with your likes.

---

## Key features

### Always-fresh feed

Posts are tracked in a persistent "shown" store. By default, a post is shown **once and never again**. Top-scoring posts (top 15% of the pool) may appear **at most twice** and are marked with a `↻ repeat` badge.

When fresh content runs low, the system automatically expands fetch limits (2× more posts per author, 2× larger pool) to pull in new candidates before ever showing a repeat.

### Topic preferences (+/-)

Each post has a `⋯` menu with **"Show more like this"** and **"Show less like this"**. These extract keywords from the post text and adjust per-topic and per-author weights stored in `data/preferences.json`:

```
topic_weight += direction × 0.5   # direction = +1 or -1
topic_weight clamped to [-3.0, +3.0]
```

Preferences persist between runs and immediately re-rank the existing candidate pool.

### Smart background warming (production mode)

When not in local mode, a background task refreshes candidate pools on a schedule tuned to user activity:

| Last seen | Refresh interval |
|-----------|------------------|
| < 10 min  | Every 2 min      |
| < 1 hour  | Every 10 min     |
| < 1 day   | Every 1 hour     |
| < 1 week  | Every 6 hours    |
| > 1 week  | Dormant (none)   |

### Local development mode

Set `LOCAL_MODE=true` in your environment to:

- Disable API rate limiting
- Disable background refresh
- Use aggressive caching (10-min feeds, 1-hour profiles)
- Enable debug UI automatically

---

## Differences from x-algorithm

| Aspect | x-algorithm (X/Twitter) | This port (Bluesky) |
|--------|------------------------|---------------------|
| **Data source** | Internal X graphs + feature stores | Public AT Protocol APIs only |
| **Candidate gen** | Heavy ML retrieval (HeavyRanker) | Follows graph + OON expansion |
| **Embeddings** | Proprietary | Ollama (`qwen3-embedding:0.6b`) |
| **Realtime** | Sub-second recency scoring | Half-life decay with configurable boost |
| **Actions** | Favorite, reply, repost, quote, click, dwell, follow, not interested, block, mute, report | Same set, mapped to Bluesky counts |
| **Liked-author affinity** | Likely implicit in HeavyRanker | Explicit boost from `getActorLikes` |
| **Freshness** | Session-level dedup | Persistent per-user shown tracking |
| **Topic prefs** | Likely implicit | Explicit +/- keyword adjustments |
| **Scale** | Billions of users | Single-instance, ~200 pools in memory |

---

## Quick start

### Prerequisites

- Python 3.12+
- A running Ollama instance with an embedding model (default: `qwen3-embedding:0.6b`)
- Bluesky account (for API auth)

### Install

```bash
cd Prism
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Environment variables

Create a `.env` file (not committed):

```bash
BSKY_HANDLE=yourhandle.bsky.social
BSKY_APP_PASSWORD=your-app-password
PDS_URL=https://bsky.social
APPVIEW_URL=https://public.api.bsky.app
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-embedding:0.6b

# Optional tuning
MAX_POST_AGE_HOURS=72
RECENCY_HALF_LIFE_HOURS=6
RECENCY_BOOST=1.25
MAX_CANDIDATE_POOL_SIZE=800
MAX_EMBED_CANDIDATES=80
LOCAL_MODE=false
DEBUG_UI=false
```

> **Note:** `BSKY_APP_PASSWORD` is **not** your account password. Generate one at [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords).

### Run

```bash
source .env
source .venv/bin/activate
uvicorn app.main:app --reload
```

Or use the helper script:

```bash
./run_local.sh
```

Open `http://127.0.0.1:8000`, enter a Bluesky handle, and explore the ranked feed.

---

## Debug UI

When `DEBUG_UI=true` (or `LOCAL_MODE=true`), every post card has a **🔍 Why this post?** toggle that reveals:

- Input features (affinity, semantic similarity, recency, engagement)
- All predicted action probabilities
- Step-by-step score breakdown (raw → recency boost → OON factor → final)
- Preference adjustments and diversity penalties

The toolbar also shows pool stats: API call count, pool size, and fresh-post count.

---

## Tests

```bash
pytest tests -v
```

All tests mock the Bluesky client and Ollama embedder — no live credentials needed.

---

## Project structure

```
Prism/
├── app/
│   ├── main.py           # FastAPI app + lifespan
│   ├── feed_service.py   # Candidate pool building + freshness logic
│   ├── ranker.py         # Probabilistic engagement scorer
│   ├── bluesky_client.py # AT Protocol HTTP client with caching
│   ├── embedding.py      # Ollama embedder + cosine similarity
│   ├── preferences.py    # Topic prefs + shown-post tracking + liked-author cache
│   ├── models.py         # CandidatePost dataclass
│   ├── config.py         # Settings from env
│   └── cache.py          # Async TTL cache
├── templates/
│   └── index.html        # Feed UI with debug panels
├── tests/
│   ├── test_feed_service.py
│   ├── test_ranker.py
│   ├── test_cache.py
│   └── test_main_template.py
├── data/
│   └── preferences.json  # Local user prefs + shown tracking (gitignored)
├── pyproject.toml
├── run_local.sh
└── README.md
```

---

## License

Same as the upstream x-algorithm repository this is derived from.

---

## Acknowledgments

- Ranking framework and engagement-weighting approach adapted from [xai-org/x-algorithm](https://github.com/xai-org/x-algorithm)
- Bluesky AT Protocol reference: [atproto.com](https://atproto.com)
