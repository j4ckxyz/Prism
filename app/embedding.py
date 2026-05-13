from __future__ import annotations

from typing import Dict, List
import math
import httpx


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.AsyncClient(timeout=60.0)
        self._cache: Dict[str, List[float]] = {}

    async def close(self) -> None:
        await self._http.aclose()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.strip()[:1200] for t in texts]
        result: List[List[float]] = []
        missing_idx = []
        missing_texts = []
        for i, t in enumerate(cleaned):
            if t in self._cache:
                result.append(self._cache[t])
            else:
                result.append([])
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            vectors = await self._embed_fallback(missing_texts)
            for idx, text, vec in zip(missing_idx, missing_texts, vectors):
                self._cache[text] = vec
                result[idx] = vec

        return result

    async def _embed_fallback(self, texts: List[str]) -> List[List[float]]:
        try:
            r = await self._http.post(
                f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}
            )
            if r.status_code < 300:
                data = r.json()
                embs = data.get("embeddings", [])
                if embs:
                    return embs
        except Exception:
            pass

        # legacy fallback endpoint
        out: List[List[float]] = []
        for t in texts:
            r = await self._http.post(
                f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": t}
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out



def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
