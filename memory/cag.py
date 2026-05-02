"""
Cache-Augmented Generation (CAG) — semantic response caching.

Goes beyond simple KV-cache prefix reuse: CAG caches full LLM responses
keyed by semantic similarity of the query. If a new query is sufficiently
similar (>= threshold) to a previously answered query, the cached response
is returned instantly — zero LLM calls.

Uses the same FAISS embedding model as the RAG vector store for consistency.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


class CacheAugmentedGeneration:
    """
    Semantic response cache for healthcare queries.

    Stores (query → full_response) pairs and uses cosine similarity
    to match new queries against cached ones. If similarity >= threshold,
    returns the cached response without calling the LLM.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        max_entries: int = 500,
    ) -> None:
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self._cache: dict[str, dict[str, Any]] = {}
        self._embeddings = None
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._total_requests = 0

    def _get_embeddings(self):
        """Lazily load the embedding model."""
        if self._embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception:
                return None
        return self._embeddings

    def _compute_similarity(self, query1: str, query2: str) -> float:
        """Compute cosine similarity between two queries."""
        emb = self._get_embeddings()
        if emb is None:
            return 0.0
        try:
            vecs = emb.embed_documents([query1, query2])
            # Cosine similarity (embeddings are already normalized)
            dot = sum(a * b for a, b in zip(vecs[0], vecs[1]))
            return dot
        except Exception:
            return 0.0

    def lookup(self, query: str) -> dict[str, Any] | None:
        """
        Check if a semantically similar query has been cached.

        Returns the cached response dict if found, else None.
        """
        with self._lock:
            self._total_requests += 1

        if not self._cache:
            with self._lock:
                self._misses += 1
            return None

        # Check exact hash match first (fastest path)
        key = hashlib.sha256(query.lower().strip().encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._hits += 1
                entry = self._cache[key]
                entry["access_count"] += 1
                entry["last_access"] = time.time()
                return entry["response"]

        # Semantic similarity check against cached queries
        best_sim = 0.0
        best_key = None

        with self._lock:
            cached_items = list(self._cache.items())

        for cached_key, entry in cached_items:
            sim = self._compute_similarity(query, entry["query"])
            if sim > best_sim:
                best_sim = sim
                best_key = cached_key

        if best_sim >= self.threshold and best_key:
            with self._lock:
                self._hits += 1
                entry = self._cache[best_key]
                entry["access_count"] += 1
                entry["last_access"] = time.time()
                return entry["response"]

        with self._lock:
            self._misses += 1
        return None

    def store(self, query: str, response: dict[str, Any]) -> None:
        """Cache a query-response pair."""
        key = hashlib.sha256(query.lower().strip().encode()).hexdigest()

        with self._lock:
            # Evict LRU if at capacity
            if len(self._cache) >= self.max_entries and key not in self._cache:
                lru_key = min(
                    self._cache,
                    key=lambda k: self._cache[k]["last_access"],
                )
                del self._cache[lru_key]

            self._cache[key] = {
                "query": query,
                "response": response,
                "created": time.time(),
                "last_access": time.time(),
                "access_count": 1,
            }

    def stats(self) -> dict[str, Any]:
        """Return CAG performance statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "cag_cache_size": len(self._cache),
                "cag_max_entries": self.max_entries,
                "cag_total_requests": self._total_requests,
                "cag_hits": self._hits,
                "cag_misses": self._misses,
                "cag_hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "similarity_threshold": self.threshold,
            }

    def clear(self) -> None:
        """Flush the CAG cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._total_requests = 0


# ── Module-level singleton ───────────────────────────────────────────────────
_global_cag = CacheAugmentedGeneration()


def get_global_cag() -> CacheAugmentedGeneration:
    """Return the global CAG singleton."""
    return _global_cag
