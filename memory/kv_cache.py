"""
KV-Cache — Python-level prompt prefix cache for LLM calls.

Caches system prompt + shared context prefixes keyed by SHA-256 hash.
On cache hit, the cached message list is reused instead of re-building,
and token savings are estimated (approx chars / 4).

Thread-safe via a simple lock.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


class KVCache:
    """
    Prompt-prefix KV cache.

    Stores message lists keyed by the SHA-256 of their serialized form.
    Tracks hit/miss/token-savings statistics.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

        # statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._total_requests = 0

    # ── Key helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_key(prefix: str) -> str:
        """SHA-256 hash of a prefix string."""
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: character count / 4."""
        return max(1, len(text) // 4)

    # ── Public API ───────────────────────────────────────────────────────

    def get(self, prefix: str) -> list[dict[str, str]] | None:
        """
        Look up cached messages for the given prefix.

        Returns the cached message list on hit, or None on miss.
        """
        key = self._make_key(prefix)

        with self._lock:
            self._total_requests += 1

            if key in self._cache:
                self._hits += 1
                entry = self._cache[key]
                entry["last_access"] = time.time()
                entry["access_count"] += 1
                self._tokens_saved += self._estimate_tokens(prefix)
                return entry["messages"]

            self._misses += 1
            return None

    def set(self, prefix: str, messages: list[dict[str, str]]) -> None:
        """
        Store a message list under the given prefix.

        Evicts the least-recently-accessed entry when the cache is full.
        """
        key = self._make_key(prefix)

        with self._lock:
            # Evict LRU if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                lru_key = min(
                    self._cache,
                    key=lambda k: self._cache[k]["last_access"],
                )
                del self._cache[lru_key]

            self._cache[key] = {
                "messages": messages,
                "prefix": prefix[:200],  # store truncated for debugging
                "created": time.time(),
                "last_access": time.time(),
                "access_count": 1,
            }

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "total_requests": self._total_requests,
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "estimated_tokens_saved": self._tokens_saved,
                "estimated_cost_saved_usd": 0.0,  # all free-tier
            }

    def clear(self) -> None:
        """Flush the entire cache and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._tokens_saved = 0
            self._total_requests = 0


# ── Module-level singleton ───────────────────────────────────────────────────
_global_cache = KVCache()


def get_global_cache() -> KVCache:
    """Return the module-level singleton KVCache."""
    return _global_cache


def cached_llm_invoke(
    llm: Any,
    system_prompt: str,
    user_message: str,
    cache: KVCache | None = None,
) -> Any:
    """
    Invoke an LLM with KV-Cache prefix reuse.

    If the system_prompt has been seen before, we reuse the cached message
    structure instead of rebuilding. Either way the LLM is called (we can't
    skip that), but the cached prefix tracks savings across calls that share
    the same system prompt across agents/queries.

    Args:
        llm: A LangChain chat model (or mock).
        system_prompt: The system message text (cacheable prefix).
        user_message: The human message text (variable suffix).
        cache: KVCache instance (defaults to the global singleton).

    Returns:
        The AIMessage response from the LLM.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if cache is None:
        cache = _global_cache

    # Look up cached system prefix
    cached = cache.get(system_prompt)
    if cached is not None:
        # Cache hit — reuse the pre-built system message list
        messages = [SystemMessage(content=m["content"]) for m in cached]
        messages.append(HumanMessage(content=user_message))
    else:
        # Cache miss — build and store
        sys_msg = SystemMessage(content=system_prompt)
        messages = [sys_msg, HumanMessage(content=user_message)]
        cache.set(
            system_prompt,
            [{"role": "system", "content": system_prompt}],
        )

    return llm.invoke(messages)
