"""
Refiner Agent — takes a draft + critic feedback and produces an improved version.

LatentMAS-inspired: receives compressed context (plan + feedback) and outputs
a refined plan. Mirrors the paper's Refiner pattern.

Uses NVIDIA llama-3.3-70b-instruct (free tier, max_tokens=500).
Returns: improved reasoning string.
"""

from __future__ import annotations

from typing import Any

from memory.kv_cache import cached_llm_invoke, get_global_cache

REFINER_SYSTEM_PROMPT = """You are a Refiner Agent. Improve the draft based on critique feedback.

Rules:
- Fix all identified issues.
- Keep the improved draft concise and focused.
- Preserve correct reasoning from the original.
- Return ONLY the improved draft text — no JSON, no commentary.
- Maximum 200 words."""


class RefinerAgent:
    """Rewrites drafts based on critic feedback."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cache = get_global_cache()

    def run(
        self,
        query: str,
        draft: str,
        critique: dict[str, Any],
        research: dict[str, Any],
    ) -> str:
        """Refine the draft based on critic feedback."""
        from memory.context_compressor import compress_critique, compress_draft

        critique_text = compress_critique(critique)
        draft_text = compress_draft(draft, max_len=500)

        user_message = (
            f"Query: {query}\n\n"
            f"Draft:\n{draft_text}\n\n"
            f"Critique:\n{critique_text}\n\n"
            "Rewrite the draft, fixing all issues. Return only the improved text."
        )

        response = cached_llm_invoke(
            self.llm,
            REFINER_SYSTEM_PROMPT,
            user_message,
            cache=self.cache,
        )

        return response.content.strip()
