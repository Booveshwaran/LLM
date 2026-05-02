"""
Clinical Refiner Agent — corrects medical inaccuracies based on reviewer feedback.

Ensures clinical accuracy, adds missing safety information, and
corrects any potentially harmful medical advice.
"""

from __future__ import annotations
from typing import Any
from memory.kv_cache import cached_llm_invoke, get_global_cache

REFINER_SYSTEM_PROMPT = """You are a Clinical Refiner Agent. Improve the medical draft based on reviewer feedback.

Rules:
- Fix all identified clinical inaccuracies.
- Add missing safety warnings or contraindications.
- Ensure evidence-based recommendations.
- Include "Consult a healthcare professional" disclaimer.
- Return ONLY the improved text — no JSON, no commentary.
- Maximum 200 words."""


class RefinerAgent:
    """Corrects medical content based on clinical reviewer feedback."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cache = get_global_cache()

    def run(self, query: str, draft: str, critique: dict[str, Any], research: dict[str, Any]) -> str:
        from memory.context_compressor import compress_critique, compress_draft
        critique_text = compress_critique(critique)
        draft_text = compress_draft(draft, max_len=500)
        user_message = (
            f"Medical Query: {query}\n\nDraft:\n{draft_text}\n\n"
            f"Clinical Review:\n{critique_text}\n\n"
            "Rewrite fixing all clinical issues. Include safety disclaimer."
        )
        response = cached_llm_invoke(self.llm, REFINER_SYSTEM_PROMPT, user_message, cache=self.cache)
        return response.content.strip()
