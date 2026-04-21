"""
Critic Agent — evaluates intermediate reasoning for correctness.

LatentMAS-inspired: produces minimal structured output (score + issues only)
to reduce inter-agent token overhead. Approval threshold at 7/10.

Uses Mistral large (free tier, temp=0, max_tokens=200) for fast evaluation.
Returns: {"issues": [...], "score": 0-10, "approved": bool}
"""

from __future__ import annotations

import json
import re
from typing import Any

from memory.kv_cache import cached_llm_invoke, get_global_cache

CRITIC_SYSTEM_PROMPT = """You are a Critic Agent. Evaluate the draft for correctness.

IMPORTANT: Respond with valid JSON only. Be brief.

Response format:
{
  "issues": ["issue 1", "issue 2"],
  "score": 8,
  "approved": true
}

Rules:
- Score 0-10. approved = true if score >= 7.
- List max 3 most critical issues only.
- Keep each issue under 20 words.
- If no issues, set issues to []."""


class CriticAgent:
    """Evaluates the quality of intermediate reasoning outputs."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cache = get_global_cache()

    def run(
        self,
        query: str,
        plan: dict[str, Any],
        research: dict[str, Any],
        draft: str,
    ) -> dict[str, Any]:
        """Critique the current draft."""
        from memory.context_compressor import compress_plan, compress_draft

        plan_text = compress_plan(plan)
        research_summary = research.get("summary", "")[:300]
        draft_text = compress_draft(draft, max_len=500)

        user_message = (
            f"Query: {query}\n"
            f"Plan:\n{plan_text}\n"
            f"Research: {research_summary}\n"
            f"Draft:\n{draft_text}\n\n"
            "Evaluate and return JSON critique."
        )

        response = cached_llm_invoke(
            self.llm,
            CRITIC_SYSTEM_PROMPT,
            user_message,
            cache=self.cache,
        )

        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse the LLM response with fallback."""
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                score = int(data.get("score", 5))
                score = max(0, min(10, score))
                return {
                    "issues": data.get("issues", [])[:3],
                    "score": score,
                    "approved": score >= 7,
                }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return {
            "issues": ["Could not parse critic response."],
            "score": 5,
            "approved": False,
        }
