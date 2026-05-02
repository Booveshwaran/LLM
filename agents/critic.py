"""
Medical Reviewer (Critic) Agent — evaluates clinical accuracy of responses.

Checks for: medical accuracy, evidence basis, dangerous advice,
drug interaction awareness, and appropriate disclaimers.
"""

from __future__ import annotations
import json, re
from typing import Any
from memory.kv_cache import cached_llm_invoke, get_global_cache

CRITIC_SYSTEM_PROMPT = """You are a Medical Reviewer Agent. Evaluate the clinical draft for accuracy and safety.

IMPORTANT: Respond with valid JSON only.

Response format:
{"issues": ["issue 1"], "score": 8, "approved": true}

Check for:
- Medical accuracy and evidence basis
- Dangerous or harmful advice
- Missing contraindications or drug interactions
- Appropriate disclaimers about consulting a doctor
- Score 0-10. approved = true if score >= 7.
- Max 3 critical issues. Keep each under 20 words."""


class CriticAgent:
    """Evaluates clinical accuracy of intermediate outputs."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cache = get_global_cache()

    def run(self, query: str, plan: dict[str, Any], research: dict[str, Any], draft: str) -> dict[str, Any]:
        from memory.context_compressor import compress_plan, compress_draft
        plan_text = compress_plan(plan)
        draft_text = compress_draft(draft, max_len=500)
        research_summary = research.get("summary", "")[:300]
        user_message = (
            f"Medical Query: {query}\nPlan:\n{plan_text}\n"
            f"Research: {research_summary}\nDraft:\n{draft_text}\n\n"
            "Evaluate clinical accuracy and safety. Return JSON."
        )
        response = cached_llm_invoke(self.llm, CRITIC_SYSTEM_PROMPT, user_message, cache=self.cache)
        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                score = max(0, min(10, int(data.get("score", 5))))
                return {"issues": data.get("issues", [])[:3], "score": score, "approved": score >= 7}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return {"issues": ["Could not parse critic response."], "score": 5, "approved": False}
