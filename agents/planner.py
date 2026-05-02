"""
Clinical Planner Agent — decomposes a medical query into a structured
differential diagnosis and clinical reasoning plan.

Uses Mistral large (temp=0, max_tokens=300) for fast clinical planning.
"""

from __future__ import annotations
import json, re
from typing import Any
from memory.kv_cache import cached_llm_invoke, get_global_cache

PLANNER_SYSTEM_PROMPT = """You are a Clinical Planner Agent in a healthcare multi-agent system.

Given a medical query, create a concise clinical reasoning plan.

IMPORTANT: Respond with valid JSON only. Max 5 steps.

Response format:
{
  "steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "context": "Brief clinical context."
}

Rules:
- Consider differential diagnoses.
- Include relevant history/exam/lab steps.
- Include a verification step.
- Keep each step under 20 words.
- Add medical disclaimer awareness."""


class PlannerAgent:
    """Decomposes a medical query into a clinical reasoning plan."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cache = get_global_cache()

    def run(self, query: str) -> dict[str, Any]:
        response = cached_llm_invoke(
            self.llm, PLANNER_SYSTEM_PROMPT,
            f"Create a clinical plan for: {query}",
            cache=self.cache,
        )
        return self._parse_response(response.content, query)

    def _parse_response(self, content: str, query: str) -> dict[str, Any]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned.strip())
            if isinstance(data, dict) and "steps" in data:
                return {"steps": data["steps"][:5], "context": data.get("context", f"Clinical plan for: {query}")}
        except (json.JSONDecodeError, KeyError):
            pass
        lines = content.strip().split("\n")
        steps = [l.strip() for l in lines if re.match(r"^\d+[\.)\s]", l.strip()) or l.strip().startswith("Step")]
        if not steps:
            steps = ["Step 1: Identify symptoms and history", "Step 2: Consider differential diagnoses",
                     "Step 3: Determine relevant investigations", "Step 4: Formulate clinical assessment"]
        return {"steps": steps[:5], "context": f"Clinical plan for: {query}"}
