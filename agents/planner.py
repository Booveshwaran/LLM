"""
Planner Agent — decomposes a raw user query into a concise step-by-step plan.

LatentMAS-inspired: produces minimal, structured output to reduce token
overhead for downstream agents in the pipeline.

Uses Mistral large (free tier, temp=0, max_tokens=300) for fast planning.
Returns: {"steps": [...], "context": "..."}
"""

from __future__ import annotations

import json
import re
from typing import Any

from memory.kv_cache import cached_llm_invoke, get_global_cache

PLANNER_SYSTEM_PROMPT = """You are a Planner Agent. Given a question, design a concise step-by-step plan.

IMPORTANT: Respond with valid JSON only. Keep it short — 3 to 5 steps max.

Response format:
{
  "steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "context": "Brief goal description."
}

Rules:
- Each step must be concrete and actionable.
- Maximum 5 steps.
- Keep each step under 15 words.
- Include a verification step at the end."""


class PlannerAgent:
    """Decomposes a query into a structured plan using Mistral large."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cache = get_global_cache()

    def run(self, query: str) -> dict[str, Any]:
        """Generate a step-by-step plan for the given query."""
        response = cached_llm_invoke(
            self.llm,
            PLANNER_SYSTEM_PROMPT,
            f"Create a plan for: {query}",
            cache=self.cache,
        )
        return self._parse_response(response.content, query)

    def _parse_response(self, content: str, query: str) -> dict[str, Any]:
        """Parse the LLM response, with robust fallback."""
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "steps" in data:
                return {
                    "steps": data["steps"][:5],
                    "context": data.get("context", f"Plan for: {query}"),
                }
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: extract numbered lines as steps
        lines = content.strip().split("\n")
        steps = [
            line.strip()
            for line in lines
            if re.match(r"^\d+[\.)\s]", line.strip()) or line.strip().startswith("Step")
        ]
        if not steps:
            steps = [
                "Step 1: Understand the problem",
                "Step 2: Identify key information",
                "Step 3: Apply reasoning and compute",
                "Step 4: Verify the result",
            ]

        return {
            "steps": steps[:5],
            "context": f"Plan for: {query}",
        }
