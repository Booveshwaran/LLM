"""
Solver Agent — the final Judger that produces the definitive answer.

LatentMAS-inspired: This is the ONLY agent that produces verbose text output.
All preceding agents communicate via compressed representations, and the Solver
receives accumulated compressed context to generate the final answer.

Uses NVIDIA llama-3.3-70b-instruct (free tier, max_tokens=800).
Returns: {"answer": "...", "reasoning_trace": "..."}
"""

from __future__ import annotations

import json
import re
from typing import Any

from memory.kv_cache import cached_llm_invoke, get_global_cache

SOLVER_SYSTEM_PROMPT = """You are the Solver — the final agent in a multi-agent reasoning system.

You receive compressed context from previous agents (plan, research, refined draft).
Your job is to produce the definitive final answer.

IMPORTANT: Respond with valid JSON only.

Response format:
{
  "answer": "Clear, complete final answer to the query.",
  "reasoning_trace": "Key reasoning steps that led to this answer."
}

Rules:
- Directly address the user's question.
- For math: show the final calculation clearly.
- For coding: include working code.
- Be thorough but do not repeat the question."""


class SolverAgent:
    """Produces the final answer from compressed agent context."""

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
        """Generate the final answer using compressed context."""
        from memory.context_compressor import build_solver_context

        compressed_context = build_solver_context(query, plan, research, draft)

        user_message = (
            f"{compressed_context}\n\n"
            "Using the above context, produce the definitive final answer as JSON."
        )

        response = cached_llm_invoke(
            self.llm,
            SOLVER_SYSTEM_PROMPT,
            user_message,
            cache=self.cache,
        )

        return self._parse_response(response.content, query)

    def _parse_response(self, content: str, query: str) -> dict[str, Any]:
        """Parse the LLM response with fallback."""
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "answer" in data:
                return {
                    "answer": data["answer"],
                    "reasoning_trace": data.get("reasoning_trace", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass

        return {
            "answer": content.strip(),
            "reasoning_trace": f"Direct response generated for: {query}",
        }
