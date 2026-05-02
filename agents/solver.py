"""
Medical Advisor (Solver) Agent — produces the final clinical answer.

This is the ONLY agent that produces verbose output (LatentMAS pattern).
Always includes medical disclaimer and evidence-based recommendations.
"""

from __future__ import annotations
import json, re
from typing import Any
from memory.kv_cache import cached_llm_invoke, get_global_cache

SOLVER_SYSTEM_PROMPT = """You are a Medical Advisor — the final agent in a healthcare multi-agent system.

You receive compressed clinical context from previous agents.
Produce the definitive, evidence-based medical answer.

IMPORTANT: Respond with valid JSON only.

Response format:
{
  "answer": "Clear, evidence-based medical answer with safety disclaimer.",
  "reasoning_trace": "Key clinical reasoning steps."
}

Rules:
- Provide evidence-based medical information.
- Include relevant dosages, guidelines, or clinical criteria when applicable.
- ALWAYS end with: "⚕️ Disclaimer: This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice."
- Be thorough but do not repeat the question."""


class SolverAgent:
    """Produces the final medical answer with disclaimer."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cache = get_global_cache()

    def run(self, query: str, plan: dict[str, Any], research: dict[str, Any], draft: str) -> dict[str, Any]:
        from memory.context_compressor import build_solver_context
        compressed = build_solver_context(query, plan, research, draft)
        user_message = f"{compressed}\n\nProduce the definitive clinical answer as JSON. Include medical disclaimer."
        response = cached_llm_invoke(self.llm, SOLVER_SYSTEM_PROMPT, user_message, cache=self.cache)
        return self._parse_response(response.content, query)

    def _parse_response(self, content: str, query: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "answer" in data:
                answer = data["answer"]
                if "disclaimer" not in answer.lower() and "consult" not in answer.lower():
                    answer += "\n\n⚕️ Disclaimer: This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice."
                return {"answer": answer, "reasoning_trace": data.get("reasoning_trace", "")}
        except (json.JSONDecodeError, KeyError):
            pass
        disclaimer = "\n\n⚕️ Disclaimer: This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice."
        return {"answer": content.strip() + disclaimer, "reasoning_trace": f"Clinical response for: {query}"}
