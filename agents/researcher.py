"""
Researcher Agent — retrieves relevant information from the local FAISS
vector store (RAG) and synthesises a concise summary.

LatentMAS-inspired: produces compressed output that minimizes token overhead
for downstream agents. No web search — RAG-only for speed.

Uses NVIDIA llama-3.3-70b-instruct (free tier at build.nvidia.com).
Returns: {"retrieved_docs": [...], "summary": "..."}
"""

from __future__ import annotations

import json
import re
from typing import Any

from memory.kv_cache import cached_llm_invoke, get_global_cache
from memory.vector_store import VectorStoreManager

RESEARCHER_SYSTEM_PROMPT = """You are a Researcher Agent in a latent multi-agent system.

You receive a plan and retrieved knowledge-base documents.
Synthesise the information into a concise research summary.

IMPORTANT: Respond with valid JSON only. Be concise — max 3-4 sentences.

Response format (strict JSON):
{
  "retrieved_docs": ["Key finding 1", "Key finding 2", "Key finding 3"],
  "summary": "Concise synthesis of findings relevant to solving the query."
}

Rules:
- Extract only the most relevant facts.
- Keep the summary under 200 words.
- Do not add information not present in the documents."""


class ResearcherAgent:
    """Retrieves and synthesises information from local RAG (no web search)."""

    def __init__(self, llm: Any, vector_store: VectorStoreManager | None = None) -> None:
        self.llm = llm
        self.cache = get_global_cache()
        self.vector_store = vector_store or VectorStoreManager()

    def _rag_search(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant documents from the FAISS vector store."""
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as exc:
            return [f"RAG retrieval error: {exc}"]

    def run(self, plan: dict[str, Any], query: str) -> dict[str, Any]:
        """
        Research the query using RAG retrieval only.

        Args:
            plan: The plan dict from the Planner (has "steps" and "context").
            query: The original user query.

        Returns:
            Dict with keys "retrieved_docs" (list[str]) and "summary" (str).
        """
        from memory.context_compressor import compress_plan

        # RAG retrieval (top-5 for better coverage)
        rag_docs = self._rag_search(query, k=5)

        # Build compressed prompt (LatentMAS-style minimal context)
        plan_text = compress_plan(plan)

        user_message = (
            f"Query: {query}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Knowledge Base Documents:\n"
            + "\n---\n".join(rag_docs[:5])
            + "\n\nSynthesise into a concise JSON research report."
        )

        response = cached_llm_invoke(
            self.llm,
            RESEARCHER_SYSTEM_PROMPT,
            user_message,
            cache=self.cache,
        )

        return self._parse_response(response.content, rag_docs)

    def _parse_response(
        self, content: str, rag_docs: list[str]
    ) -> dict[str, Any]:
        """Parse the LLM response with fallback."""
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "summary" in data:
                return {
                    "retrieved_docs": data.get("retrieved_docs", rag_docs[:5]),
                    "summary": data["summary"],
                }
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback
        return {
            "retrieved_docs": rag_docs[:5],
            "summary": content[:500],
        }
