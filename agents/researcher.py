"""
Medical Researcher Agent — retrieves evidence from the healthcare FAISS
knowledge base (100+ medical docs) and synthesises clinical findings.

RAG-only, no web search. Uses NVIDIA LLaMA for synthesis.
"""

from __future__ import annotations
import json, re
from typing import Any
from memory.kv_cache import cached_llm_invoke, get_global_cache
from memory.vector_store import VectorStoreManager

RESEARCHER_SYSTEM_PROMPT = """You are a Medical Researcher Agent in a healthcare multi-agent system.

You receive a clinical plan and retrieved medical knowledge base documents.
Synthesise the evidence into a concise clinical research summary.

IMPORTANT: Respond with valid JSON only. Be evidence-based and concise.

Response format:
{
  "retrieved_docs": ["Key clinical finding 1", "Key finding 2"],
  "summary": "Evidence-based synthesis relevant to the clinical query."
}

Rules:
- Focus on evidence-based medical information.
- Cite relevant clinical guidelines when available.
- Note any contraindications or drug interactions.
- Keep summary under 200 words."""


class ResearcherAgent:
    """Retrieves and synthesises medical information from RAG."""

    def __init__(self, llm: Any, vector_store: VectorStoreManager | None = None) -> None:
        self.llm = llm
        self.cache = get_global_cache()
        self.vector_store = vector_store or VectorStoreManager()

    def _rag_search(self, query: str, k: int = 5) -> list[str]:
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as exc:
            return [f"RAG retrieval error: {exc}"]

    def run(self, plan: dict[str, Any], query: str) -> dict[str, Any]:
        from memory.context_compressor import compress_plan
        rag_docs = self._rag_search(query, k=5)
        plan_text = compress_plan(plan)
        user_message = (
            f"Medical Query: {query}\n\nClinical Plan:\n{plan_text}\n\n"
            f"Medical Knowledge Base:\n" + "\n---\n".join(rag_docs[:5])
            + "\n\nSynthesise into a concise clinical JSON report."
        )
        response = cached_llm_invoke(self.llm, RESEARCHER_SYSTEM_PROMPT, user_message, cache=self.cache)
        return self._parse_response(response.content, rag_docs)

    def _parse_response(self, content: str, rag_docs: list[str]) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "summary" in data:
                return {"retrieved_docs": data.get("retrieved_docs", rag_docs[:5]), "summary": data["summary"]}
        except (json.JSONDecodeError, KeyError):
            pass
        return {"retrieved_docs": rag_docs[:5], "summary": content[:500]}
