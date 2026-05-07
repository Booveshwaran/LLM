"""
FastAPI REST interface for the Healthcare Multi-Agent LLM System.

Endpoints:
  POST /query        — Run a clinical query (synchronous)
  GET  /query/stream — Run with SSE streaming (real-time agent progress)
  GET  /cache-stats  — KV-Cache statistics
  GET  /cag-stats    — Cache-Augmented Generation statistics
  GET  /health       — Health check

Healthcare domain with 100+ medical RAG docs and CAG.
"""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from graph.workflow import run_workflow, run_workflow_streaming
from memory.kv_cache import get_global_cache
from memory.vector_store import preload_global_store
from memory.cag import get_global_cag
from router.llm_router import LLM_CONFIG

# ── App initialisation ──────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent LLM Collaboration System",
    description=(
        "A LatentMAS-inspired multi-agent reasoning framework with compressed "
        "inter-agent communication, RAG retrieval, and SSE streaming."
    ),
    version="2.0.0",
)

# ── CORS middleware ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Preload FAISS on startup ────────────────────────────────────────────────

@app.on_event("startup")
async def startup_preload():
    """Eagerly load embedding model + FAISS index at server startup."""
    import threading
    threading.Thread(target=preload_global_store, daemon=True).start()


# ── Request / Response schemas ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or task to process.")
    mock: bool = Field(
        default=False,
        description="If true, uses mock LLMs (no API keys required).",
    )


class QueryResponse(BaseModel):
    final_answer: str
    reasoning_trace: list[str]
    token_stats: dict[str, Any]
    latency_seconds: float
    retry_count: int


class CacheStatsResponse(BaseModel):
    cache_size: int
    max_size: int
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    estimated_tokens_saved: int
    estimated_cost_saved_usd: float


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]
    api_keys_configured: dict[str, bool]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Run a query through the full multi-agent workflow (synchronous)."""
    try:
        t0 = time.time()
        result = run_workflow(query=request.query, mock=request.mock)
        elapsed = round(time.time() - t0, 2)

        return QueryResponse(
            final_answer=result.get("final_answer", "No answer generated."),
            reasoning_trace=result.get("reasoning_trace", []),
            token_stats=result.get("token_stats", {}),
            latency_seconds=elapsed,
            retry_count=result.get("retry_count", 0),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(exc)}",
        )


@app.get("/query/stream")
async def stream_query(
    q: str = Query(..., min_length=1, description="The query to process."),
    mock: bool = Query(default=False, description="Use mock mode."),
):
    """
    Stream agent progress via Server-Sent Events (SSE).

    Events:
      - type: agent_update — {agent, status, detail}
      - type: result — {final_answer, reasoning_trace, token_stats}
      - type: error — {message}
    """
    def event_generator():
        yield from run_workflow_streaming(query=q, mock=mock)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/cache-stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    """Return KV-Cache hit/miss/tokens-saved statistics."""
    stats = get_global_cache().stats()
    return CacheStatsResponse(**stats)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check — reports model routing and API key status."""
    models = {
        agent: cfg["model"]
        for agent, cfg in LLM_CONFIG.items()
    }
    api_keys = {
        agent: bool(os.getenv(cfg["env_key"]))
        for agent, cfg in LLM_CONFIG.items()
    }
    return HealthResponse(
        status="ok",
        models=models,
        api_keys_configured=api_keys,
    )


@app.get("/cag-stats")
async def cag_stats():
    """Return Cache-Augmented Generation statistics."""
    return get_global_cag().stats()


@app.post("/evaluate")
async def evaluate_query(request: QueryRequest):
    """
    Run a query through the pipeline AND return evaluation metrics.

    Returns: answer + answer_eval (correctness) + rag_eval (retrieval quality).
    Requires 'expected_keywords' in the request body for meaningful scoring.
    """
    from eval.metrics import evaluate_answer, evaluate_retrieval

    t0 = time.time()
    result = run_workflow(query=request.query, mock=request.mock)
    elapsed = round(time.time() - t0, 2)

    answer = result.get("final_answer", "")
    research = result.get("research", {})
    retrieved_docs = research.get("retrieved_docs", [])

    # Extract keywords from the query for auto-evaluation
    import re
    query_words = re.findall(r"\b[a-z]{4,}\b", request.query.lower())
    stopwords = {"what", "which", "when", "where", "that", "this", "with", "from",
                 "have", "does", "most", "been", "first", "line", "treatment"}
    auto_keywords = [w for w in query_words if w not in stopwords][:5]

    answer_eval = evaluate_answer(answer, request.query, auto_keywords)
    rag_eval = evaluate_retrieval(retrieved_docs, auto_keywords, k=5)

    return {
        "final_answer": answer,
        "latency_seconds": elapsed,
        "answer_eval": answer_eval,
        "rag_eval": rag_eval,
        "retrieved_docs_count": len(retrieved_docs),
    }


@app.get("/evaluate/rag")
async def evaluate_rag_retrieval(
    q: str = Query(..., min_length=1, description="Query to evaluate RAG retrieval for."),
    k: int = Query(default=5, description="Number of documents to retrieve."),
):
    """
    Standalone RAG retrieval evaluation — no LLM, just FAISS search + metrics.

    Returns: retrieved chunks with per-chunk relevance scores.
    """
    from eval.metrics import evaluate_retrieval
    from memory.vector_store import get_global_store

    vs = get_global_store()
    docs = vs.similarity_search(q, k=k)

    import re
    query_words = re.findall(r"\b[a-z]{4,}\b", q.lower())
    stopwords = {"what", "which", "when", "where", "that", "this", "with", "from",
                 "have", "does", "most", "been", "first", "line", "treatment"}
    keywords = [w for w in query_words if w not in stopwords][:8]

    rag_eval = evaluate_retrieval(docs, keywords, k=k)

    return {
        "query": q,
        "k": k,
        "keywords_used": keywords,
        **rag_eval,
    }


# ── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
