"""
Healthcare Benchmark — evaluate multi-agent clinical system vs single-LLM.

Tests on 8 clinical scenarios with comprehensive evaluation metrics:
  1. Answer Correctness — keyword F1, medical safety, composite score
  2. RAG Retrieval Quality — precision@k, recall@k, MRR, NDCG@5

Usage:
    python eval/benchmark.py --mock          # no API keys needed
    python eval/benchmark.py                 # live mode with API keys
    python eval/benchmark.py --mock --verbose # show per-chunk RAG analysis
"""

from __future__ import annotations
import io, json, os, sys, time
from typing import Any

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from tabulate import tabulate
from graph.workflow import run_workflow
from memory.kv_cache import cached_llm_invoke, get_global_cache, KVCache
from memory.vector_store import get_global_store
from eval.metrics import (
    evaluate_answer,
    evaluate_retrieval,
    print_evaluation_report,
    medical_safety_score,
)

# ── Benchmark problems with expected keywords and RAG relevance keywords ─────

BENCHMARK_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "dx_1", "type": "diagnosis",
        "query": "A 55-year-old male presents with crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath. What is the most likely diagnosis and immediate management?",
        "expected": "myocardial infarction",
        "keywords": ["myocardial infarction", "aspirin"],
        "rag_keywords": ["myocardial infarction", "chest pain", "troponin", "aspirin", "ECG", "STEMI"],
    },
    {
        "id": "dx_2", "type": "diagnosis",
        "query": "A patient with diabetes has blood glucose of 450 mg/dL, pH 7.2, and positive ketones. What is the diagnosis and treatment?",
        "expected": "DKA",
        "keywords": ["ketoacidosis", "insulin"],
        "rag_keywords": ["ketoacidosis", "DKA", "insulin", "glucose", "pH", "potassium", "IV fluids"],
    },
    {
        "id": "tx_1", "type": "treatment",
        "query": "What is the first-line treatment for newly diagnosed Type 2 diabetes?",
        "expected": "metformin",
        "keywords": ["metformin"],
        "rag_keywords": ["metformin", "Type 2 diabetes", "HbA1c", "first-line", "SGLT2"],
    },
    {
        "id": "tx_2", "type": "treatment",
        "query": "What is the recommended treatment for uncomplicated urinary tract infection in women?",
        "expected": "nitrofurantoin",
        "keywords": ["nitrofurantoin"],
        "rag_keywords": ["nitrofurantoin", "urinary tract infection", "cystitis", "TMP-SMX", "antibiotic"],
    },
    {
        "id": "drug_1", "type": "drug_interaction",
        "query": "A patient on warfarin is prescribed ibuprofen. What is the major concern?",
        "expected": "bleeding",
        "keywords": ["bleeding"],
        "rag_keywords": ["warfarin", "NSAID", "bleeding", "drug interaction", "anticoagul"],
    },
    {
        "id": "lab_1", "type": "lab_interpretation",
        "query": "A patient has TSH of 12 mIU/L and free T4 of 0.4 ng/dL. What is the diagnosis?",
        "expected": "hypothyroidism",
        "keywords": ["hypothyroidism"],
        "rag_keywords": ["TSH", "T4", "hypothyroidism", "thyroid", "levothyroxine"],
    },
    {
        "id": "emerg_1", "type": "emergency",
        "query": "A patient is experiencing anaphylaxis. What is the first-line treatment and dose?",
        "expected": "epinephrine",
        "keywords": ["epinephrine"],
        "rag_keywords": ["anaphylaxis", "epinephrine", "EpiPen", "IM", "0.3"],
    },
    {
        "id": "prev_1", "type": "prevention",
        "query": "What cancer screening is recommended starting at age 45?",
        "expected": "colonoscopy",
        "keywords": ["colon"],
        "rag_keywords": ["colonoscopy", "colorectal", "screening", "age 45", "FIT"],
    },
]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── RAG-only retrieval (no LLM) ─────────────────────────────────────────────

def _run_rag_retrieval(query: str, k: int = 5) -> list[str]:
    """Retrieve top-k docs from FAISS without running any LLM."""
    try:
        vs = get_global_store()
        return vs.similarity_search(query, k=k)
    except Exception as exc:
        print(f"  [WARN] RAG retrieval failed: {exc}")
        return []


# ── Single LLM baseline ─────────────────────────────────────────────────────

def _run_single_llm(query: str, mock: bool = False) -> dict[str, Any]:
    t0 = time.time()
    if mock:
        from router.llm_router import get_mock_llm
        llm = get_mock_llm("solver")
    else:
        from router.llm_router import get_llm
        llm = get_llm("solver")
    cache = KVCache()
    response = cached_llm_invoke(
        llm, "You are a medical assistant. Answer concisely with evidence-based information.",
        query, cache=cache,
    )
    elapsed = round(time.time() - t0, 2)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer, "latency": elapsed, "tokens": _estimate_tokens(answer)}


# ── Multi-agent pipeline ────────────────────────────────────────────────────

def _run_multi_agent(query: str, mock: bool = False) -> dict[str, Any]:
    get_global_cache().clear()
    t0 = time.time()
    result = run_workflow(query=query, mock=mock)
    elapsed = round(time.time() - t0, 2)
    answer = result.get("final_answer", "")
    research = result.get("research", {})
    retrieved_docs = research.get("retrieved_docs", [])
    return {
        "answer": answer,
        "latency": elapsed,
        "tokens": _estimate_tokens(answer),
        "cache_stats": result.get("token_stats", {}),
        "retry_count": result.get("retry_count", 0),
        "retrieved_docs": retrieved_docs,
    }


# ── Main benchmark runner ───────────────────────────────────────────────────

def run_benchmark(mock: bool = False, verbose: bool = False) -> list[dict[str, Any]]:
    """
    Run the full benchmark with evaluation metrics.

    Returns a list of result dicts, each containing:
      - answer_eval: answer correctness metrics
      - rag_eval: RAG retrieval quality metrics
      - single_llm / multi_agent: raw results
    """
    results = []
    print("\n" + "=" * 90)
    print("  HEALTHCARE MULTI-AGENT BENCHMARK + EVALUATION METRICS")
    print("  Mode:", "MOCK" if mock else "LIVE")
    print("=" * 90 + "\n")

    for i, problem in enumerate(BENCHMARK_PROBLEMS, 1):
        print(f"[{i}/{len(BENCHMARK_PROBLEMS)}] {problem['id']}: {problem['query'][:70]}...")

        # ── 1. Single LLM baseline ──
        print("  → Running single-LLM baseline...")
        single = _run_single_llm(problem["query"], mock=mock)
        single_eval = evaluate_answer(single["answer"], problem["query"], problem["keywords"])

        # ── 2. Multi-agent pipeline ──
        print("  → Running multi-agent system...")
        multi = _run_multi_agent(problem["query"], mock=mock)
        multi_eval = evaluate_answer(multi["answer"], problem["query"], problem["keywords"])

        # ── 3. RAG retrieval evaluation ──
        print("  → Evaluating RAG retrieval quality...")
        # Use docs from multi-agent pipeline, or do standalone retrieval
        rag_docs = multi.get("retrieved_docs", [])
        if not rag_docs:
            rag_docs = _run_rag_retrieval(problem["query"], k=5)

        rag_eval = evaluate_retrieval(rag_docs, problem.get("rag_keywords", problem["keywords"]), k=5)

        # Build result
        result = {
            "id": problem["id"],
            "type": problem["type"],
            "query": problem["query"],
            "expected": problem["expected"],
            "single_llm": {
                "answer": single["answer"][:200],
                "correct": single_eval["correct"],
                "composite_score": single_eval["composite_score"],
                "latency_s": single["latency"],
                "tokens": single["tokens"],
            },
            "multi_agent": {
                "answer": multi["answer"][:200],
                "correct": multi_eval["correct"],
                "composite_score": multi_eval["composite_score"],
                "latency_s": multi["latency"],
                "tokens": multi["tokens"],
                "cache_stats": multi.get("cache_stats", {}),
                "retry_count": multi.get("retry_count", 0),
            },
            "answer_eval": multi_eval,
            "rag_eval": rag_eval,
        }
        results.append(result)

        # Status line
        s_mark = "✓" if single_eval["correct"] else "✗"
        m_mark = "✓" if multi_eval["correct"] else "✗"
        print(f"  [OK] Single: {s_mark} ({single_eval['composite_score']:.2f}) | "
              f"Multi: {m_mark} ({multi_eval['composite_score']:.2f}) | "
              f"RAG P@5: {rag_eval['precision_at_k']:.2f} MRR: {rag_eval['mrr']:.2f}")

        # Verbose: show per-chunk breakdown
        if verbose and rag_eval["per_chunk_scores"]:
            print("  ── RAG Chunks ──")
            for chunk in rag_eval["per_chunk_scores"]:
                rel = "●" if chunk["relevance"] > 0.3 else "○"
                print(f"    {rel} [{chunk['rank']}] rel={chunk['relevance']:.2f} "
                      f"kw={chunk['keyword_coverage']} | {chunk['chunk_preview'][:80]}")

        print()

    return results


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print comparison table between single-LLM and multi-agent."""
    headers = ["ID", "Type", "S-LLM", "S-Score", "Multi", "M-Score", "P@5", "MRR", "Retries"]
    rows = []
    for r in results:
        rows.append([
            r["id"], r["type"],
            "✓" if r["single_llm"]["correct"] else "✗",
            f"{r['single_llm']['composite_score']:.2f}",
            "✓" if r["multi_agent"]["correct"] else "✗",
            f"{r['multi_agent']['composite_score']:.2f}",
            f"{r['rag_eval']['precision_at_k']:.2f}",
            f"{r['rag_eval']['mrr']:.2f}",
            r["multi_agent"]["retry_count"],
        ])

    print("\n" + "=" * 90)
    print("  COMPARISON: Single LLM vs Multi-Agent")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    single_correct = sum(1 for r in results if r["single_llm"]["correct"])
    multi_correct = sum(1 for r in results if r["multi_agent"]["correct"])
    avg_single = sum(r["single_llm"]["composite_score"] for r in results) / len(results)
    avg_multi = sum(r["multi_agent"]["composite_score"] for r in results) / len(results)

    print(f"\n  Single LLM: {single_correct}/{len(results)} correct (avg score: {avg_single:.3f})")
    print(f"  Multi-Agent: {multi_correct}/{len(results)} correct (avg score: {avg_multi:.3f})")

    improvement = ((avg_multi - avg_single) / avg_single * 100) if avg_single > 0 else 0
    print(f"  Improvement: {improvement:+.1f}%\n")


def save_results(results: list[dict[str, Any]], path: str = "eval_results.json") -> None:
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Healthcare Multi-Agent Benchmark + Evaluation")
    parser.add_argument("--mock", action="store_true", help="Use mock LLMs (no API keys)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-chunk RAG analysis")
    args = parser.parse_args()

    results = run_benchmark(mock=args.mock, verbose=args.verbose)

    # Print detailed evaluation report (answer + RAG metrics)
    print_evaluation_report(results)

    # Print comparison table
    print_results_table(results)

    # Save results
    save_results(results)
