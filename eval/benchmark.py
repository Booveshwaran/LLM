"""
Healthcare Benchmark — evaluate multi-agent clinical system vs single-LLM.

Tests on 8 clinical scenarios: diagnosis, treatment, drug interactions,
lab interpretation, emergency management, and preventive care.
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

BENCHMARK_PROBLEMS: list[dict[str, Any]] = [
    {"id": "dx_1", "type": "diagnosis", "query": "A 55-year-old male presents with crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath. What is the most likely diagnosis and immediate management?", "expected": "myocardial infarction", "keywords": ["myocardial infarction", "aspirin"]},
    {"id": "dx_2", "type": "diagnosis", "query": "A patient with diabetes has blood glucose of 450 mg/dL, pH 7.2, and positive ketones. What is the diagnosis and treatment?", "expected": "DKA", "keywords": ["ketoacidosis", "insulin"]},
    {"id": "tx_1", "type": "treatment", "query": "What is the first-line treatment for newly diagnosed Type 2 diabetes?", "expected": "metformin", "keywords": ["metformin"]},
    {"id": "tx_2", "type": "treatment", "query": "What is the recommended treatment for uncomplicated urinary tract infection in women?", "expected": "nitrofurantoin", "keywords": ["nitrofurantoin"]},
    {"id": "drug_1", "type": "drug_interaction", "query": "A patient on warfarin is prescribed ibuprofen. What is the major concern?", "expected": "bleeding", "keywords": ["bleeding"]},
    {"id": "lab_1", "type": "lab_interpretation", "query": "A patient has TSH of 12 mIU/L and free T4 of 0.4 ng/dL. What is the diagnosis?", "expected": "hypothyroidism", "keywords": ["hypothyroidism"]},
    {"id": "emerg_1", "type": "emergency", "query": "A patient is experiencing anaphylaxis. What is the first-line treatment and dose?", "expected": "epinephrine", "keywords": ["epinephrine"]},
    {"id": "prev_1", "type": "prevention", "query": "What cancer screening is recommended starting at age 45?", "expected": "colonoscopy", "keywords": ["colon"]},
]

def _check_correctness(answer: str, problem: dict[str, Any]) -> bool:
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in problem["keywords"])

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def _run_single_llm(query: str, mock: bool = False) -> dict[str, Any]:
    t0 = time.time()
    if mock:
        from router.llm_router import get_mock_llm
        llm = get_mock_llm("solver")
    else:
        from router.llm_router import get_llm
        llm = get_llm("solver")
    cache = KVCache()
    response = cached_llm_invoke(llm, "You are a medical assistant. Answer concisely with evidence-based information.", query, cache=cache)
    elapsed = round(time.time() - t0, 2)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer, "latency": elapsed, "tokens": _estimate_tokens(answer)}

def _run_multi_agent(query: str, mock: bool = False) -> dict[str, Any]:
    get_global_cache().clear()
    t0 = time.time()
    result = run_workflow(query=query, mock=mock)
    elapsed = round(time.time() - t0, 2)
    answer = result.get("final_answer", "")
    return {"answer": answer, "latency": elapsed, "tokens": _estimate_tokens(answer), "cache_stats": result.get("token_stats", {}), "retry_count": result.get("retry_count", 0)}

def run_benchmark(mock: bool = False) -> list[dict[str, Any]]:
    results = []
    print("\n" + "=" * 80)
    print("  HEALTHCARE MULTI-AGENT BENCHMARK")
    print("  Mode:", "MOCK" if mock else "LIVE")
    print("=" * 80 + "\n")
    for i, problem in enumerate(BENCHMARK_PROBLEMS, 1):
        print(f"[{i}/{len(BENCHMARK_PROBLEMS)}] {problem['id']}: {problem['query'][:60]}...")
        print("  -> Running single-LLM baseline...")
        single = _run_single_llm(problem["query"], mock=mock)
        single_correct = _check_correctness(single["answer"], problem)
        print("  -> Running multi-agent system...")
        multi = _run_multi_agent(problem["query"], mock=mock)
        multi_correct = _check_correctness(multi["answer"], problem)
        result = {"id": problem["id"], "type": problem["type"], "query": problem["query"], "expected": problem["expected"],
                  "single_llm": {"answer": single["answer"][:200], "correct": single_correct, "latency_s": single["latency"], "tokens": single["tokens"]},
                  "multi_agent": {"answer": multi["answer"][:200], "correct": multi_correct, "latency_s": multi["latency"], "tokens": multi["tokens"], "cache_stats": multi.get("cache_stats", {}), "retry_count": multi.get("retry_count", 0)}}
        results.append(result)
        print(f"  [OK] Single: {'CORRECT' if single_correct else 'WRONG'} | Multi: {'CORRECT' if multi_correct else 'WRONG'}\n")
    return results

def print_results_table(results: list[dict[str, Any]]) -> None:
    headers = ["ID", "Type", "Single OK", "Single Time", "Multi OK", "Multi Time", "Retries"]
    rows = []
    for r in results:
        rows.append([r["id"], r["type"], "Y" if r["single_llm"]["correct"] else "N", f"{r['single_llm']['latency_s']:.2f}s",
                     "Y" if r["multi_agent"]["correct"] else "N", f"{r['multi_agent']['latency_s']:.2f}s", r["multi_agent"]["retry_count"]])
    print("\n" + "=" * 80)
    print("  HEALTHCARE BENCHMARK RESULTS")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    single_correct = sum(1 for r in results if r["single_llm"]["correct"])
    multi_correct = sum(1 for r in results if r["multi_agent"]["correct"])
    print(f"\n  Single LLM: {single_correct}/{len(results)} correct")
    print(f"  Multi-Agent: {multi_correct}/{len(results)} correct\n")

def save_results(results: list[dict[str, Any]], path: str = "eval_results.json") -> None:
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Healthcare Multi-Agent Benchmark")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    results = run_benchmark(mock=args.mock)
    print_results_table(results)
    save_results(results)
