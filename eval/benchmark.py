"""
Benchmark — evaluate multi-agent system vs. single-LLM baseline.

Tests on 5 GSM8K math problems + 3 HumanEval-style coding problems.
Compares:
  - Single LLM (Groq llama direct)
  - Full multi-agent system

Metrics per question: correctness, token count estimate, latency.
Prints a formatted comparison table and saves results to eval_results.json.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from typing import Any

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from tabulate import tabulate

from graph.workflow import run_workflow
from memory.kv_cache import cached_llm_invoke, get_global_cache, KVCache


# ── Benchmark problems ──────────────────────────────────────────────────────

BENCHMARK_PROBLEMS: list[dict[str, Any]] = [
    # GSM8K-style math problems
    {
        "id": "gsm8k_1",
        "type": "math",
        "query": (
            "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
            "What is the total distance traveled?"
        ),
        "expected": "270",
        "keywords": ["270"],
    },
    {
        "id": "gsm8k_2",
        "type": "math",
        "query": (
            "Janet has 3 times as many marbles as Tom. Tom has 12 marbles. "
            "Janet gives half of her marbles to Tom. How many marbles does Tom have now?"
        ),
        "expected": "30",
        "keywords": ["30"],
    },
    {
        "id": "gsm8k_3",
        "type": "math",
        "query": (
            "A store sells apples for $2 each and oranges for $3 each. "
            "If Mark buys 5 apples and 4 oranges, how much does he spend in total?"
        ),
        "expected": "22",
        "keywords": ["22"],
    },
    {
        "id": "gsm8k_4",
        "type": "math",
        "query": (
            "A swimming pool is being filled at 50 liters per minute. "
            "If the pool holds 3000 liters and is currently 1/3 full, "
            "how many minutes will it take to fill the rest?"
        ),
        "expected": "40",
        "keywords": ["40"],
    },
    {
        "id": "gsm8k_5",
        "type": "math",
        "query": (
            "A baker makes 120 cookies. She packs them in boxes of 8. "
            "She sells each box for $5. She gives away 3 boxes for free. "
            "How much money does she make?"
        ),
        "expected": "60",
        "keywords": ["60"],
    },
    # HumanEval-style coding problems
    {
        "id": "code_1",
        "type": "code",
        "query": (
            "Write a Python function called `is_palindrome(s)` that returns True "
            "if the string s is a palindrome (case-insensitive, ignoring spaces), "
            "False otherwise."
        ),
        "expected": "def is_palindrome",
        "keywords": ["def is_palindrome", "return", "lower"],
    },
    {
        "id": "code_2",
        "type": "code",
        "query": (
            "Write a Python function called `fibonacci(n)` that returns the nth "
            "Fibonacci number (0-indexed, so fibonacci(0)=0, fibonacci(1)=1, fibonacci(6)=8)."
        ),
        "expected": "def fibonacci",
        "keywords": ["def fibonacci", "return"],
    },
    {
        "id": "code_3",
        "type": "code",
        "query": (
            "Write a Python function called `flatten(lst)` that takes a nested list "
            "and returns a flat list. For example, flatten([1, [2, [3, 4], 5], 6]) "
            "should return [1, 2, 3, 4, 5, 6]."
        ),
        "expected": "def flatten",
        "keywords": ["def flatten", "return"],
    },
]


def _check_correctness(answer: str, problem: dict[str, Any]) -> bool:
    """Check if the answer contains the expected keywords."""
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in problem["keywords"])


def _estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return max(1, len(text) // 4)


def _run_single_llm(query: str, mock: bool = False) -> dict[str, Any]:
    """Run a query through a single LLM (Groq llama) for baseline comparison."""
    t0 = time.time()

    if mock:
        from router.llm_router import get_mock_llm
        llm = get_mock_llm("solver")
    else:
        from router.llm_router import get_llm
        llm = get_llm("solver")

    cache = KVCache()  # Separate cache for baseline
    response = cached_llm_invoke(
        llm,
        "You are a helpful assistant. Answer the question directly and concisely.",
        query,
        cache=cache,
    )

    elapsed = round(time.time() - t0, 2)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "latency": elapsed,
        "tokens": _estimate_tokens(answer),
    }


def _run_multi_agent(query: str, mock: bool = False) -> dict[str, Any]:
    """Run a query through the full multi-agent pipeline."""
    # Reset global cache for clean measurement
    get_global_cache().clear()

    t0 = time.time()
    result = run_workflow(query=query, mock=mock)
    elapsed = round(time.time() - t0, 2)

    answer = result.get("final_answer", "")

    return {
        "answer": answer,
        "latency": elapsed,
        "tokens": _estimate_tokens(answer),
        "cache_stats": result.get("token_stats", {}),
        "retry_count": result.get("retry_count", 0),
    }


def run_benchmark(mock: bool = False) -> list[dict[str, Any]]:
    """
    Run the full benchmark suite.

    Args:
        mock: If True, uses mock LLMs (no API keys needed).

    Returns:
        List of result dicts, one per problem.
    """
    results = []

    print("\n" + "=" * 80)
    print("  MULTI-AGENT LLM BENCHMARK")
    print("  Mode:", "MOCK" if mock else "LIVE")
    print("=" * 80 + "\n")

    for i, problem in enumerate(BENCHMARK_PROBLEMS, 1):
        print(f"[{i}/{len(BENCHMARK_PROBLEMS)}] {problem['id']}: {problem['query'][:60]}...")

        # Single LLM baseline
        print("  -> Running single-LLM baseline...")
        single = _run_single_llm(problem["query"], mock=mock)
        single_correct = _check_correctness(single["answer"], problem)

        # Multi-agent system
        print("  -> Running multi-agent system...")
        multi = _run_multi_agent(problem["query"], mock=mock)
        multi_correct = _check_correctness(multi["answer"], problem)

        result = {
            "id": problem["id"],
            "type": problem["type"],
            "query": problem["query"],
            "expected": problem["expected"],
            "single_llm": {
                "answer": single["answer"][:200],
                "correct": single_correct,
                "latency_s": single["latency"],
                "tokens": single["tokens"],
            },
            "multi_agent": {
                "answer": multi["answer"][:200],
                "correct": multi_correct,
                "latency_s": multi["latency"],
                "tokens": multi["tokens"],
                "cache_stats": multi.get("cache_stats", {}),
                "retry_count": multi.get("retry_count", 0),
            },
        }
        results.append(result)
        print(f"  [OK] Single: {'CORRECT' if single_correct else 'WRONG'} | "
              f"Multi: {'CORRECT' if multi_correct else 'WRONG'}\n")

    return results


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    headers = [
        "ID", "Type",
        "Single Correct", "Single Latency", "Single Tokens",
        "Multi Correct", "Multi Latency", "Multi Tokens", "Retries",
    ]

    rows = []
    for r in results:
        rows.append([
            r["id"],
            r["type"],
            "Y" if r["single_llm"]["correct"] else "N",
            f"{r['single_llm']['latency_s']:.2f}s",
            r["single_llm"]["tokens"],
            "Y" if r["multi_agent"]["correct"] else "N",
            f"{r['multi_agent']['latency_s']:.2f}s",
            r["multi_agent"]["tokens"],
            r["multi_agent"]["retry_count"],
        ])

    print("\n" + "=" * 80)
    print("  RESULTS COMPARISON")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Summary stats
    single_correct = sum(1 for r in results if r["single_llm"]["correct"])
    multi_correct = sum(1 for r in results if r["multi_agent"]["correct"])
    single_latency = sum(r["single_llm"]["latency_s"] for r in results)
    multi_latency = sum(r["multi_agent"]["latency_s"] for r in results)

    print(f"\n  Summary:")
    print(f"    Single LLM: {single_correct}/{len(results)} correct, "
          f"total latency: {single_latency:.2f}s")
    print(f"    Multi-Agent: {multi_correct}/{len(results)} correct, "
          f"total latency: {multi_latency:.2f}s")
    print()


def save_results(results: list[dict[str, Any]], path: str = "eval_results.json") -> None:
    """Save results to a JSON file."""
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        path,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent LLM Benchmark")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLMs (no API keys needed)",
    )
    args = parser.parse_args()

    results = run_benchmark(mock=args.mock)
    print_results_table(results)
    save_results(results)
