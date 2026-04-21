"""
Multi-Agent LLM Collaboration System — CLI entrypoint.

Usage:
    python main.py --query "Your question here"
    python main.py --query "test" --mock    # no API keys needed

Runs the full LangGraph workflow:
    Planner → Researcher → Critic → [Refiner loop] → Solver

Outputs: final answer, reasoning trace, KV-Cache stats, and latency.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from graph.workflow import run_workflow
from memory.kv_cache import get_global_cache


def _print_header() -> None:
    """Print a styled header."""
    print()
    print("+" + "=" * 68 + "+")
    print("|" + " Multi-Agent LLM Collaboration System ".center(68) + "|")
    print("|" + " with KV-Cache Optimization ".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print()


def _print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 4} {title} {'=' * (60 - len(title))}")


def _print_kv(key: str, value: str, indent: int = 2) -> None:
    """Print a key-value pair."""
    print(f"{' ' * indent}* {key}: {value}")


def main() -> None:
    """CLI entrypoint for the multi-agent system."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent LLM Collaboration System with KV-Cache Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --query \"Solve: A train travels 60 km/h for 2.5 hours...\"\n"
            "  python main.py --query \"test question\" --mock\n"
        ),
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The question or task to process.",
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        default=False,
        help="Use mock LLMs (no API keys needed — for testing the graph).",
    )

    args = parser.parse_args()

    _print_header()

    mode = "[MOCK] MOCK MODE (no API calls)" if args.mock else "[LIVE] LIVE MODE (using real APIs)"
    print(f"  Mode: {mode}")
    print(f"  Query: {args.query}")

    # Reset cache for clean stats
    cache = get_global_cache()
    cache.clear()

    # ── Run workflow ───────────────────────────────────────────────────
    _print_section("RUNNING PIPELINE")
    t0 = time.time()

    try:
        result = run_workflow(query=args.query, mock=args.mock)
    except Exception as exc:
        print(f"\n  [ERROR] Pipeline failed: {exc}")
        sys.exit(1)

    total_time = round(time.time() - t0, 2)

    # ── Final Answer ─────────────────────────────────────────────────
    _print_section("FINAL ANSWER")
    answer = result.get("final_answer", "No answer generated.")
    print(f"\n  {answer}\n")

    # ── Reasoning Trace ──────────────────────────────────────────────
    _print_section("REASONING TRACE")
    trace = result.get("reasoning_trace", [])
    if trace:
        for i, step in enumerate(trace, 1):
            print(f"  {i}. {step}")
    else:
        print("  (no trace available)")

    # ── KV-Cache Statistics ──────────────────────────────────────────
    _print_section("KV-CACHE STATISTICS")
    stats = cache.stats()
    _print_kv("Cache size", str(stats["cache_size"]))
    _print_kv("Total requests", str(stats["total_requests"]))
    _print_kv("Cache hits", str(stats["cache_hits"]))
    _print_kv("Cache misses", str(stats["cache_misses"]))
    _print_kv("Hit rate", f"{stats['hit_rate']:.1%}")
    _print_kv("Est. tokens saved", str(stats["estimated_tokens_saved"]))

    # Before/After comparison
    _print_section("TOKEN SAVINGS (Before vs After KV-Cache)")
    tokens_without_cache = stats["total_requests"] * 500  # rough est per call
    tokens_with_cache = tokens_without_cache - stats["estimated_tokens_saved"]
    _print_kv("Without cache (est.)", f"~{tokens_without_cache:,} tokens")
    _print_kv("With cache (est.)", f"~{tokens_with_cache:,} tokens")
    savings_pct = (
        (stats["estimated_tokens_saved"] / tokens_without_cache * 100)
        if tokens_without_cache > 0 else 0
    )
    _print_kv("Savings", f"~{stats['estimated_tokens_saved']:,} tokens ({savings_pct:.1f}%)")

    # ── Performance ──────────────────────────────────────────────────
    _print_section("PERFORMANCE")
    _print_kv("Total latency", f"{total_time}s")
    _print_kv("Retry count", str(result.get("retry_count", 0)))

    # ── Model Routing ────────────────────────────────────────────────
    _print_section("MODEL ROUTING")
    from router.llm_router import LLM_CONFIG
    for agent, cfg in LLM_CONFIG.items():
        _print_kv(
            agent.capitalize(),
            f"{cfg['model']} ({cfg['provider']})"
        )

    print(f"\n{'=' * 70}")
    print(f"  Done in {total_time}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
