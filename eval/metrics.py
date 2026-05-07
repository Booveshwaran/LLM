"""
Evaluation Metrics for Healthcare Multi-Agent System.

Provides two evaluation categories:

1. **Answer Correctness** — keyword F1, medical completeness, safety checks
2. **RAG Retrieval Quality** — precision@k, recall@k, MRR, NDCG, relevance

Usage:
    from eval.metrics import evaluate_answer, evaluate_retrieval, print_evaluation_report
"""

from __future__ import annotations

import re
import math
from typing import Any
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ANSWER CORRECTNESS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def keyword_precision_recall_f1(
    answer: str,
    expected_keywords: list[str],
) -> dict[str, float]:
    """
    Compute keyword-based precision, recall, and F1 against expected keywords.

    - Recall: fraction of expected keywords found in the answer
    - Precision: fraction of matched keywords among expected (capped at 1.0)
    - F1: harmonic mean of precision and recall
    """
    if not expected_keywords:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    hits = len(matched)
    total = len(expected_keywords)

    recall = hits / total if total > 0 else 0.0
    precision = hits / total if total > 0 else 0.0  # all keywords are relevant
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched_keywords": matched,
        "missed_keywords": [kw for kw in expected_keywords if kw.lower() not in answer_lower],
    }


def medical_safety_score(answer: str) -> dict[str, Any]:
    """
    Check that the answer includes required medical safety elements.

    Checks:
      - Disclaimer present ("disclaimer" or "consult" or "educational purposes")
      - No absolute claims ("always works", "guaranteed cure", "100% safe")
      - Evidence markers ("guideline", "evidence", "recommended", "first-line")
    """
    answer_lower = answer.lower()

    # Disclaimer check
    disclaimer_patterns = ["disclaimer", "consult", "educational purposes", "healthcare professional"]
    has_disclaimer = any(p in answer_lower for p in disclaimer_patterns)

    # Dangerous absolute claims check
    dangerous_patterns = [
        r"\balways works\b", r"\bguaranteed cure\b", r"\b100% safe\b",
        r"\bno side effects\b", r"\brisk.?free\b",
    ]
    dangerous_claims = [p for p in dangerous_patterns if re.search(p, answer_lower)]
    no_dangerous_claims = len(dangerous_claims) == 0

    # Evidence-based language check
    evidence_patterns = [
        "guideline", "evidence", "recommended", "first-line", "clinical",
        "study", "trial", "mg", "dose", "treatment",
    ]
    evidence_count = sum(1 for p in evidence_patterns if p in answer_lower)
    has_evidence_language = evidence_count >= 2

    # Composite score (0.0 - 1.0)
    score = 0.0
    if has_disclaimer:
        score += 0.4
    if no_dangerous_claims:
        score += 0.3
    if has_evidence_language:
        score += 0.3

    return {
        "safety_score": round(score, 2),
        "has_disclaimer": has_disclaimer,
        "no_dangerous_claims": no_dangerous_claims,
        "has_evidence_language": has_evidence_language,
        "evidence_markers_found": evidence_count,
        "dangerous_claims_found": dangerous_claims,
    }


def answer_completeness_score(
    answer: str,
    query: str,
    expected_keywords: list[str],
) -> dict[str, Any]:
    """
    Evaluate overall answer completeness.

    Components:
      - Length adequacy: penalise very short (<50 chars) or empty answers
      - Keyword coverage: F1 score against expected keywords
      - Relevance: does the answer address the query topic?
      - Safety: medical disclaimer and evidence language
    """
    # Length adequacy (0.0 - 1.0)
    length = len(answer.strip())
    if length == 0:
        length_score = 0.0
    elif length < 50:
        length_score = 0.3
    elif length < 150:
        length_score = 0.7
    else:
        length_score = 1.0

    # Keyword F1
    kw_metrics = keyword_precision_recall_f1(answer, expected_keywords)

    # Query relevance: check if key terms from query appear in answer
    query_words = set(re.findall(r"\b[a-z]{4,}\b", query.lower()))
    stopwords = {"what", "which", "when", "where", "that", "this", "with", "from", "have", "does", "most", "been"}
    query_words -= stopwords
    if query_words:
        query_overlap = sum(1 for w in query_words if w in answer.lower()) / len(query_words)
    else:
        query_overlap = 0.0

    # Safety
    safety = medical_safety_score(answer)

    # Composite score
    composite = (
        0.15 * length_score
        + 0.35 * kw_metrics["f1"]
        + 0.20 * query_overlap
        + 0.30 * safety["safety_score"]
    )

    return {
        "composite_score": round(composite, 4),
        "length_score": round(length_score, 2),
        "keyword_f1": kw_metrics["f1"],
        "query_relevance": round(query_overlap, 4),
        "safety_score": safety["safety_score"],
        "matched_keywords": kw_metrics["matched_keywords"],
        "missed_keywords": kw_metrics["missed_keywords"],
        "has_disclaimer": safety["has_disclaimer"],
        "answer_length": length,
    }


def evaluate_answer(
    answer: str,
    query: str,
    expected_keywords: list[str],
) -> dict[str, Any]:
    """
    Full answer evaluation — combines all answer metrics.

    Returns a dict with:
      - correct (bool): all expected keywords present
      - composite_score (float): weighted overall quality (0.0-1.0)
      - keyword_f1, query_relevance, safety_score, etc.
    """
    completeness = answer_completeness_score(answer, query, expected_keywords)
    correct = len(completeness["missed_keywords"]) == 0

    return {
        "correct": correct,
        **completeness,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. RAG RETRIEVAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def _keyword_relevance(doc: str, keywords: list[str]) -> float:
    """
    Score a retrieved document's relevance based on keyword overlap.

    Returns a score from 0.0 to 1.0.
    """
    if not keywords:
        return 0.0
    doc_lower = doc.lower()
    matched = sum(1 for kw in keywords if kw.lower() in doc_lower)
    return matched / len(keywords)


def precision_at_k(
    retrieved_docs: list[str],
    relevance_keywords: list[str],
    k: int = 5,
    threshold: float = 0.3,
) -> float:
    """
    Precision@k: fraction of top-k retrieved docs that are relevant.

    A doc is "relevant" if its keyword_relevance >= threshold.
    """
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for doc in top_k if _keyword_relevance(doc, relevance_keywords) >= threshold)
    return round(relevant / len(top_k), 4)


def recall_at_k(
    retrieved_docs: list[str],
    relevance_keywords: list[str],
    k: int = 5,
    threshold: float = 0.3,
) -> float:
    """
    Recall@k: fraction of relevant keywords covered by top-k docs.

    A keyword is "covered" if it appears in any of the top-k documents.
    """
    if not relevance_keywords:
        return 0.0
    top_k = retrieved_docs[:k]
    combined_text = " ".join(top_k).lower()
    covered = sum(1 for kw in relevance_keywords if kw.lower() in combined_text)
    return round(covered / len(relevance_keywords), 4)


def mean_reciprocal_rank(
    retrieved_docs: list[str],
    relevance_keywords: list[str],
    threshold: float = 0.3,
) -> float:
    """
    Mean Reciprocal Rank (MRR): 1/rank of the first relevant document.

    MRR = 1.0 means the first document retrieved is relevant.
    MRR = 0.5 means the second document is the first relevant one.
    MRR = 0.0 means no relevant document was retrieved.
    """
    for i, doc in enumerate(retrieved_docs, 1):
        if _keyword_relevance(doc, relevance_keywords) >= threshold:
            return round(1.0 / i, 4)
    return 0.0


def ndcg_at_k(
    retrieved_docs: list[str],
    relevance_keywords: list[str],
    k: int = 5,
) -> float:
    """
    Normalised Discounted Cumulative Gain (NDCG@k).

    Measures ranking quality — rewards relevant docs appearing higher.
    """
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0

    # Compute relevance scores for each position
    gains = [_keyword_relevance(doc, relevance_keywords) for doc in top_k]

    # DCG: sum of gain / log2(position + 1)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    # Ideal DCG: sort gains descending
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))

    if idcg == 0:
        return 0.0

    return round(dcg / idcg, 4)


def chunk_relevance_scores(
    retrieved_docs: list[str],
    relevance_keywords: list[str],
) -> list[dict[str, Any]]:
    """
    Score each retrieved chunk individually for detailed analysis.

    Returns a list of {chunk_preview, relevance, matched_keywords, rank}.
    """
    results = []
    for i, doc in enumerate(retrieved_docs, 1):
        doc_lower = doc.lower()
        matched = [kw for kw in relevance_keywords if kw.lower() in doc_lower]
        relevance = len(matched) / len(relevance_keywords) if relevance_keywords else 0.0

        results.append({
            "rank": i,
            "chunk_preview": doc[:120] + "..." if len(doc) > 120 else doc,
            "relevance": round(relevance, 4),
            "matched_keywords": matched,
            "keyword_coverage": f"{len(matched)}/{len(relevance_keywords)}",
        })

    return results


def evaluate_retrieval(
    retrieved_docs: list[str],
    relevance_keywords: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """
    Full RAG retrieval evaluation — combines all retrieval metrics.

    Args:
        retrieved_docs: list of text chunks returned by FAISS
        relevance_keywords: keywords that a relevant doc should contain
        k: top-k for precision/recall/NDCG

    Returns a dict with:
      - precision_at_k, recall_at_k, mrr, ndcg_at_k
      - per_chunk_scores: detailed per-document breakdown
    """
    p_at_k = precision_at_k(retrieved_docs, relevance_keywords, k=k)
    r_at_k = recall_at_k(retrieved_docs, relevance_keywords, k=k)
    mrr = mean_reciprocal_rank(retrieved_docs, relevance_keywords)
    ndcg = ndcg_at_k(retrieved_docs, relevance_keywords, k=k)
    chunks = chunk_relevance_scores(retrieved_docs, relevance_keywords)

    # F1 of precision and recall
    f1 = (2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0

    return {
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "f1_at_k": round(f1, 4),
        "mrr": mrr,
        "ndcg_at_k": ndcg,
        "k": k,
        "total_docs_retrieved": len(retrieved_docs),
        "per_chunk_scores": chunks,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_evaluation_report(results: list[dict[str, Any]]) -> None:
    """
    Print a formatted evaluation report to console.

    Args:
        results: list of per-problem dicts from run_evaluation()
    """
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    print("\n" + "=" * 90)
    print("  EVALUATION REPORT — Healthcare Multi-Agent System")
    print("=" * 90)

    # ── Answer Correctness Table ──
    print("\n  ── Answer Correctness ──\n")
    headers = ["ID", "Type", "Correct", "Composite", "KW F1", "Relevance", "Safety", "Disclaimer"]
    rows = []
    for r in results:
        a = r["answer_eval"]
        rows.append([
            r["id"], r["type"],
            "✓" if a["correct"] else "✗",
            f"{a['composite_score']:.2f}",
            f"{a['keyword_f1']:.2f}",
            f"{a['query_relevance']:.2f}",
            f"{a['safety_score']:.2f}",
            "✓" if a["has_disclaimer"] else "✗",
        ])

    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(f"  {'  '.join(headers)}")
        for row in rows:
            print(f"  {'  '.join(str(c) for c in row)}")

    # Aggregates
    correct_count = sum(1 for r in results if r["answer_eval"]["correct"])
    avg_composite = sum(r["answer_eval"]["composite_score"] for r in results) / len(results)
    avg_f1 = sum(r["answer_eval"]["keyword_f1"] for r in results) / len(results)
    avg_safety = sum(r["answer_eval"]["safety_score"] for r in results) / len(results)

    print(f"\n  Accuracy:        {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"  Avg Composite:   {avg_composite:.3f}")
    print(f"  Avg Keyword F1:  {avg_f1:.3f}")
    print(f"  Avg Safety:      {avg_safety:.3f}")

    # ── RAG Retrieval Table ──
    has_rag = any("rag_eval" in r for r in results)
    if has_rag:
        print("\n  ── RAG Retrieval Quality ──\n")
        rag_headers = ["ID", "P@5", "R@5", "F1@5", "MRR", "NDCG@5", "Docs"]
        rag_rows = []
        for r in results:
            if "rag_eval" not in r:
                continue
            g = r["rag_eval"]
            rag_rows.append([
                r["id"],
                f"{g['precision_at_k']:.2f}",
                f"{g['recall_at_k']:.2f}",
                f"{g['f1_at_k']:.2f}",
                f"{g['mrr']:.2f}",
                f"{g['ndcg_at_k']:.2f}",
                g["total_docs_retrieved"],
            ])

        if tabulate:
            print(tabulate(rag_rows, headers=rag_headers, tablefmt="grid"))
        else:
            print(f"  {'  '.join(rag_headers)}")
            for row in rag_rows:
                print(f"  {'  '.join(str(c) for c in row)}")

        rag_results = [r for r in results if "rag_eval" in r]
        avg_p = sum(r["rag_eval"]["precision_at_k"] for r in rag_results) / len(rag_results)
        avg_r = sum(r["rag_eval"]["recall_at_k"] for r in rag_results) / len(rag_results)
        avg_mrr = sum(r["rag_eval"]["mrr"] for r in rag_results) / len(rag_results)
        avg_ndcg = sum(r["rag_eval"]["ndcg_at_k"] for r in rag_results) / len(rag_results)

        print(f"\n  Avg Precision@5: {avg_p:.3f}")
        print(f"  Avg Recall@5:    {avg_r:.3f}")
        print(f"  Avg MRR:         {avg_mrr:.3f}")
        print(f"  Avg NDCG@5:      {avg_ndcg:.3f}")

    print("\n" + "=" * 90 + "\n")
