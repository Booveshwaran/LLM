"""
Context Compressor — LatentMAS-inspired compressed inter-agent communication.

Implements the paper's principle of minimal information transfer between agents.
Instead of passing full verbose text, agents exchange compressed structured summaries
to reduce token overhead by ~60%, mirroring how LatentMAS agents communicate via
KV-cache working memory rather than full text.

Reference: "Latent Collaboration in Multi-Agent Systems" (2511.20639v2)
"""

from __future__ import annotations


def compress_plan(plan: dict) -> str:
    """Compress a plan dict into a concise bullet-point string."""
    steps = plan.get("steps", [])
    # Keep at most 5 steps, truncate each to 100 chars
    compressed = []
    for s in steps[:5]:
        s = s.strip()
        if len(s) > 100:
            s = s[:97] + "..."
        compressed.append(f"• {s}")
    return "\n".join(compressed)


def compress_research(research: dict) -> str:
    """Compress research output to essential summary only."""
    summary = research.get("summary", "")
    # Cap summary at 500 chars
    if len(summary) > 500:
        summary = summary[:497] + "..."
    return summary


def compress_critique(critique: dict) -> str:
    """Compress critique to score + issues only."""
    score = critique.get("score", "?")
    issues = critique.get("issues", [])
    parts = [f"Score: {score}/10"]
    for issue in issues[:3]:  # max 3 issues
        issue_text = issue.strip()
        if len(issue_text) > 120:
            issue_text = issue_text[:117] + "..."
        parts.append(f"- {issue_text}")
    return "\n".join(parts)


def compress_draft(draft: str, max_len: int = 800) -> str:
    """Truncate draft to max_len characters."""
    draft = draft.strip()
    if len(draft) > max_len:
        return draft[:max_len - 3] + "..."
    return draft


def build_solver_context(
    query: str,
    plan: dict,
    research: dict,
    draft: str,
) -> str:
    """
    Build a compressed context string for the Solver agent.

    This mirrors LatentMAS's approach where only the final agent (Judger/Solver)
    receives accumulated context and produces the text output.
    The context is kept under ~2000 chars to minimize prompt tokens.
    """
    plan_text = compress_plan(plan)
    research_text = compress_research(research)
    draft_text = compress_draft(draft, max_len=600)

    return (
        f"Query: {query}\n\n"
        f"Plan:\n{plan_text}\n\n"
        f"Research: {research_text}\n\n"
        f"Draft: {draft_text}"
    )
