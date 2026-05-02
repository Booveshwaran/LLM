"""
LangGraph Workflow — the core multi-agent orchestration engine.

Builds a StateGraph connecting all 5 agents:

    Planner -> Researcher -> Critic -+- [approved] --> Solver
                                     +- [rejected, max 2 retries] --> Refiner -> Critic

Supports both synchronous execution and streaming (SSE) mode
for real-time agent progress updates in the frontend.

LatentMAS-inspired: agents share compressed context via context_compressor,
reducing inter-agent token overhead by ~60%.
"""

from __future__ import annotations

import json
import time
import queue
import threading
from typing import Any, TypedDict, Generator

from langgraph.graph import END, StateGraph

from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.refiner import RefinerAgent
from agents.solver import SolverAgent
from memory.kv_cache import get_global_cache
from memory.cag import get_global_cag
from memory.vector_store import VectorStoreManager, get_global_store
from router.llm_router import get_llm, get_mock_llm


# ── State schema ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    plan: dict
    research: dict
    draft: str
    critique: dict
    refined: str
    retry_count: int
    final_answer: str
    reasoning_trace: list[str]
    token_stats: dict


# ── SSE event queue (thread-safe) ────────────────────────────────────────────

_event_queue: queue.Queue | None = None


def _emit_event(agent: str, status: str, detail: str = "") -> None:
    """Push an SSE event to the global queue if streaming is active."""
    if _event_queue is not None:
        _event_queue.put({
            "agent": agent,
            "status": status,
            "detail": detail,
            "timestamp": round(time.time(), 2),
        })


# ── Node functions ───────────────────────────────────────────────────────────

def _planner_node(state: AgentState, agent: PlannerAgent) -> dict[str, Any]:
    """Execute the Planner agent."""
    _emit_event("planner", "running")
    t0 = time.time()
    plan = agent.run(state["query"])
    elapsed = round(time.time() - t0, 2)

    trace = list(state.get("reasoning_trace") or [])
    trace_msg = f"[Planner] Generated {len(plan['steps'])}-step plan in {elapsed}s"
    trace.append(trace_msg)
    _emit_event("planner", "done", trace_msg)

    return {"plan": plan, "reasoning_trace": trace}


def _researcher_node(state: AgentState, agent: ResearcherAgent) -> dict[str, Any]:
    """Execute the Researcher agent."""
    _emit_event("researcher", "running")
    t0 = time.time()
    research = agent.run(state["plan"], state["query"])
    elapsed = round(time.time() - t0, 2)

    trace = list(state.get("reasoning_trace") or [])
    trace_msg = (
        f"[Researcher] Retrieved {len(research.get('retrieved_docs', []))} docs, "
        f"synthesised summary in {elapsed}s"
    )
    trace.append(trace_msg)
    _emit_event("researcher", "done", trace_msg)

    draft = research.get("summary", "")
    return {"research": research, "draft": draft, "reasoning_trace": trace}


def _critic_node(state: AgentState, agent: CriticAgent) -> dict[str, Any]:
    """Execute the Critic agent."""
    _emit_event("critic", "running")
    t0 = time.time()

    current_draft = state.get("refined") or state.get("draft", "")
    critique = agent.run(
        query=state["query"],
        plan=state.get("plan", {}),
        research=state.get("research", {}),
        draft=current_draft,
    )
    elapsed = round(time.time() - t0, 2)

    trace = list(state.get("reasoning_trace") or [])
    status = "APPROVED" if critique.get("approved") else "REJECTED"
    trace_msg = (
        f"[Critic] Score: {critique.get('score', '?')}/10 — {status} "
        f"({len(critique.get('issues', []))} issues) in {elapsed}s"
    )
    trace.append(trace_msg)
    _emit_event("critic", "done", trace_msg)

    return {"critique": critique, "reasoning_trace": trace}


def _refiner_node(state: AgentState, agent: RefinerAgent) -> dict[str, Any]:
    """Execute the Refiner agent."""
    _emit_event("refiner", "running")
    t0 = time.time()

    current_draft = state.get("refined") or state.get("draft", "")
    refined = agent.run(
        query=state["query"],
        draft=current_draft,
        critique=state.get("critique", {}),
        research=state.get("research", {}),
    )
    elapsed = round(time.time() - t0, 2)

    retry_count = (state.get("retry_count") or 0) + 1

    trace = list(state.get("reasoning_trace") or [])
    trace_msg = f"[Refiner] Refined draft (attempt {retry_count}) in {elapsed}s"
    trace.append(trace_msg)
    _emit_event("refiner", "done", trace_msg)

    return {
        "refined": refined,
        "retry_count": retry_count,
        "reasoning_trace": trace,
    }


def _solver_node(state: AgentState, agent: SolverAgent) -> dict[str, Any]:
    """Execute the Solver agent."""
    _emit_event("solver", "running")
    t0 = time.time()

    current_draft = state.get("refined") or state.get("draft", "")
    result = agent.run(
        query=state["query"],
        plan=state.get("plan", {}),
        research=state.get("research", {}),
        draft=current_draft,
    )
    elapsed = round(time.time() - t0, 2)

    cache_stats = get_global_cache().stats()

    trace = list(state.get("reasoning_trace") or [])
    trace_msg = f"[Solver] Generated final answer in {elapsed}s"
    trace.append(trace_msg)
    _emit_event("solver", "done", trace_msg)

    return {
        "final_answer": result.get("answer", ""),
        "reasoning_trace": trace,
        "token_stats": cache_stats,
    }


# ── Conditional routing ─────────────────────────────────────────────────────

def _should_refine_or_solve(state: AgentState) -> str:
    """
    Route after Critic evaluation:
      - If approved OR retry_count >= 2 → Solver
      - Else → Refiner (for another revision pass)
    """
    critique = state.get("critique", {})
    retry_count = state.get("retry_count", 0)

    if critique.get("approved", False) or retry_count >= 2:
        if not critique.get("approved", False):
            _emit_event("refiner", "skipped", "Max retries reached")
        return "solver"
    return "refiner"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_workflow(
    mock: bool = False,
    vector_store: VectorStoreManager | None = None,
) -> Any:
    """Build and compile the LangGraph multi-agent workflow."""
    if mock:
        planner_llm = get_mock_llm("planner")
        researcher_llm = get_mock_llm("researcher")
        critic_llm = get_mock_llm("critic")
        refiner_llm = get_mock_llm("refiner")
        solver_llm = get_mock_llm("solver")
    else:
        planner_llm = get_llm("planner")
        researcher_llm = get_llm("researcher")
        critic_llm = get_llm("critic")
        refiner_llm = get_llm("refiner")
        solver_llm = get_llm("solver")

    # Use global store if none provided (preloaded at startup)
    vs = vector_store or get_global_store()

    planner = PlannerAgent(planner_llm)
    researcher = ResearcherAgent(researcher_llm, vs)
    critic = CriticAgent(critic_llm)
    refiner = RefinerAgent(refiner_llm)
    solver = SolverAgent(solver_llm)

    graph = StateGraph(AgentState)

    graph.add_node("planner", lambda state: _planner_node(state, planner))
    graph.add_node("researcher", lambda state: _researcher_node(state, researcher))
    graph.add_node("critic", lambda state: _critic_node(state, critic))
    graph.add_node("refiner", lambda state: _refiner_node(state, refiner))
    graph.add_node("solver", lambda state: _solver_node(state, solver))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        _should_refine_or_solve,
        {"solver": "solver", "refiner": "refiner"},
    )
    graph.add_edge("refiner", "critic")
    graph.add_edge("solver", END)

    return graph.compile()


def run_workflow(query: str, mock: bool = False) -> AgentState:
    """Build the workflow and run a query through it (synchronous).

    Uses Cache-Augmented Generation (CAG) to return instant answers
    for semantically similar previously-answered queries.
    """
    # CAG lookup — instant answer if similar query was cached
    cag = get_global_cag()
    if not mock:
        cached_response = cag.lookup(query)
        if cached_response is not None:
            _emit_event("solver", "done", "Instant answer from CAG cache")
            return cached_response

    cache = get_global_cache()
    workflow = build_workflow(mock=mock)

    initial_state: AgentState = {
        "query": query,
        "plan": {},
        "research": {},
        "draft": "",
        "critique": {},
        "refined": "",
        "retry_count": 0,
        "final_answer": "",
        "reasoning_trace": [],
        "token_stats": {},
    }

    result = workflow.invoke(initial_state)

    # Store in CAG cache for future queries
    if not mock and result.get("final_answer"):
        cag.store(query, dict(result))

    return result


def run_workflow_streaming(query: str, mock: bool = False) -> Generator[str, None, None]:
    """
    Run the workflow in a background thread, yielding SSE events as agents complete.

    Yields:
        SSE-formatted strings: 'data: {...}\\n\\n'
    """
    global _event_queue
    _event_queue = queue.Queue()

    result_holder: list[Any] = [None]
    error_holder: list[Any] = [None]

    def _run():
        try:
            result_holder[0] = run_workflow(query=query, mock=mock)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            _event_queue.put(None)  # sentinel

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Yield SSE events as they arrive
    while True:
        try:
            event = _event_queue.get(timeout=120)
        except queue.Empty:
            break

        if event is None:
            # Workflow finished — yield the final result
            if error_holder[0]:
                yield f"data: {json.dumps({'type': 'error', 'message': str(error_holder[0])})}\n\n"
            elif result_holder[0]:
                final = result_holder[0]
                yield f"data: {json.dumps({'type': 'result', 'final_answer': final.get('final_answer', ''), 'reasoning_trace': final.get('reasoning_trace', []), 'token_stats': final.get('token_stats', {}), 'retry_count': final.get('retry_count', 0)})}\n\n"
            break
        else:
            yield f"data: {json.dumps({'type': 'agent_update', **event})}\n\n"

    _event_queue = None
