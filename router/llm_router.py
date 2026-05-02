"""
Central LLM Router — factory for agent-specific LLM instances.

Routes each agent to the optimal free-tier model with LatentMAS-inspired
token budget limits to minimize generation overhead:

  Planner    -> Mistral large  (temp=0, max_tokens=300)  — concise plans
  Researcher -> NVIDIA LLaMA   (temp=0.2, max_tokens=400) — synthesis
  Critic     -> Mistral large  (temp=0, max_tokens=200)  — fast scoring
  Refiner    -> NVIDIA LLaMA   (temp=0.3, max_tokens=500) — rewriting
  Solver     -> NVIDIA LLaMA   (temp=0.3, max_tokens=800) — final answer

Inspired by: "Latent Collaboration in Multi-Agent Systems" (2511.20639v2)
  - Only the final agent (Solver) produces verbose output
  - Intermediate agents use minimal token budgets
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Model routing table with LatentMAS-inspired token budgets ────────────────
LLM_CONFIG: dict[str, dict[str, Any]] = {
    "planner": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
        "temperature": 0,
        "max_tokens": 300,
    },
    "researcher": {
        "provider": "nvidia",
        "model": "meta/llama-3.3-70b-instruct",
        "env_key": "NVIDIA_API_KEY",
        "temperature": 0.2,
        "max_tokens": 400,
    },
    "critic": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
        "temperature": 0,
        "max_tokens": 200,
    },
    "refiner": {
        "provider": "nvidia",
        "model": "meta/llama-3.3-70b-instruct",
        "env_key": "NVIDIA_API_KEY",
        "temperature": 0.3,
        "max_tokens": 500,
    },
    "solver": {
        "provider": "nvidia",
        "model": "meta/llama-3.3-70b-instruct",
        "env_key": "NVIDIA_API_KEY",
        "temperature": 0.3,
        "max_tokens": 800,
    },
}


def get_llm(agent_name: str, **kwargs: Any):
    """
    Return the correct LangChain chat model for the given agent.

    Uses per-agent temperature and max_tokens from the routing table
    to implement LatentMAS-style compressed generation.
    """
    agent_name = agent_name.lower()
    if agent_name not in LLM_CONFIG:
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Valid agents: {list(LLM_CONFIG.keys())}"
        )

    cfg = LLM_CONFIG[agent_name]
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"Missing environment variable {cfg['env_key']}. "
            f"Get a free key at the provider console."
        )

    provider = cfg["provider"]
    model = cfg["model"]
    temperature = kwargs.pop("temperature", cfg.get("temperature", 0.3))
    max_tokens = kwargs.pop("max_tokens", cfg.get("max_tokens", None))

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    raise ValueError(f"Unsupported provider: {provider}")


# ── Mock LLM for --mock mode ────────────────────────────────────────────────

class _MockChatModel:
    """
    A lightweight mock that quacks like a LangChain ChatModel.

    Returns deterministic stub responses keyed by agent name so the
    entire LangGraph pipeline can be tested end-to-end without any API keys.
    """

    _RESPONSES: dict[str, str] = {
        "planner": (
            '{"steps": ['
            '"Step 1: Identify presenting symptoms and patient history",'
            '"Step 2: Consider differential diagnoses",'
            '"Step 3: Determine relevant investigations (labs, imaging)",'
            '"Step 4: Formulate clinical assessment and treatment plan",'
            '"Step 5: Verify with clinical guidelines"'
            '], "context": "Clinical reasoning plan for patient assessment."}'
        ),
        "researcher": (
            '{"retrieved_docs": ['
            '"Hypertension is defined as systolic BP >= 140 mmHg or diastolic BP >= 90 mmHg.",'
            '"First-line treatment includes ACE inhibitors, ARBs, or thiazide diuretics.",'
            '"Lifestyle modifications: DASH diet, sodium restriction, regular exercise."'
            '], "summary": "Evidence-based guidelines recommend stepwise approach to '
            'management with lifestyle modifications as foundation and pharmacotherapy '
            'when targets are not met."}'
        ),
        "critic": (
            '{"issues": [], "score": 9, "approved": true}'
        ),
        "critic_reject": (
            '{"issues": ["Missing contraindication check for renal impairment.", '
            '"No drug interaction assessment provided."], "score": 5, "approved": false}'
        ),
        "refiner": (
            "Refined clinical assessment: Based on evidence-based guidelines, the recommended "
            "approach includes comprehensive patient evaluation, appropriate diagnostic workup, "
            "and individualized treatment plan considering comorbidities and contraindications."
        ),
        "solver": (
            '{"answer": "Based on clinical evidence, the recommended management includes: '
            '1) Lifestyle modifications (DASH diet, exercise, sodium restriction), '
            '2) First-line pharmacotherapy with ACE inhibitor or ARB, '
            '3) Regular monitoring of blood pressure and renal function. '
            'Target BP < 130/80 mmHg for most patients.\\n\\n'
            'Disclaimer: This information is for educational purposes only. '
            'Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "Applied evidence-based clinical reasoning with '
            'guideline-directed management approach."}'
        ),
    }

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._call_count = 0

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """Return a mock AIMessage."""
        from langchain_core.messages import AIMessage

        self._call_count += 1

        # Critic alternates: first call rejects, second approves
        if self.agent_name == "critic":
            key = "critic_reject" if self._call_count == 1 else "critic"
        else:
            key = self.agent_name

        return AIMessage(content=self._RESPONSES.get(key, '{"result": "mock response"}'))

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """Async variant — just delegates to sync for mocking."""
        return self.invoke(messages, **kwargs)

    def bind(self, **kwargs: Any) -> "_MockChatModel":
        """No-op bind for compatibility."""
        return self

    def __or__(self, other: Any) -> Any:
        """Support pipe syntax (llm | parser)."""
        return other


def get_mock_llm(agent_name: str) -> _MockChatModel:
    """Return a deterministic mock LLM for the given agent."""
    return _MockChatModel(agent_name.lower())
