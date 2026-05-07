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

    # Query-aware solver responses keyed by clinical trigger words
    _SOLVER_OVERRIDES: dict[str, str] = {
        "chest pain": (
            '{"answer": "The presentation is consistent with acute myocardial infarction (STEMI). '
            'Immediate management: 1) Administer aspirin 325mg chewed, 2) Sublingual nitroglycerin, '
            '3) Morphine for pain, 4) Obtain 12-lead ECG, 5) Activate cath lab for PCI within 90 min. '
            'Check troponin levels. Administer heparin and dual antiplatelet therapy.\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "Acute coronary syndrome workup with guideline-directed STEMI management."}'
        ),
        "ketoacidosis": (
            '{"answer": "Diagnosis: Diabetic ketoacidosis (DKA). Blood glucose >250, pH <7.3, positive ketones '
            'confirm DKA. Treatment: 1) IV normal saline bolus (1-1.5L/hr), 2) Insulin drip at 0.1 units/kg/hr, '
            '3) Potassium replacement when K<5.3, 4) Monitor glucose hourly and electrolytes every 2-4 hours. '
            'Bicarbonate only if pH <6.9.\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "DKA management per ADA guidelines with insulin and fluid resuscitation."}'
        ),
        "ketone": (
            '{"answer": "Diagnosis: Diabetic ketoacidosis (DKA). Treatment: IV fluids, insulin drip, '
            'potassium replacement. Monitor glucose hourly.\\n\\n'
            'Disclaimer: For educational purposes only. Consult a healthcare professional.", '
            '"reasoning_trace": "DKA protocol."}'
        ),
        "type 2 diabetes": (
            '{"answer": "First-line treatment for newly diagnosed Type 2 diabetes is metformin (500mg BID, '
            'titrated to 1000mg BID). Target HbA1c <7%. If metformin alone insufficient, add SGLT2 inhibitor '
            '(empagliflozin) or GLP-1 agonist (semaglutide) — both have cardiovascular and renal benefits. '
            'Lifestyle modifications: diet, exercise, weight management.\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "ADA Standards of Care 2024 — stepwise glucose-lowering therapy."}'
        ),
        "urinary tract infection": (
            '{"answer": "For uncomplicated UTI in women: First-line is nitrofurantoin 100mg BID for 5 days. '
            'Alternative: TMP-SMX 160/800mg BID for 3 days (if local resistance <20%). '
            'Avoid fluoroquinolones for uncomplicated cystitis. Send urine culture if recurrent.\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "IDSA guidelines for uncomplicated cystitis management."}'
        ),
        "warfarin": (
            '{"answer": "Major concern: Increased bleeding risk. Ibuprofen (NSAID) inhibits platelet aggregation '
            'and can cause GI mucosal damage. Combined with warfarin anticoagulation, this significantly increases '
            'risk of GI bleeding and hemorrhage. Recommendation: Use acetaminophen instead. '
            'If NSAID required, add PPI (omeprazole) and monitor INR closely.\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "Drug interaction: warfarin + NSAID = additive bleeding risk."}'
        ),
        "tsh": (
            '{"answer": "Diagnosis: Primary hypothyroidism. TSH 12 mIU/L (elevated, normal 0.4-4.0) with '
            'free T4 0.4 ng/dL (low, normal 0.8-1.8) confirms hypothyroidism. Treatment: Levothyroxine '
            '1.6 mcg/kg/day. Recheck TSH in 6-8 weeks and titrate dose. Monitor for symptoms: fatigue, '
            'weight gain, cold intolerance, constipation.\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "Thyroid function interpretation per ATA guidelines."}'
        ),
        "anaphylaxis": (
            '{"answer": "First-line treatment: Epinephrine 0.3-0.5mg IM in the anterolateral thigh (EpiPen). '
            'Repeat every 5-15 minutes if needed. Adjunct therapy: IV fluids for hypotension, '
            'diphenhydramine 50mg IV, methylprednisolone 125mg IV, albuterol nebulizer for bronchospasm. '
            'Observe for biphasic reaction (4-6 hours minimum).\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "Anaphylaxis management per WAO/EAACI guidelines."}'
        ),
        "cancer screening": (
            '{"answer": "Cancer screening recommended starting at age 45: Colorectal cancer screening with '
            'colonoscopy every 10 years, or annual FIT/FOBT, or CT colonography every 5 years. '
            'The ACS updated guidelines in 2018 to recommend colon cancer screening beginning at age 45 '
            'for average-risk adults. Additional screenings by age: mammography, lung CT for smokers.\\n\\n'
            'Disclaimer: This is for educational purposes only. Always consult a qualified healthcare professional.", '
            '"reasoning_trace": "ACS/USPSTF colorectal cancer screening guidelines."}'
        ),
        "colonoscopy": (
            '{"answer": "Colon cancer screening: colonoscopy every 10 years starting at age 45.\\n\\n'
            'Disclaimer: For educational purposes only. Consult a healthcare professional.", '
            '"reasoning_trace": "CRC screening."}'
        ),
    }

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._call_count = 0

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """Return a mock AIMessage — query-aware for the solver agent."""
        from langchain_core.messages import AIMessage

        self._call_count += 1

        # Critic alternates: first call rejects, second approves
        if self.agent_name == "critic":
            key = "critic_reject" if self._call_count == 1 else "critic"
            return AIMessage(content=self._RESPONSES.get(key, '{"result": "mock response"}'))

        # Solver: scan the user message for clinical keywords to pick the right mock answer
        if self.agent_name == "solver":
            user_text = ""
            if isinstance(messages, list):
                for m in messages:
                    content = m.content if hasattr(m, "content") else str(m)
                    user_text += content.lower() + " "
            else:
                user_text = str(messages).lower()

            for trigger, response in self._SOLVER_OVERRIDES.items():
                if trigger in user_text:
                    return AIMessage(content=response)

        return AIMessage(content=self._RESPONSES.get(self.agent_name, '{"result": "mock response"}'))

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
