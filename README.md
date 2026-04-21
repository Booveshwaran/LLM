# Multi-Agent LLM Collaboration System

A production-ready multi-agent reasoning framework where specialised agents collaborate through a **LangGraph** workflow to solve complex tasks better than any single LLM. Features **KV-Cache** optimisation, **RAG** retrieval with FAISS, and intelligent routing across **three free-tier LLMs**.

---

## Architecture

```
User Query
    │
    ▼
┌─────────┐    ┌────────────┐    ┌────────┐    ┌────────┐
│ Planner  │───▶│ Researcher │───▶│ Critic │───▶│ Solver │───▶ Final Answer
│ (Mistral)│    │  (NVIDIA)  │    │(Mistral)│    │(NVIDIA)│
└─────────┘    └────────────┘    └───┬─────┘    └────────┘
                                     │  ▲
                                     │  │ (max 2 retries)
                                     ▼  │
                                 ┌─────────┐
                                 │ Refiner  │
                                 │ (NVIDIA) │
                                 └─────────┘
```

## LLM Routing (All Free Tier)

| Agent      | Model                         | Provider | Why                          |
|------------|-------------------------------|----------|------------------------------|
| Planner    | mistral-large-latest          | Mistral  | Best reasoning, free tier    |
| Researcher | meta/llama-3.3-70b-instruct   | NVIDIA   | Strong retrieval synthesis   |
| Critic     | mistral-large-latest          | Mistral  | Strong critical reasoning    |
| Refiner    | meta/llama-3.3-70b-instruct   | NVIDIA   | Good at rewriting            |
| Solver     | meta/llama-3.3-70b-instruct   | NVIDIA   | Fast final answers via build.nvidia.com |

---

## Getting Free API Keys

1. **Mistral AI** — [console.mistral.ai](https://console.mistral.ai)
   - Sign up → API Keys → Create new key
   - Free tier includes `mistral-large-latest`

2. **NVIDIA** — [build.nvidia.com](https://build.nvidia.com)
   - Sign up → API Keys → Generate key
   - Free tier includes `meta/llama-3.3-70b-instruct`

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your two free API keys:

```env
MISTRAL_API_KEY=your_mistral_key_here
NVIDIA_API_KEY=your_nvidia_key_here
```

### 3. Run the CLI

```bash
python main.py --query "Solve: A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. What is the total distance?"
```

### 4. Run with mock mode (no API keys needed)

```bash
python main.py --query "test question" --mock
```

This runs the entire LangGraph pipeline with deterministic stub responses — perfect for testing the graph logic without any API calls.

### 5. Start the REST API

```bash
uvicorn api.app:app --reload
```

Then visit:
- **POST** `http://localhost:8000/query` — run a query (body: `{"query": "...", "mock": false}`)
- **GET** `http://localhost:8000/cache-stats` — KV-Cache statistics
- **GET** `http://localhost:8000/health` — health check with model info
- **Docs** `http://localhost:8000/docs` — interactive Swagger UI

### 6. Run benchmarks

```bash
python eval/benchmark.py          # with real APIs
python eval/benchmark.py --mock   # with mock LLMs
```

Evaluates on 5 GSM8K math problems + 3 HumanEval-style coding problems, comparing single-LLM baseline vs full multi-agent system.

---

## Running the UI

### Start the API server

```bash
uvicorn api.app:app --reload --port 8000
```

### Open the frontend

Simply double-click `frontend/index.html` in your file explorer, or open it directly:

```
file:///path/to/multi_agent_llm/frontend/index.html
```

### Test without API keys

Check the **"Mock Mode"** toggle in the UI before clicking **Run**. This returns instant stub results without making any LLM calls.

### Test with real LLMs

Make sure your `.env` file has valid API keys and the API server is running on port 8000, then leave Mock Mode unchecked.

### Keyboard shortcut

Press **Ctrl+Enter** in the query box to run the query.

---

## Features

### Multi-Agent Orchestration
- **5 specialised agents** with distinct roles and optimal model assignments
- **LangGraph StateGraph** with conditional edges and retry logic
- **Critic loop** — up to 2 revision cycles before forcing solver output
- **Full reasoning trace** — every agent action is logged with timestamps

### KV-Cache Optimisation
- SHA-256 keyed prompt prefix cache
- LRU eviction policy (default 256 entries)
- Thread-safe with statistics tracking
- Before/after token savings displayed in CLI output

### RAG Retrieval
- **FAISS** vector store with **sentence-transformers** (all-MiniLM-L6-v2)
- 15 curated documents covering AI reasoning, math, debugging, and logic
- Persisted to disk at `./faiss_index/`
- Augmented with **DuckDuckGo** web search (no API key needed)

### REST API
- FastAPI with Pydantic validation
- Interactive Swagger docs at `/docs`
- Cache statistics endpoint
- CORS enabled for frontend access

### Frontend UI
- Dark-themed, responsive single-page interface
- Real-time agent pipeline tracker with animated status cards
- Mock mode toggle for testing without API keys
- Collapsible reasoning trace and cache statistics display
- Zero dependencies — works by opening `index.html` directly

---

## Project Structure

```
multi_agent_llm/
├── frontend/
│   └── index.html       # Self-contained dark-themed UI
├── agents/
│   ├── __init__.py
│   ├── planner.py       # Decomposes queries into plans (Mistral)
│   ├── researcher.py    # RAG + web search (NVIDIA LLaMA)
│   ├── critic.py        # Evaluates reasoning quality (Mistral)
│   ├── refiner.py       # Rewrites based on feedback (NVIDIA LLaMA)
│   └── solver.py        # Final answer generation (NVIDIA LLaMA)
├── router/
│   ├── __init__.py
│   └── llm_router.py    # Central LLM factory + mock LLMs
├── memory/
│   ├── __init__.py
│   ├── kv_cache.py      # SHA-256 prompt prefix cache with stats
│   └── vector_store.py  # FAISS + HuggingFace embeddings + default corpus
├── graph/
│   ├── __init__.py
│   └── workflow.py      # LangGraph StateGraph orchestration
├── api/
│   ├── __init__.py
│   └── app.py           # FastAPI REST interface (with CORS)
├── eval/
│   └── benchmark.py     # Benchmark suite with comparison tables
├── main.py              # CLI entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

---

## Technical Details

### State Schema (TypedDict)

```python
class AgentState(TypedDict):
    query: str              # Original user query
    plan: dict              # Planner output: {"steps": [...], "context": "..."}
    research: dict          # Researcher output: {"retrieved_docs": [...], "summary": "..."}
    draft: str              # Initial draft from research
    critique: dict          # Critic output: {"issues": [...], "score": 0-10, "approved": bool}
    refined: str            # Refiner output: improved draft
    retry_count: int        # Number of refine-critic loops completed
    final_answer: str       # Solver output
    reasoning_trace: list   # Timestamped log of all agent actions
    token_stats: dict       # KV-Cache statistics snapshot
```

### Environment Variables

| Variable           | Required | Source                         |
|--------------------|----------|--------------------------------|
| `MISTRAL_API_KEY`  | Yes*     | [console.mistral.ai](https://console.mistral.ai) |
| `NVIDIA_API_KEY`   | Yes*     | [build.nvidia.com](https://build.nvidia.com) |

*Not required when using `--mock` mode.

---

## License

MIT
