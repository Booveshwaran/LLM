"""
FAISS Vector Store — local RAG retrieval with sentence-transformers embeddings.

Uses HuggingFaceEmbeddings with all-MiniLM-L6-v2 for free, local embedding.
Persists the FAISS index to disk at ./faiss_index/.

On first run, ingests a default corpus of 25+ curated text chunks covering
AI reasoning, math, code, logic, science, and general knowledge.

Supports eager preloading via preload_global_store() for zero-latency first queries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Default index persistence path
_DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")

# ── Default corpus (25+ documents) ──────────────────────────────────────────
DEFAULT_CORPUS: list[str] = [
    # AI Reasoning
    (
        "Chain-of-thought prompting significantly improves large language model reasoning "
        "by encouraging the model to decompose complex problems into intermediate steps "
        "before arriving at a final answer."
    ),
    (
        "Multi-agent collaboration in LLM systems works by assigning specialised roles — "
        "such as planner, researcher, critic, and solver — to different model instances, "
        "then orchestrating their outputs through a state graph."
    ),
    (
        "Self-consistency sampling generates multiple reasoning paths and selects the most "
        "common answer, which helps reduce hallucination and improve reliability in "
        "complex reasoning tasks."
    ),
    (
        "Tree-of-thought reasoning extends chain-of-thought by exploring multiple branches "
        "of reasoning simultaneously and using search algorithms to identify the most "
        "promising solution path."
    ),
    (
        "LatentMAS enables LLM agents to collaborate entirely in latent space by sharing "
        "KV-cache working memory instead of text, achieving 4x faster inference with "
        "70-84% fewer tokens while improving accuracy by up to 14.6%."
    ),
    # Math Problem Solving
    (
        "When solving a word problem, first identify all given quantities and unknowns, "
        "then translate the narrative into mathematical equations before performing any "
        "calculations."
    ),
    (
        "The GSM8K benchmark tests grade-school math reasoning. A common strategy is to "
        "show intermediate arithmetic: e.g., 'If a train travels 60 km/h for 2.5 hours, "
        "distance = 60 × 2.5 = 150 km.'"
    ),
    (
        "Dimensional analysis is a powerful verification tool: always check that the units "
        "on both sides of an equation match before accepting a numerical result."
    ),
    (
        "Estimation and sanity checks help catch errors in multi-step math problems. "
        "Before finalising an answer, verify that it falls within an intuitive range."
    ),
    (
        "Percentage calculations: to find X% of Y, compute (X/100) × Y. For discounts, "
        "subtract the discount amount from the original. For tax, add tax amount to the "
        "discounted price."
    ),
    (
        "When dividing items equally, use integer division for whole items. If 7 apples "
        "are shared among 3 people, each gets 2 with 1 remaining (7 ÷ 3 = 2 remainder 1)."
    ),
    # Code & Programming
    (
        "Systematic debugging starts with reproducing the bug, then isolating the smallest "
        "code change that triggers the error. Use binary search on recent commits if the "
        "regression is unclear."
    ),
    (
        "Common Python runtime errors include TypeError (wrong argument type), KeyError "
        "(missing dict key), and IndexError (list index out of range). Each has distinct "
        "debugging strategies."
    ),
    (
        "Static analysis tools like mypy, pylint, and ruff catch type mismatches and style "
        "violations before runtime. Integrating them into CI prevents a large class of bugs."
    ),
    (
        "A palindrome reads the same forwards and backwards. To check: compare the string "
        "with its reverse. In Python: s == s[::-1]. Handle case-insensitivity by converting "
        "to lowercase first."
    ),
    (
        "The Fibonacci sequence: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2). An iterative approach "
        "runs in O(n) time, while naive recursion has O(2^n) complexity. Use memoization "
        "or dynamic programming for efficiency."
    ),
    # Logic & Reasoning
    (
        "Modus ponens: if P implies Q, and P is true, then Q must be true. This is the "
        "foundational rule of deductive reasoning used in formal logic."
    ),
    (
        "Proof by contradiction assumes the negation of the statement to be proved, then "
        "derives a logical contradiction, thereby establishing the original statement."
    ),
    (
        "Inductive reasoning generalises from specific observations to broader rules. "
        "While powerful, it requires caution — a single counter-example can invalidate "
        "an inductive conclusion."
    ),
    # Science
    (
        "Newton's second law: Force = mass × acceleration (F = ma). This fundamental "
        "equation relates how forces cause objects to accelerate, measured in Newtons (N)."
    ),
    (
        "The speed of light in vacuum is approximately 3 × 10^8 meters per second. "
        "Einstein's E = mc² shows mass-energy equivalence, where c is the speed of light."
    ),
    (
        "Water boils at 100°C (212°F) at standard atmospheric pressure. The boiling point "
        "decreases at higher altitudes due to lower atmospheric pressure."
    ),
    # General Knowledge
    (
        "The Earth orbits the Sun at approximately 150 million kilometers (1 AU). One orbit "
        "takes 365.25 days, which is why we have leap years every 4 years."
    ),
    (
        "The human brain contains approximately 86 billion neurons. Neural networks in AI "
        "are loosely inspired by biological neural connections but differ fundamentally "
        "in their operation."
    ),
    (
        "The scientific method involves: observation, hypothesis formation, experimentation, "
        "data analysis, and conclusion. Reproducibility is essential for validating results."
    ),
    (
        "When debugging performance issues, use profiling tools (cProfile, line_profiler) "
        "to identify hotspots before optimising, because premature optimisation misdirects "
        "effort."
    ),
]


class VectorStoreManager:
    """
    Manages a FAISS vector store with HuggingFace sentence-transformer embeddings.

    Supports:
      - Ingesting texts as documents
      - Similarity search (top-k, default k=5)
      - Persisting/loading the index to/from disk
      - Seeding with a default corpus on first use
      - Eager preloading via preload() for server startup
    """

    def __init__(
        self,
        index_dir: str = _DEFAULT_INDEX_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model
        self._embeddings: Any = None
        self._store: Any = None

    # ── Lazy init ────────────────────────────────────────────────────────

    def _get_embeddings(self) -> Any:
        """Lazily initialise the HuggingFace embedding model."""
        if self._embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _load_or_create_store(self) -> Any:
        """Load an existing FAISS index from disk, or create a new one."""
        if self._store is not None:
            return self._store

        from langchain_community.vectorstores import FAISS

        index_path = Path(self.index_dir)
        embeddings = self._get_embeddings()

        if (index_path / "index.faiss").exists():
            self._store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # Seed with the default corpus
            self._store = FAISS.from_texts(DEFAULT_CORPUS, embeddings)
            self.save()

        return self._store

    def preload(self) -> None:
        """Eagerly initialise embeddings and load/create the FAISS index."""
        self._load_or_create_store()

    # ── Public API ───────────────────────────────────────────────────────

    def add_documents(self, texts: list[str]) -> int:
        """Add texts to the vector store."""
        store = self._load_or_create_store()
        store.add_texts(texts)
        self.save()
        return len(texts)

    def similarity_search(self, query: str, k: int = 5) -> list[str]:
        """Retrieve the top-k most similar documents for a query."""
        store = self._load_or_create_store()
        docs = store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        if self._store is not None:
            index_path = Path(self.index_dir)
            index_path.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(index_path))

    def document_count(self) -> int:
        """Return the number of documents in the store."""
        store = self._load_or_create_store()
        return store.index.ntotal


# ── Module-level preloadable singleton ───────────────────────────────────────
_global_store: VectorStoreManager | None = None


def get_global_store() -> VectorStoreManager:
    """Return (and cache) a module-level VectorStoreManager singleton."""
    global _global_store
    if _global_store is None:
        _global_store = VectorStoreManager()
    return _global_store


def preload_global_store() -> None:
    """Eagerly load embeddings + FAISS index at server startup."""
    store = get_global_store()
    store.preload()
