"""
FAISS Vector Store — Healthcare domain RAG retrieval.

Uses HuggingFaceEmbeddings with all-MiniLM-L6-v2 for free, local embedding.
Persists the FAISS index to disk at ./faiss_index/.

On first run, ingests 100+ curated medical knowledge chunks covering
cardiology, respiratory, endocrinology, neurology, emergency medicine,
oncology, pediatrics, pharmacology, lab values, and clinical reasoning.

Supports eager preloading via preload_global_store() for zero-latency first queries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from memory.medical_corpus import MEDICAL_CORPUS

# Default index persistence path
_DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")

# Use healthcare corpus
DEFAULT_CORPUS = MEDICAL_CORPUS




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
