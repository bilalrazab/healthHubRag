"""
rag/vec_retriever.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ChromaDB semantic search with metadata pre-filters.

Key design: filters applied BEFORE semantic ranking.
A query about JVC only searches JVC chunks, not the
entire corpus. Without this, a cardiology description
from Arabian Center could outscore JVC content on a
JVC-specific query.

Lazy-loads the model and collection on first call.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL,
    TOP_K_SEMANTIC,
)

log = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────
_model:      Optional[SentenceTransformer] = None
_collection  = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DIR}\n"
                "Run: python -m ingestion.vec_loader"
            )
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(CHROMA_COLLECTION)
        log.info("ChromaDB collection loaded: %d chunks", _collection.count())
    return _collection


# ── Filter builder ────────────────────────────────────────────

def _build_where(source_types: Optional[list[str]] = None,
                 branch: Optional[str] = None,
                 speciality: Optional[str] = None,
                 doctor_name: Optional[str] = None) -> Optional[dict]:
    """
    Build ChromaDB metadata filter.
    Only applies filters where we have actual entity values.
    Never over-filters — if no entities, returns None (search all).
    """
    conditions = []

    if source_types:
        if len(source_types) == 1:
            conditions.append({"source_type": {"$eq": source_types[0]}})
        else:
            conditions.append({"source_type": {"$in": source_types}})

    if branch:
        # Match branch name or "all" (network-wide content)
        conditions.append({
            "branch": {"$in": [branch, "all"]}
        })

    if speciality:
        conditions.append({
            "speciality": {"$in": [speciality, "all", "none"]}
        })

    if doctor_name:
        conditions.append({
            "doctor_name": {"$eq": doctor_name}
        })

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Main retrieval function ───────────────────────────────────

def search(query: str,
           source_types: Optional[list[str]] = None,
           branch: Optional[str] = None,
           speciality: Optional[str] = None,
           doctor_name: Optional[str] = None,
           top_k: int = TOP_K_SEMANTIC) -> list[dict]:
    """
    Semantic search with optional metadata pre-filters.

    Returns list of dicts:
    {
        text, source_type, branch, speciality,
        doctor_name, url, title, score
    }
    """
    model      = _get_model()
    collection = _get_collection()

    # Embed the query
    query_emb = model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    # Build metadata filter
    where = _build_where(source_types, branch, speciality, doctor_name)

    # Guard: don't request more results than exist
    n_results = min(top_k, max(1, collection.count()))

    # Query ChromaDB
    kwargs = {
        "query_embeddings": query_emb,
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        log.warning("ChromaDB query failed (relaxing filter): %s", e)
        # Relax filter and retry without branch/speciality constraint
        kwargs.pop("where", None)
        results = collection.query(**kwargs)

    # Format results
    chunks = []
    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        chunks.append({
            "text":        doc,
            "source_type": meta.get("source_type", ""),
            "branch":      meta.get("branch", ""),
            "speciality":  meta.get("speciality", ""),
            "doctor_name": meta.get("doctor_name", ""),
            "url":         meta.get("url", ""),
            "title":       meta.get("title", ""),
            "score":       round(1 - dist, 4),   # cosine similarity
        })

    return chunks
