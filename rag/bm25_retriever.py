"""
rag/bm25_retriever.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM25 keyword search over the ChromaDB corpus.

Fixed:
  - ChromaDB 0.5.5 collection.get() returns object with _type
    field — must use .get() with explicit limit, then access
    .documents / .metadatas / .ids as attributes, not dict keys
  - Added CHROMA_BATCH paging to avoid memory issues on large
    collections (ChromaDB has a default limit of 100 per get())
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_COLLECTION, TOP_K_KEYWORD

log = logging.getLogger(__name__)

_bm25:   Optional[BM25Okapi] = None
_corpus: list[dict]           = []


def _build_index() -> None:
    global _bm25, _corpus

    import chromadb
    from chromadb.config import Settings

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}\n"
            "Run: python -m ingestion.vec_loader"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(CHROMA_COLLECTION)
    total = collection.count()
    log.info("Building BM25 index from %d chunks...", total)

    # ── FIX: ChromaDB 0.5.5 returns a GetResult object, not a dict ──
    # Fetch in pages to avoid the default 100-item limit
    PAGE = 500
    all_docs:  list[str]  = []
    all_metas: list[dict] = []
    all_ids:   list[str]  = []

    offset = 0
    while offset < total:
        result = collection.get(
            limit=PAGE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        # ChromaDB 0.5.x: result is a GetResult with .documents, .metadatas, .ids
        # Access as dict keys (it implements __getitem__)
        docs  = result["documents"]  or []
        metas = result["metadatas"]  or []
        ids   = result["ids"]        or []

        all_docs.extend(docs)
        all_metas.extend(metas)
        all_ids.extend(ids)
        offset += PAGE

        if len(docs) < PAGE:
            break   # last page

    _corpus = []
    tokenized = []

    for doc, meta, doc_id in zip(all_docs, all_metas, all_ids):
        if not doc:
            continue
        _corpus.append({
            "text":        doc,
            "id":          doc_id,
            "source_type": meta.get("source_type", ""),
            "branch":      meta.get("branch", ""),
            "speciality":  meta.get("speciality", ""),
            "doctor_name": meta.get("doctor_name", ""),
            "url":         meta.get("url", ""),
            "title":       meta.get("title", ""),
        })
        tokenized.append(doc.lower().split())

    _bm25 = BM25Okapi(tokenized)
    log.info("BM25 index ready: %d documents", len(_corpus))


def search(query: str, top_k: int = TOP_K_KEYWORD) -> list[dict]:
    global _bm25, _corpus

    if _bm25 is None:
        _build_index()

    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)

    ranked = sorted(
        zip(scores, _corpus),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    results = []
    for score, entry in ranked:
        if score > 0:
            results.append({
                "text":        entry["text"],
                "source_type": entry["source_type"],
                "branch":      entry["branch"],
                "speciality":  entry["speciality"],
                "doctor_name": entry["doctor_name"],
                "url":         entry["url"],
                "title":       entry["title"],
                "score":       round(float(score), 4),
            })

    return results
