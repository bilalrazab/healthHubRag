"""
interfaces/explore_api.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ChromaDB Explorer API — mounted at /explore/*

GET  /explore/stats       — collection overview
GET  /explore/chunks      — paginated chunk list
POST /explore/search      — semantic search with scores
GET  /explore/embeddings  — 2D PCA projection for scatter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY",     "false")

from config import CHROMA_DIR, CHROMA_COLLECTION

log    = logging.getLogger("explore_api")
router = APIRouter(prefix="/explore", tags=["explore"])

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=DefaultEmbeddingFunction(),
        )
        log.info("Explorer: %d chunks loaded", _collection.count())
    return _collection


# ── Stats ─────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats():
    col   = _get_collection()
    total = col.count()
    result    = col.get(include=["metadatas"], limit=total)
    metadatas = result["metadatas"] or []

    from collections import Counter
    source_types = Counter(m.get("source_type", "unknown") for m in metadatas)
    branches     = Counter(
        m.get("branch", "") for m in metadatas
        if m.get("branch") not in ("all", "none", "", None)
    )
    specialities = Counter(
        m.get("speciality", "") for m in metadatas
        if m.get("speciality") not in ("all", "none", "", None)
    )
    return {
        "total_chunks":     total,
        "source_types":     dict(source_types.most_common()),
        "top_branches":     dict(branches.most_common(15)),
        "top_specialities": dict(specialities.most_common(15)),
        "collection":       CHROMA_COLLECTION,
        "chroma_dir":       str(CHROMA_DIR),
    }


# ── Chunks ────────────────────────────────────────────────────

@router.get("/chunks")
async def list_chunks(
    source_type: Optional[str] = Query(None),
    branch:      Optional[str] = Query(None),
    limit:       int           = Query(50, le=200),
    offset:      int           = Query(0),
):
    col = _get_collection()

    conditions = []
    if source_type:
        conditions.append({"source_type": {"$eq": source_type}})
    if branch:
        conditions.append({"branch": {"$in": [branch, "all"]}})

    where = None
    if len(conditions) == 1:  where = conditions[0]
    elif len(conditions) > 1: where = {"$and": conditions}

    kwargs = {"limit": limit, "offset": offset,
              "include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where

    result    = col.get(**kwargs)
    docs      = result.get("documents") or []
    metadatas = result.get("metadatas")  or []
    ids       = result.get("ids")        or []

    chunks = []
    for doc, meta, cid in zip(docs, metadatas, ids):
        chunks.append({
            "id":           cid,
            "text":         doc,
            "text_preview": doc[:180],
            "source_type":  meta.get("source_type", ""),
            "branch":       meta.get("branch", ""),
            "speciality":   meta.get("speciality", ""),
            "doctor_name":  meta.get("doctor_name", ""),
            "title":        meta.get("title", ""),
            "url":          meta.get("url", ""),
            "word_count":   len(doc.split()),
        })

    return {"chunks": chunks, "count": len(chunks), "offset": offset}


# ── Search ────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:        str
    source_types: Optional[list[str]] = None
    branch:       Optional[str]       = None
    top_k:        int                 = 10


@router.post("/search")
async def semantic_search(req: SearchRequest):
    col = _get_collection()

    conditions = []
    if req.source_types:
        if len(req.source_types) == 1:
            conditions.append({"source_type": {"$eq": req.source_types[0]}})
        else:
            conditions.append({"source_type": {"$in": req.source_types}})
    if req.branch:
        conditions.append({"branch": {"$in": [req.branch, "all"]}})

    where = None
    if len(conditions) == 1:  where = conditions[0]
    elif len(conditions) > 1: where = {"$and": conditions}

    n = min(req.top_k, col.count())
    kwargs = {
        "query_texts": [req.query],
        "n_results":    n,
        "include":     ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        result = col.query(**kwargs)
    except Exception:
        kwargs.pop("where", None)
        result = col.query(**kwargs)

    docs      = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas",  [[]])[0]
    distances = result.get("distances",  [[]])[0]
    ids       = result.get("ids",        [[]])[0]

    results = []
    for rank, (doc, meta, dist, cid) in enumerate(
        zip(docs, metadatas, distances, ids), 1
    ):
        results.append({
            "rank":         rank,
            "id":           cid,
            "score":        round(1 - dist, 4),
            "distance":     round(dist, 4),
            "text":         doc,
            "text_preview": doc[:200],
            "source_type":  meta.get("source_type", ""),
            "branch":       meta.get("branch", ""),
            "speciality":   meta.get("speciality", ""),
            "doctor_name":  meta.get("doctor_name", ""),
            "title":        meta.get("title", ""),
            "url":          meta.get("url", ""),
            "word_count":   len(doc.split()),
        })

    return {"query": req.query, "results": results, "count": len(results)}


# ── Embeddings (PCA) ──────────────────────────────────────────

@router.get("/embeddings")
async def get_embeddings(
    source_type: Optional[str] = Query(None),
    limit:       int           = Query(500, le=2000),
):
    """2D PCA projection of chunk embeddings for scatter plot."""
    col   = _get_collection()
    total = col.count()

    where  = {"source_type": {"$eq": source_type}} if source_type else None
    kwargs = {
        "limit":   min(limit, total),
        "include": ["embeddings", "metadatas", "documents"],
    }
    if where:
        kwargs["where"] = where

    result     = col.get(**kwargs)
    embeddings = result.get("embeddings") or []
    metadatas  = result.get("metadatas")  or []
    docs       = result.get("documents")  or []
    ids        = result.get("ids")        or []

    if not embeddings:
        return {"points": [], "explained_variance": 0, "total_points": 0}

    try:
        import numpy as np
        from sklearn.decomposition import PCA

        X   = np.array(embeddings)
        pca = PCA(n_components=2, random_state=42)
        X2d = pca.fit_transform(X)
        explained = float(pca.explained_variance_ratio_.sum())

        points = []
        for coords, meta, doc, cid in zip(X2d, metadatas, docs, ids):
            points.append({
                "id":           cid,
                "x":            float(coords[0]),
                "y":            float(coords[1]),
                "source_type":  meta.get("source_type", "unknown"),
                "branch":       meta.get("branch", ""),
                "speciality":   meta.get("speciality", ""),
                "doctor_name":  meta.get("doctor_name", ""),
                "title":        meta.get("title", ""),
                "url":          meta.get("url", ""),
                "text_preview": doc[:160] if doc else "",
            })

        return {
            "points":             points,
            "explained_variance": round(explained, 3),
            "total_points":       len(points),
            "dimensions_reduced": int(X.shape[1]) if X.ndim > 1 else 0,
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="scikit-learn not installed. Run: pip install scikit-learn numpy"
        )
