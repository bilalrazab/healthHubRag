"""
rag/rrf.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reciprocal Rank Fusion

Combines two ranked result lists (semantic + keyword)
into one unified ranking without needing to normalise
or compare raw scores across different systems.

Formula:  RRF(d) = Σ  1 / (k + rank(d))
          k = 60  (standard constant, softens rank gaps)

Industry standard approach used in production search
systems at Microsoft, Elasticsearch, and Cohere.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K_FINAL, RRF_K


def fuse(semantic: list[dict],
         keyword:  list[dict],
         top_k:    int = TOP_K_FINAL) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Both lists contain dicts with at least a 'text' key.
    Uses text[:120] as the deduplication key.

    Returns the top_k fused results, each with a 'rrf_score'.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, chunk in enumerate(semantic):
        key = chunk["text"][:120]
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_map[key]   = chunk

    for rank, chunk in enumerate(keyword):
        key = chunk["text"][:120]
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_map[key]   = chunk

    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    fused = []
    for key in sorted_keys[:top_k]:
        entry = dict(chunk_map[key])
        entry["rrf_score"] = round(rrf_scores[key], 6)
        fused.append(entry)

    return fused
