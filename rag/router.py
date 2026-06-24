"""
rag/router.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Routes each intent to the correct retrieval strategy.
Now returns a RouteResult with full trace data for debug.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.intent          import Intent
from rag.sql_retriever   import (get_branch_info, get_doctors,
                                  get_doctor_schedule, get_insurance,
                                  get_packages, get_speciality_branches)
from rag.vec_retriever   import search as _vec_search
from rag.bm25_retriever  import search as _bm25_search
from rag.rrf             import fuse as _rrf_fuse
from rag.context_builder import build
from chatbot.debug       import (SQLDebug, VectorDebug, BM25Debug,
                                  RRFDebug, Timer)

log = logging.getLogger(__name__)


@dataclass
class RouteResult:
    context:      str        = ""
    route_taken:  str        = ""
    sql_debug:    SQLDebug   = field(default_factory=SQLDebug)
    vec_debug:    VectorDebug= field(default_factory=VectorDebug)
    bm25_debug:   BM25Debug  = field(default_factory=BM25Debug)
    rrf_debug:    RRFDebug   = field(default_factory=RRFDebug)


# ── Instrumented wrappers ─────────────────────────────────────

def _sql(fn, *args, tables: list[str] = None, **kwargs) -> tuple[list, SQLDebug]:
    t = Timer()
    rows = fn(*args, **kwargs)
    ms = t.stop()
    d = SQLDebug(
        queries_run=[f"{fn.__name__}({args}, {kwargs})"],
        total_rows=len(rows),
        latency_ms=ms,
        tables_hit=tables or [fn.__name__.replace("get_", "")],
    )
    return rows, d


def _vec(query: str, source_types=None, branch=None,
         speciality=None, top_k=5) -> tuple[list, VectorDebug]:
    t = Timer()
    chunks = _vec_search(query, source_types=source_types,
                         branch=branch, speciality=speciality, top_k=top_k)
    ms = t.stop()
    filters = {}
    if source_types: filters["source_type"] = source_types
    if branch:       filters["branch"] = branch
    if speciality:   filters["speciality"] = speciality
    d = VectorDebug(
        query_used=query[:80],
        filters_applied=filters,
        chunks_returned=len(chunks),
        top_score=chunks[0]["score"] if chunks else 0.0,
        bottom_score=chunks[-1]["score"] if chunks else 0.0,
        top_titles=[f"{c.get('title','?')} [{c.get('source_type','?')}]"
                    for c in chunks[:4]],
        latency_ms=ms,
    )
    return chunks, d


def _bm25(query: str, top_k=5) -> tuple[list, BM25Debug]:
    t = Timer()
    chunks = _bm25_search(query, top_k=top_k)
    ms = t.stop()
    d = BM25Debug(
        chunks_returned=len(chunks),
        top_score=chunks[0]["score"] if chunks else 0.0,
        top_titles=[f"{c.get('title','?')} [{c.get('source_type','?')}]"
                    for c in chunks[:4]],
        latency_ms=ms,
    )
    return chunks, d


def _rrf(vec_chunks: list, bm25_chunks: list,
         top_k: int = 5) -> tuple[list, RRFDebug]:
    fused = _rrf_fuse(vec_chunks, bm25_chunks, top_k=top_k)
    d = RRFDebug(
        input_vec_count=len(vec_chunks),
        input_bm25_count=len(bm25_chunks),
        output_count=len(fused),
        top_rrf_score=fused[0]["rrf_score"] if fused else 0.0,
        bottom_rrf_score=fused[-1]["rrf_score"] if fused else 0.0,
        top_chunks=[{"title": c.get("title","?"),
                     "source_type": c.get("source_type","?"),
                     "rrf_score": c.get("rrf_score", 0)}
                    for c in fused[:4]],
    )
    return fused, d


def _merge_sql_debug(a: SQLDebug, b: SQLDebug) -> SQLDebug:
    return SQLDebug(
        queries_run=a.queries_run + b.queries_run,
        total_rows=a.total_rows + b.total_rows,
        latency_ms=a.latency_ms + b.latency_ms,
        tables_hit=list(set(a.tables_hit + b.tables_hit)),
    )


# ── Router ────────────────────────────────────────────────────

def route(intent: Intent) -> RouteResult:
    i = intent
    res = RouteResult()
    no_sql  = SQLDebug()
    no_vec  = VectorDebug()
    no_bm25 = BM25Debug()
    no_rrf  = RRFDebug()

    log.info("Routing: %s | branch=%s spec=%s doctor=%s ins=%s",
             i.intent, i.branch, i.speciality, i.doctor_name, i.insurance)

    # ── BRANCH INFO ───────────────────────────────────────────
    if i.intent == "branch_info":
        rows, sd = _sql(get_branch_info, branch_hint=i.branch,
                        tables=["branches"])
        vec, vd  = _vec(i.original_query,
                         source_types=["branch_page"], branch=i.branch, top_k=3)
        res.context     = build(branch_rows=rows, vector_chunks=vec or None)
        res.route_taken = "SQL(branches) + Vector(branch_page)"
        res.sql_debug   = sd
        res.vec_debug   = vd
        return res

    # ── BRANCH HOURS ──────────────────────────────────────────
    if i.intent == "branch_hours":
        rows, sd = _sql(get_branch_info, branch_hint=i.branch,
                        tables=["branches"])
        res.context     = build(branch_rows=rows)
        res.route_taken = "SQL(branches)"
        res.sql_debug   = sd
        return res

    # ── DOCTOR SEARCH ─────────────────────────────────────────
    if i.intent == "doctor_search":
        dr, sd   = _sql(get_doctors, branch_hint=i.branch,
                        speciality_hint=i.speciality,
                        doctor_name_hint=i.doctor_name,
                        tables=["doctors", "doctor_branches"])
        vec, vd  = _vec(i.original_query, source_types=["doctor_profile"],
                         branch=i.branch, speciality=i.speciality, top_k=5)
        bm, bd   = _bm25(i.original_query, top_k=5)
        bm_doc   = [r for r in bm if r.get("source_type") == "doctor_profile"]
        fused, rd = _rrf(vec, bm_doc, top_k=3)

        if dr:
            res.context = build(doctor_rows=dr[:5])
            res.route_taken = "SQL(doctors) — primary"
        else:
            res.context = build(vector_chunks=fused)
            res.route_taken = "SQL(doctors) miss → Hybrid(RRF)"

        res.sql_debug  = sd
        res.vec_debug  = vd
        res.bm25_debug = bd
        res.rrf_debug  = rd
        return res

    # ── DOCTOR AVAILABILITY ───────────────────────────────────
    if i.intent == "doctor_availability":
        if i.doctor_name:
            sched, sd1 = _sql(get_doctor_schedule,
                              doctor_name_hint=i.doctor_name,
                              branch_hint=i.branch, day_hint=i.day,
                              tables=["doctor_schedules"])
            dr, sd2    = _sql(get_doctors, doctor_name_hint=i.doctor_name,
                              tables=["doctors"])
            res.context     = build(doctor_rows=dr[:1], schedule_rows=sched)
            res.route_taken = "SQL(doctors + schedules)"
            res.sql_debug   = _merge_sql_debug(sd1, sd2)
        else:
            vec, vd = _vec(i.original_query, source_types=["general"], top_k=3)
            res.context     = build(vector_chunks=vec)
            res.route_taken = "Vector(general) — no doctor named"
            res.vec_debug   = vd
        return res

    # ── SPECIALITY INFO ───────────────────────────────────────
    if i.intent == "speciality_info":
        hint = i.speciality or i.original_query
        sb, sd   = _sql(get_speciality_branches, speciality_hint=hint,
                        tables=["specialities", "branch_specialities"])
        vec, vd  = _vec(i.original_query, source_types=["speciality"],
                         speciality=i.speciality, top_k=5)
        bm, bd   = _bm25(i.original_query, top_k=5)
        bm_spec  = [r for r in bm if r.get("source_type") == "speciality"]
        fused, rd = _rrf(vec, bm_spec, top_k=4)
        res.context     = build(spec_branch_rows=sb, vector_chunks=fused)
        res.route_taken = "SQL(specialities) + Hybrid(RRF)"
        res.sql_debug   = sd
        res.vec_debug   = vd
        res.bm25_debug  = bd
        res.rrf_debug   = rd
        return res

    # ── INSURANCE CHECK ───────────────────────────────────────
    if i.intent == "insurance_check":
        ins, sd  = _sql(get_insurance, insurance_hint=i.insurance,
                        branch_hint=i.branch,
                        tables=["insurance_providers", "branch_insurance"])
        vec, vd  = VectorDebug(), VectorDebug()
        if not ins:
            vec_chunks, vd = _vec(i.original_query,
                                   source_types=["general", "branch_page"], top_k=3)
        else:
            vec_chunks = []
        res.context     = build(insurance_rows=ins,
                                vector_chunks=vec_chunks or None)
        res.route_taken = "SQL(insurance)" + (" + Vector fallback" if not ins else "")
        res.sql_debug   = sd
        res.vec_debug   = vd
        return res

    # ── APPOINTMENT GUIDE ─────────────────────────────────────
    if i.intent == "appointment_guide":
        vec, vd  = _vec(i.original_query, source_types=["general"], top_k=5)
        bm, bd   = _bm25(i.original_query, top_k=5)
        fused, rd = _rrf(vec, bm, top_k=4)
        res.context     = build(vector_chunks=fused)
        res.route_taken = "Hybrid(RRF: Vector + BM25)"
        res.vec_debug   = vd
        res.bm25_debug  = bd
        res.rrf_debug   = rd
        return res

    # ── PACKAGE INFO ──────────────────────────────────────────
    if i.intent == "package_info":
        hint = i.speciality or _pkg_hint(i.original_query)
        pkgs, sd = _sql(get_packages, name_hint=hint,
                        tables=["health_packages"])
        vec, vd  = _vec(i.original_query, source_types=["package"], top_k=4)
        bm, bd   = _bm25(i.original_query, top_k=4)
        bm_pkg   = [r for r in bm if r.get("source_type") == "package"]
        fused, rd = _rrf(vec, bm_pkg, top_k=3)
        res.context     = build(package_rows=pkgs,
                                vector_chunks=fused or None)
        res.route_taken = "SQL(packages) + Hybrid(RRF)"
        res.sql_debug   = sd
        res.vec_debug   = vd
        res.bm25_debug  = bd
        res.rrf_debug   = rd
        return res

    # ── TELEHEALTH ────────────────────────────────────────────
    if i.intent == "telehealth":
        vec, vd  = _vec(i.original_query, source_types=["general"], top_k=5)
        bm, bd   = _bm25(i.original_query, top_k=5)
        fused, rd = _rrf(vec, bm, top_k=4)
        res.context     = build(vector_chunks=fused)
        res.route_taken = "Hybrid(RRF: Vector + BM25)"
        res.vec_debug   = vd
        res.bm25_debug  = bd
        res.rrf_debug   = rd
        return res

    # ── GENERAL HEALTH ────────────────────────────────────────
    if i.intent == "general_health":
        vec, vd  = _vec(i.original_query,
                         source_types=["speciality", "general"],
                         speciality=i.speciality, top_k=6)
        bm, bd   = _bm25(i.original_query, top_k=6)
        fused, rd = _rrf(vec, bm, top_k=5)
        res.context     = build(vector_chunks=fused)
        res.route_taken = "Hybrid(RRF: Vector + BM25)"
        res.vec_debug   = vd
        res.bm25_debug  = bd
        res.rrf_debug   = rd
        return res

    # ── COMPLAINT / OUT_OF_SCOPE ──────────────────────────────
    if i.intent in ("complaint", "out_of_scope"):
        res.context     = ""
        res.route_taken = "Rule — no retrieval"
        return res

    # ── Fallback ──────────────────────────────────────────────
    vec, vd  = _vec(i.original_query, top_k=5)
    bm, bd   = _bm25(i.original_query, top_k=5)
    fused, rd = _rrf(vec, bm, top_k=4)
    res.context     = build(vector_chunks=fused)
    res.route_taken = "Fallback — Hybrid(RRF)"
    res.vec_debug   = vd
    res.bm25_debug  = bd
    res.rrf_debug   = rd
    return res


def _pkg_hint(query: str) -> str:
    keywords = ["dental", "flu", "vaccine", "iv drip", "vitamin", "wellness",
                "checkup", "check-up", "woman", "women", "diabetes",
                "heart", "cancer", "weight", "hair"]
    q = query.lower()
    for kw in keywords:
        if kw in q:
            return kw
    return query
