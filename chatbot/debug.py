"""
chatbot/debug.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-grade debug panel for the HealthHub RAG system.

Collects timing, retrieval details, token usage, and cost
estimates at every stage of the pipeline.

Two outputs:
  1. Rich terminal panel (shown inline in CLI when /debug ON)
  2. JSONL evaluation log (data/eval/eval_log.jsonl)
     One line per query — importable into any analysis tool.

Evaluation metrics tracked:
  - Intent classification: class, confidence, entities
  - SQL retrieval: tables hit, row counts, query time
  - Vector retrieval: chunks returned, top scores, filter used
  - BM25 retrieval: chunks returned, top BM25 score
  - RRF fusion: final chunk count, score range
  - LLM response: model, tokens in/out, latency, cost estimate
  - Overall: end-to-end latency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Eval log path
EVAL_DIR = Path(__file__).parent.parent / "data" / "eval"
EVAL_LOG  = EVAL_DIR / "eval_log.jsonl"

# Cost per million tokens (claude-sonnet-4-6 pricing)
COST_INPUT_PER_1M  = 3.00   # USD
COST_OUTPUT_PER_1M = 15.00  # USD


# ── Stage timing context manager ──────────────────────────────

class Timer:
    def __init__(self):
        self._start = time.perf_counter()
        self.elapsed_ms: float = 0.0

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return self.elapsed_ms


# ── Data classes for each pipeline stage ──────────────────────

@dataclass
class IntentDebug:
    intent:       str   = ""
    confidence:   float = 0.0
    branch:       Optional[str] = None
    speciality:   Optional[str] = None
    doctor_name:  Optional[str] = None
    insurance:    Optional[str] = None
    day:          Optional[str] = None
    was_fast_path: bool  = False   # True = no API call made
    latency_ms:   float = 0.0


@dataclass
class SQLDebug:
    queries_run:    list[str]  = field(default_factory=list)
    total_rows:     int        = 0
    latency_ms:     float      = 0.0
    tables_hit:     list[str]  = field(default_factory=list)


@dataclass
class VectorDebug:
    query_used:      str        = ""
    filters_applied: dict       = field(default_factory=dict)
    chunks_returned: int        = 0
    top_score:       float      = 0.0
    bottom_score:    float      = 0.0
    top_titles:      list[str]  = field(default_factory=list)
    latency_ms:      float      = 0.0


@dataclass
class BM25Debug:
    chunks_returned: int       = 0
    top_score:       float     = 0.0
    top_titles:      list[str] = field(default_factory=list)
    latency_ms:      float     = 0.0


@dataclass
class RRFDebug:
    input_vec_count:  int        = 0
    input_bm25_count: int        = 0
    output_count:     int        = 0
    top_rrf_score:    float      = 0.0
    bottom_rrf_score: float      = 0.0
    top_chunks:       list[dict] = field(default_factory=list)  # [{title, source, score}]


@dataclass
class LLMDebug:
    model:          str   = ""
    input_tokens:   int   = 0
    output_tokens:  int   = 0
    latency_ms:     float = 0.0
    cost_usd:       float = 0.0
    context_words:  int   = 0
    context_chars:  int   = 0


@dataclass
class PipelineTrace:
    """Complete trace of one handle_message() call."""
    turn:           int    = 0
    session_id:     str    = ""
    query:          str    = ""
    timestamp:      str    = ""
    total_ms:       float  = 0.0

    intent:  IntentDebug = field(default_factory=IntentDebug)
    sql:     SQLDebug    = field(default_factory=SQLDebug)
    vector:  VectorDebug = field(default_factory=VectorDebug)
    bm25:    BM25Debug   = field(default_factory=BM25Debug)
    rrf:     RRFDebug    = field(default_factory=RRFDebug)
    llm:     LLMDebug    = field(default_factory=LLMDebug)

    route_taken:    str   = ""   # e.g. "SQL+Vector", "SQL_only", "Rule"
    context_length: int   = 0    # chars of context sent to LLM
    reply_preview:  str   = ""   # first 120 chars of reply


# ── Terminal renderer ──────────────────────────────────────────

# ANSI colours
_R  = "\033[0m"
_B  = "\033[1m"
_DIM= "\033[2m"
_T  = "\033[96m"    # teal
_G  = "\033[90m"    # grey
_Y  = "\033[93m"    # yellow
_GR = "\033[92m"    # green
_RE = "\033[91m"    # red
_BL = "\033[94m"    # blue
_MA = "\033[95m"    # magenta


def _bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    """ASCII confidence/score bar."""
    filled = int((value / max(max_val, 0.0001)) * width)
    filled = min(filled, width)
    empty  = width - filled
    color  = _GR if value >= 0.75 else _Y if value >= 0.5 else _RE
    return f"{color}{'█' * filled}{'░' * empty}{_R}"


def _ms(ms: float) -> str:
    """Format milliseconds readably."""
    if ms < 1:
        return f"{ms:.2f}ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.2f}s"


def _cost(usd: float) -> str:
    if usd < 0.001:
        return f"<$0.001"
    return f"${usd:.4f}"


def render_terminal(trace: PipelineTrace) -> str:
    """Render a full debug panel for terminal output."""
    lines = []
    W = 62  # panel width

    def divider(char="─", label=""):
        if label:
            pad = (W - len(label) - 4) // 2
            lines.append(f"{_G}  {'─'*pad} {_T}{_B}{label}{_R}{_G} {'─'*pad}{_R}")
        else:
            lines.append(f"{_G}  {'─'*W}{_R}")

    def row(label, value, color=_R):
        lines.append(f"{_G}  {_DIM}{label:<22}{_R}{color}{value}{_R}")

    def blank():
        lines.append("")

    # ── Header ────────────────────────────────────────────────
    blank()
    lines.append(f"{_G}  {'═'*W}{_R}")
    lines.append(f"{_G}  {_B}{_T}  RAG PIPELINE TRACE  "
                 f"{_G}Turn #{trace.turn}   Total: {_ms(trace.total_ms)}{_R}")
    lines.append(f"{_G}  {'═'*W}{_R}")

    # ── 1. Intent ─────────────────────────────────────────────
    divider(label="① INTENT CLASSIFICATION")
    i = trace.intent
    conf_bar = _bar(i.confidence)
    fast_tag = f"  {_G}[fast-path, $0]{_R}" if i.was_fast_path else \
               f"  {_G}[Claude API, {_ms(i.latency_ms)}]{_R}"
    row("Intent:",      f"{_B}{_T}{i.intent}{_R}{fast_tag}")
    row("Confidence:",  f"{conf_bar}  {_Y}{i.confidence:.0%}{_R}")

    entities_found = {k: v for k, v in {
        "branch": i.branch, "speciality": i.speciality,
        "doctor": i.doctor_name, "insurance": i.insurance,
        "day": i.day,
    }.items() if v}

    if entities_found:
        for k, v in entities_found.items():
            row(f"  entity.{k}:", f"{_BL}{v}{_R}")
    else:
        row("Entities:", f"{_DIM}none extracted{_R}")

    row("Route taken:",  f"{_MA}{trace.route_taken}{_R}")

    # ── 2. SQL Retrieval ──────────────────────────────────────
    divider(label="② SQL RETRIEVAL")
    s = trace.sql
    if s.queries_run:
        row("Tables hit:",    f"{_GR}{', '.join(s.tables_hit) or 'none'}{_R}")
        row("Rows returned:", f"{_B}{s.total_rows}{_R}")
        row("Latency:",       f"{_ms(s.latency_ms)}")
        for q in s.queries_run[:3]:
            short = q.strip().replace("\n", " ")[:55]
            lines.append(f"{_G}  {_DIM}  SQL: {short}...{_R}")
    else:
        row("Status:", f"{_DIM}skipped (not needed for this intent){_R}")

    # ── 3. Vector Retrieval ───────────────────────────────────
    divider(label="③ VECTOR RETRIEVAL (ChromaDB)")
    v = trace.vector
    if v.chunks_returned > 0:
        filter_str = json.dumps(v.filters_applied) if v.filters_applied else "none"
        row("Filter applied:",  f"{_BL}{filter_str[:50]}{_R}")
        row("Chunks returned:", f"{_B}{v.chunks_returned}{_R}")
        row("Score range:",     f"{_GR}{v.top_score:.4f}{_R}  →  {_RE}{v.bottom_score:.4f}{_R}")
        row("Latency:",         f"{_ms(v.latency_ms)}")
        if v.top_titles:
            lines.append(f"{_G}  {_DIM}  Top results:{_R}")
            for t in v.top_titles[:3]:
                lines.append(f"{_G}  {_DIM}    • {t[:54]}{_R}")
    else:
        row("Status:", f"{_DIM}skipped or returned 0 results{_R}")

    # ── 4. BM25 Retrieval ─────────────────────────────────────
    divider(label="④ BM25 KEYWORD RETRIEVAL")
    b = trace.bm25
    if b.chunks_returned > 0:
        row("Chunks returned:", f"{_B}{b.chunks_returned}{_R}")
        row("Top BM25 score:",  f"{_GR}{b.top_score:.4f}{_R}")
        row("Latency:",         f"{_ms(b.latency_ms)}")
        if b.top_titles:
            lines.append(f"{_G}  {_DIM}  Top results:{_R}")
            for t in b.top_titles[:3]:
                lines.append(f"{_G}  {_DIM}    • {t[:54]}{_R}")
    else:
        row("Status:", f"{_DIM}skipped or returned 0 results{_R}")

    # ── 5. RRF Fusion ─────────────────────────────────────────
    divider(label="⑤ RRF FUSION")
    r = trace.rrf
    if r.output_count > 0:
        row("Input (vec + bm25):", f"{r.input_vec_count} + {r.input_bm25_count} chunks")
        row("Output (fused):",     f"{_B}{r.output_count} chunks{_R}")
        row("RRF score range:",    f"{_GR}{r.top_rrf_score:.6f}{_R}  →  {r.bottom_rrf_score:.6f}")
        if r.top_chunks:
            lines.append(f"{_G}  {_DIM}  Fused ranking:{_R}")
            for idx, chunk in enumerate(r.top_chunks[:3], 1):
                title  = chunk.get("title", "—")[:40]
                src    = chunk.get("source_type", "")
                score  = chunk.get("rrf_score", 0)
                lines.append(f"{_G}  {_DIM}    [{idx}] {title} ({src})  rrf={score:.6f}{_R}")
    else:
        row("Status:", f"{_DIM}not used (SQL handled this intent){_R}")

    # ── 6. Context sent to LLM ────────────────────────────────
    divider(label="⑥ CONTEXT → LLM")
    row("Context length:", f"{_B}{trace.context_length:,}{_R} chars")
    row("Est. words:",     f"~{trace.context_length // 5:,}")

    # ── 7. LLM Response ───────────────────────────────────────
    divider(label="⑦ LLM  (claude-sonnet-4-6)")
    l = trace.llm
    if l.input_tokens > 0:
        total_tokens = l.input_tokens + l.output_tokens
        row("Model:",         f"{_T}{l.model}{_R}")
        row("Tokens in:",     f"{_Y}{l.input_tokens:,}{_R}")
        row("Tokens out:",    f"{_Y}{l.output_tokens:,}{_R}")
        row("Total tokens:",  f"{_B}{total_tokens:,}{_R}")
        row("LLM latency:",   f"{_ms(l.latency_ms)}")
        row("Est. cost:",     f"{_GR}{_cost(l.cost_usd)}{_R}")
    else:
        row("Status:", f"{_GR}$0 — hardcoded response (no LLM call){_R}")

    # ── 8. Summary ────────────────────────────────────────────
    divider(label="⑧ SUMMARY")
    api_calls = (0 if trace.intent.was_fast_path else 1) + (1 if l.input_tokens > 0 else 0)
    row("API calls made:",  f"{_B}{api_calls}{_R}")
    row("Total latency:",   f"{_B}{_ms(trace.total_ms)}{_R}")
    row("Total cost:",      f"{_GR}{_cost(l.cost_usd)}{_R}")
    row("Reply preview:",   f"{_DIM}{trace.reply_preview[:55]}...{_R}"
                            if len(trace.reply_preview) > 55
                            else f"{_DIM}{trace.reply_preview}{_R}")

    lines.append(f"{_G}  {'═'*W}{_R}")
    blank()

    return "\n".join(lines)


# ── JSONL evaluation logger ────────────────────────────────────

def log_to_file(trace: PipelineTrace) -> None:
    """Append trace as one JSON line to the eval log."""
    import datetime
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp":     datetime.datetime.utcnow().isoformat() + "Z",
        "turn":          trace.turn,
        "session_id":    trace.session_id,
        "query":         trace.query,
        "total_ms":      round(trace.total_ms, 2),
        "route_taken":   trace.route_taken,
        "intent": {
            "class":        trace.intent.intent,
            "confidence":   trace.intent.confidence,
            "branch":       trace.intent.branch,
            "speciality":   trace.intent.speciality,
            "doctor_name":  trace.intent.doctor_name,
            "insurance":    trace.intent.insurance,
            "was_fast_path":trace.intent.was_fast_path,
            "latency_ms":   round(trace.intent.latency_ms, 2),
        },
        "sql": {
            "tables_hit":   trace.sql.tables_hit,
            "rows_returned":trace.sql.total_rows,
            "latency_ms":   round(trace.sql.latency_ms, 2),
        },
        "vector": {
            "filters":      trace.vector.filters_applied,
            "chunks":       trace.vector.chunks_returned,
            "top_score":    trace.vector.top_score,
            "latency_ms":   round(trace.vector.latency_ms, 2),
        },
        "bm25": {
            "chunks":       trace.bm25.chunks_returned,
            "top_score":    trace.bm25.top_score,
            "latency_ms":   round(trace.bm25.latency_ms, 2),
        },
        "rrf": {
            "output_chunks":trace.rrf.output_count,
            "top_score":    trace.rrf.top_rrf_score,
        },
        "llm": {
            "model":         trace.llm.model,
            "tokens_in":     trace.llm.input_tokens,
            "tokens_out":    trace.llm.output_tokens,
            "latency_ms":    round(trace.llm.latency_ms, 2),
            "cost_usd":      round(trace.llm.cost_usd, 6),
        },
        "context_chars":  trace.context_length,
        "reply_preview":  trace.reply_preview[:200],
    }

    with open(EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
