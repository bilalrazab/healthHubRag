"""
interfaces/api.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI backend for HealthHub RAG.

Fix: STATIC_DIR now uses Path(__file__).resolve() so the
absolute path is always correct regardless of working
directory or Docker WORKDIR setting.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys

# Resolve project root from this file's absolute location.
# Works correctly in every environment:
#   local:  D:/HealthHubRag/interfaces/api.py  → D:/HealthHubRag
#   Docker: /app/interfaces/api.py             → /app
ROOT_DIR   = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"

sys.path.insert(0, str(ROOT_DIR))

from chatbot.bot     import handle_message
from chatbot.session import clear as session_clear

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("api")

# Log paths at startup — visible in Render logs, useful for debugging
log.info("ROOT_DIR:   %s", ROOT_DIR)
log.info("STATIC_DIR: %s  (exists=%s)", STATIC_DIR, STATIC_DIR.exists())

app = FastAPI(
    title="HealthHub RAG API",
    description="Multi-branch healthcare RAG system for HealthHub by Al-Futtaim",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files — always use the absolute resolved path
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    log.info("Static files mounted OK")
else:
    log.error("STATIC_DIR not found — UI will not load. Expected: %s", STATIC_DIR)


# ── Request / Response models ──────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str  = "web-default"
    debug:      bool = False


class DebugInfo(BaseModel):
    intent:        str
    confidence:    float
    branch:        str | None
    speciality:    str | None
    doctor_name:   str | None
    insurance:     str | None
    route_taken:   str
    sql_rows:      int
    vec_chunks:    int
    bm25_chunks:   int
    rrf_chunks:    int
    context_chars: int
    tokens_in:     int
    tokens_out:    int
    cost_usd:      float
    total_ms:      float
    was_fast_path: bool


class ChatResponse(BaseModel):
    reply:  str
    intent: str
    debug:  DebugInfo | None = None


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web chat UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    # Detailed error so it's clear what path was looked up
    return HTMLResponse(
        content=(
            f"<h2>UI not found</h2>"
            f"<p>Expected: <code>{index}</code></p>"
            f"<p>STATIC_DIR exists: <code>{STATIC_DIR.exists()}</code></p>"
            f"<p>Files in ROOT_DIR: <code>{list(ROOT_DIR.iterdir())}</code></p>"
        ),
        status_code=404,
    )


@app.get("/health")
async def health():
    """Render health check."""
    return {
        "status":      "ok",
        "service":     "healthhub-rag",
        "static_dir":  str(STATIC_DIR),
        "static_ok":   STATIC_DIR.exists(),
        "index_ok":    (STATIC_DIR / "index.html").exists(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    log.info("Chat | session=%s | msg=%s", req.session_id[:8], req.message[:60])

    result = handle_message(
        user_message=req.message,
        session_id=req.session_id,
        debug=req.debug,
    )

    debug_info = None
    if req.debug:
        trace = result.get("trace")
        if trace:
            debug_info = DebugInfo(
                intent=       trace.intent.intent,
                confidence=   trace.intent.confidence,
                branch=       trace.intent.branch,
                speciality=   trace.intent.speciality,
                doctor_name=  trace.intent.doctor_name,
                insurance=    trace.intent.insurance,
                route_taken=  trace.route_taken,
                sql_rows=     trace.sql_debug.total_rows,
                vec_chunks=   trace.vec_debug.chunks_returned,
                bm25_chunks=  trace.bm25_debug.chunks_returned,
                rrf_chunks=   trace.rrf_debug.output_count,
                context_chars=trace.context_length,
                tokens_in=    trace.llm.input_tokens,
                tokens_out=   trace.llm.output_tokens,
                cost_usd=     trace.llm.cost_usd,
                total_ms=     trace.total_ms,
                was_fast_path=trace.intent.was_fast_path,
            )

    return ChatResponse(
        reply=result["reply"],
        intent=result["intent"],
        debug=debug_info,
    )


@app.get("/session/clear")
async def clear_session(session_id: str = "web-default"):
    session_clear(session_id)
    log.info("Session cleared: %s", session_id[:8])
    return {"status": "cleared", "session_id": session_id}
