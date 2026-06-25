"""
interfaces/api.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI backend for HealthHub RAG.

Endpoints:
  POST /chat          — send a message, get reply + trace
  GET  /session/clear — clear a session
  GET  /health        — health check for Render
  GET  /              — serves the web UI (static/index.html)

CORS enabled for demo/testing.
Session ID comes from the client (cookie or header).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.bot     import handle_message
from chatbot.session import clear as session_clear

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("api")

app = FastAPI(
    title="HealthHub RAG API",
    description="Multi-branch healthcare RAG system for HealthHub by Al-Futtaim",
    version="1.0.0",
)

# CORS — open for demo; tighten to your domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the web UI)
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / Response models ──────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str = "web-default"
    debug:      bool = False


class DebugInfo(BaseModel):
    intent:      str
    confidence:  float
    branch:      str | None
    speciality:  str | None
    doctor_name: str | None
    insurance:   str | None
    route_taken: str
    sql_rows:    int
    vec_chunks:  int
    bm25_chunks: int
    rrf_chunks:  int
    context_chars: int
    tokens_in:   int
    tokens_out:  int
    cost_usd:    float
    total_ms:    float
    was_fast_path: bool


class ChatResponse(BaseModel):
    reply:   str
    intent:  str
    debug:   DebugInfo | None = None


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web chat UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>HealthHub RAG API</h1><p>UI not found.</p>")


@app.get("/health")
async def health():
    """Render health check — must return 200."""
    return {"status": "ok", "service": "healthhub-rag"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Calls handle_message() and formats the PipelineTrace
    into a flat DebugInfo object for the frontend.
    """
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
    """Clear a session's conversation history."""
    session_clear(session_id)
    log.info("Session cleared: %s", session_id[:8])
    return {"status": "cleared", "session_id": session_id}
