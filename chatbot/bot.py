"""
chatbot/bot.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core chatbot — handle_message()

Now returns a full PipelineTrace alongside the reply,
collecting timing and debug data at every stage.
Transport-agnostic — CLI and WhatsApp both call this.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import datetime
import logging
import re
import time

import anthropic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config          import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, TEMPERATURE
from chatbot.guards  import check_emergency, check_complaint
from chatbot.prompt  import SYSTEM_PROMPT, GREETING
from chatbot.session import get as session_get, append as session_append
from chatbot.debug   import (PipelineTrace, IntentDebug, LLMDebug,
                              SQLDebug, VectorDebug, BM25Debug, RRFDebug,
                              Timer, log_to_file,
                              COST_INPUT_PER_1M, COST_OUTPUT_PER_1M)
from rag.intent      import classify
from rag.router      import route

log = logging.getLogger(__name__)
_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Turn counter per session
_turn_counts: dict[str, int] = {}

_CAPABILITY_PATTERNS = [
    r"what can you (help|do|assist)",
    r"how can you help",
    r"what (are your|do you offer|services)",
    r"what do you (know|cover|handle)",
    r"(tell me|show me) what you can",
    r"your (services|capabilities|features)",
    r"how does this (work|bot|assistant)",
    r"what (is|are) you",
    r"who are you",
]

_OUT_OF_SCOPE_RESPONSE = (
    "I'm here to help with questions about HealthHub clinics — "
    "our doctors, services, locations, appointments, and health packages.\n\n"
    "How can I help you today?"
)


def _is_capability_question(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _CAPABILITY_PATTERNS)


def handle_message(
    user_message: str,
    session_id:   str  = "default",
    debug:        bool = False,
) -> dict:
    """
    Process one patient message end-to-end.

    Returns:
        reply   — str: the assistant's response
        intent  — str: classified intent
        trace   — PipelineTrace: full pipeline trace (always populated)
                  Use debug=True to have CLI render it
    """
    total_timer = Timer()
    user_message = user_message.strip()

    # Turn counter
    _turn_counts[session_id] = _turn_counts.get(session_id, 0) + 1
    turn = _turn_counts[session_id]

    trace = PipelineTrace(
        turn=turn,
        session_id=session_id,
        query=user_message,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )

    if not user_message:
        reply = "I didn't catch that — could you type your question?"
        trace.total_ms      = total_timer.stop()
        trace.route_taken   = "empty"
        trace.reply_preview = reply
        return {"reply": reply, "intent": "empty", "trace": trace}

    # ── Guard: emergency (0 API calls) ───────────────────────
    reply = check_emergency(user_message)
    if reply:
        _save(session_id, user_message, reply)
        trace.intent.intent      = "emergency"
        trace.intent.confidence  = 1.0
        trace.intent.was_fast_path = True
        trace.route_taken        = "Rule — emergency guard"
        trace.reply_preview      = reply[:120]
        trace.total_ms           = total_timer.stop()
        log_to_file(trace)
        return {"reply": reply, "intent": "emergency", "trace": trace}

    # ── Guard: complaint (0 API calls) ───────────────────────
    reply = check_complaint(user_message)
    if reply:
        _save(session_id, user_message, reply)
        trace.intent.intent      = "complaint"
        trace.intent.confidence  = 1.0
        trace.intent.was_fast_path = True
        trace.route_taken        = "Rule — complaint guard"
        trace.reply_preview      = reply[:120]
        trace.total_ms           = total_timer.stop()
        log_to_file(trace)
        return {"reply": reply, "intent": "complaint", "trace": trace}

    # ── Capability question (0 API calls) ────────────────────
    if _is_capability_question(user_message):
        reply = GREETING
        _save(session_id, user_message, reply)
        trace.intent.intent      = "capability"
        trace.intent.confidence  = 1.0
        trace.intent.was_fast_path = True
        trace.route_taken        = "Rule — capability pattern"
        trace.reply_preview      = reply[:120]
        trace.total_ms           = total_timer.stop()
        log_to_file(trace)
        return {"reply": reply, "intent": "capability", "trace": trace}

    # ── Session history ───────────────────────────────────────
    history = session_get(session_id)

    # ── Intent classification ─────────────────────────────────
    intent_timer = Timer()
    intent = classify(user_message, history=history)
    intent_ms = intent_timer.stop()

    trace.intent = IntentDebug(
        intent=intent.intent,
        confidence=intent.confidence,
        branch=intent.branch,
        speciality=intent.speciality,
        doctor_name=intent.doctor_name,
        insurance=intent.insurance,
        day=intent.day,
        was_fast_path=False,   # got here = used Claude API
        latency_ms=intent_ms,
    )

    log.info("Intent: %s (%.2f) branch=%s spec=%s doctor=%s ins=%s",
             intent.intent, intent.confidence,
             intent.branch, intent.speciality,
             intent.doctor_name, intent.insurance)

    # ── Pure greeting — no history, no LLM ───────────────────
    if intent.intent == "out_of_scope" and not history and intent.confidence >= 0.8:
        reply = GREETING
        _save(session_id, user_message, reply)
        trace.intent.was_fast_path = True
        trace.route_taken          = "Rule — greeting (no history)"
        trace.reply_preview        = reply[:120]
        trace.total_ms             = total_timer.stop()
        log_to_file(trace)
        return {"reply": reply, "intent": "out_of_scope", "trace": trace}

    # ── out_of_scope WITH history → LLM redirect ─────────────
    if intent.intent == "out_of_scope" and history:
        system   = SYSTEM_PROMPT.format(
            context="No specific clinic information needed. "
                    "Gently redirect the patient to ask about HealthHub."
        )
        messages = history + [{"role": "user", "content": user_message}]
        llm_timer = Timer()
        response  = _client.messages.create(
            model=CLAUDE_MODEL, max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE, system=system, messages=messages,
        )
        llm_ms = llm_timer.stop()
        reply  = response.content[0].text.strip()
        _save(session_id, user_message, reply)

        usage = response.usage
        cost  = ((usage.input_tokens  / 1_000_000) * COST_INPUT_PER_1M +
                 (usage.output_tokens / 1_000_000) * COST_OUTPUT_PER_1M)
        trace.llm          = LLMDebug(model=CLAUDE_MODEL,
                                      input_tokens=usage.input_tokens,
                                      output_tokens=usage.output_tokens,
                                      latency_ms=llm_ms, cost_usd=cost,
                                      context_chars=0)
        trace.route_taken  = "Rule — out_of_scope redirect (LLM)"
        trace.reply_preview= reply[:120]
        trace.total_ms     = total_timer.stop()
        log_to_file(trace)
        return {"reply": reply, "intent": "out_of_scope", "trace": trace}

    # ── Route → retrieve context ──────────────────────────────
    route_result = route(intent)

    trace.route_taken  = route_result.route_taken
    trace.sql_debug    = route_result.sql_debug
    trace.vec_debug    = route_result.vec_debug
    trace.bm25_debug   = route_result.bm25_debug
    trace.rrf_debug    = route_result.rrf_debug
    trace.context_length = len(route_result.context)

    # ── LLM response ──────────────────────────────────────────
    context  = route_result.context
    system   = SYSTEM_PROMPT.format(
        context=context or "No specific information found for this query."
    )
    messages = history + [{"role": "user", "content": user_message}]

    llm_timer = Timer()
    response  = _client.messages.create(
        model=CLAUDE_MODEL, max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE, system=system, messages=messages,
    )
    llm_ms = llm_timer.stop()
    reply  = response.content[0].text.strip()
    _save(session_id, user_message, reply)

    usage = response.usage
    cost  = ((usage.input_tokens  / 1_000_000) * COST_INPUT_PER_1M +
             (usage.output_tokens / 1_000_000) * COST_OUTPUT_PER_1M)

    trace.llm = LLMDebug(
        model=CLAUDE_MODEL,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=llm_ms,
        cost_usd=cost,
        context_words=len(context.split()),
        context_chars=len(context),
    )
    trace.reply_preview = reply[:120]
    trace.total_ms      = total_timer.stop()

    log_to_file(trace)

    return {
        "reply":  reply,
        "intent": intent.intent,
        "trace":  trace,
    }


def _save(session_id: str, user_msg: str, reply: str) -> None:
    session_append(session_id, "user",      user_msg)
    session_append(session_id, "assistant", reply)
