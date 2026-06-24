"""
rag/intent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Intent Classifier + Entity Extractor

Key fix: classifier now receives recent conversation history
so it can resolve follow-up queries like:
  "I have NAS, is it covered in Qusais?" → NAS + Qusais
  "is it covered in DFC?"                → still NAS + DFC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import anthropic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY",     "false")

log = logging.getLogger(__name__)
_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_INTENT_VALUES = {
    "branch_info", "branch_hours", "doctor_search",
    "doctor_availability", "speciality_info", "insurance_check",
    "appointment_guide", "package_info", "telehealth",
    "emergency", "general_health", "complaint", "out_of_scope",
}

_GREETINGS = {
    "hi", "hello", "hey", "salam", "مرحبا", "السلام عليكم",
    "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there", "ok", "okay",
    "yes", "no", "thanks", "thank you", "شكرا",
}

_EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "not breathing",
    "difficulty breathing", "heart attack", "unconscious", "fainted",
    "collapsed", "stroke", "seizure", "severe bleeding", "choking",
    "overdose", "allergic reaction", "anaphylaxis", "emergency",
    "ألم في الصدر", "ضيق في التنفس", "إسعاف",
]

_COMPLAINT_KEYWORDS = [
    "complaint", "complain", "bad experience",
    "unhappy", "dissatisfied", "refund", "rude", "مشكلة", "شكوى",
]

# ── Branch aliases (comprehensive) ────────────────────────────
BRANCH_ALIASES = {
    # Short codes / common abbreviations
    "dfc":   "Festival City",
    "jvc":   "JVC",
    "dso":   "Silicon Oasis",
    "ic":    "International City",
    "dg":    "Discovery Gardens",
    # Area names
    "karama":           "Al Karama",
    "al karama":        "Al Karama",
    "nahda":            "Al Nahda",
    "al nahda":         "Al Nahda",
    "qusais":           "Al Qusais",
    "al qusais":        "Al Qusais",
    "warqa":            "Al Warqa",
    "al warqa":         "Al Warqa",
    "arabian center":   "Arabian Center",
    "arabian centre":   "Arabian Center",
    "barsha":           "Barsha Heights",
    "barsha heights":   "Barsha Heights",
    "tecom":            "Barsha Heights",
    "festival plaza":   "Festival Plaza",
    "jebel ali":        "Festival Plaza",
    "international city": "International City",
    "silicon oasis":    "Silicon Oasis",
    "discovery gardens":"Discovery Gardens",
    "jumeirah village circle": "JVC",
    "jumeirah village": "JVC",
    "festival city":    "Festival City",
    "day surgery":      "Festival City",
    "dubai festival city": "Festival City",
    "dfac":             "Festival City",
}


def _resolve_branch_alias(text: str) -> Optional[str]:
    """Map abbreviations and area names to canonical branch names."""
    t = text.lower().strip()
    return BRANCH_ALIASES.get(t)


_SYSTEM = """You are an intent classification engine for HealthHub by Al-Futtaim, a clinic network in Dubai.

You will receive the recent conversation history followed by the new patient query.
Use the conversation history to resolve references like "it", "there", "same place", "that clinic".

Your ONLY job: return a JSON object classifying the latest query in context.

CRITICAL RULES:
1. Return ONLY valid JSON — no text before or after, no markdown fences
2. Use conversation history to fill in implied entities (e.g. if patient mentioned NAS earlier and now asks "is it covered in DFC?", extract insurance=NAS, branch=DFC)
3. If query is a bare location name like "Al Qusais!" or "DFC?" — classify as branch_info
4. Short follow-ups like "and there?" or "what about JVC?" are continuation queries — classify based on prior intent

Intent classes:
- branch_info        : address, location, phone, parking
- branch_hours       : opening hours, closing time
- doctor_search      : find a doctor or specialist
- doctor_availability: when is a doctor available
- speciality_info    : what does a speciality treat
- insurance_check    : insurance coverage, accepted providers
- appointment_guide  : how to book, cancel, walk-in
- package_info       : health packages, prices
- telehealth         : online/video consultation
- emergency          : chest pain, can't breathe, stroke
- general_health     : symptoms, disease info
- complaint          : bad experience, feedback
- out_of_scope       : genuine greeting with no question, truly unrelated

Return exactly this JSON:
{"intent":"<class>","confidence":<0.0-1.0>,"entities":{"branch":<string|null>,"speciality":<string|null>,"doctor_name":<string|null>,"insurance":<string|null>,"day":<string|null>}}"""


@dataclass
class Intent:
    intent:       str
    confidence:   float
    branch:       Optional[str] = None
    speciality:   Optional[str] = None
    doctor_name:  Optional[str] = None
    insurance:    Optional[str] = None
    day:          Optional[str] = None
    original_query: str = ""

    @property
    def has_branch(self)    -> bool: return bool(self.branch)
    @property
    def has_speciality(self) -> bool: return bool(self.speciality)
    @property
    def has_doctor(self)    -> bool: return bool(self.doctor_name)


def _quick_check(query: str, has_history: bool) -> Optional[Intent]:
    """
    Rule-based fast paths — 0 API cost.
    Only fires for pure greetings with NO conversation history.
    If there IS history, short messages are follow-ups → classify normally.
    """
    q = query.lower().strip().rstrip(".,!?")

    if any(kw in q for kw in _EMERGENCY_KEYWORDS):
        return Intent(intent="emergency", confidence=1.0, original_query=query)

    if any(kw in q for kw in _COMPLAINT_KEYWORDS):
        return Intent(intent="complaint", confidence=1.0, original_query=query)

    # Only treat as out_of_scope if there's no prior conversation
    if not has_history and (q in _GREETINGS or len(query.split()) <= 2):
        return Intent(intent="out_of_scope", confidence=0.95, original_query=query)

    return None


def _parse_response(raw: str, query: str) -> Intent:
    """Parse JSON response. Extracts first {...} block as safety net."""
    raw = re.sub(r"```json\s*", "", raw).strip()
    raw = re.sub(r"```\s*",     "", raw).strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        log.warning("No JSON in classifier response: %s", raw[:100])
        return Intent(intent="general_health", confidence=0.3,
                      original_query=query)

    try:
        parsed   = json.loads(json_match.group(0))
        entities = parsed.get("entities", {})
        intent   = parsed.get("intent", "general_health")

        if intent not in _INTENT_VALUES:
            intent = "general_health"

        # Resolve branch aliases from extracted entity
        branch = entities.get("branch")
        if branch:
            resolved = _resolve_branch_alias(branch)
            if resolved:
                branch = resolved

        return Intent(
            intent=intent,
            confidence=float(parsed.get("confidence", 0.5)),
            branch=      branch,
            speciality=  entities.get("speciality"),
            doctor_name= entities.get("doctor_name"),
            insurance=   entities.get("insurance"),
            day=         entities.get("day"),
            original_query=query,
        )

    except (json.JSONDecodeError, ValueError) as e:
        log.warning("Intent parse error: %s | raw: %s", e, raw[:150])
        return Intent(intent="general_health", confidence=0.3,
                      original_query=query)


def classify(query: str,
             history: Optional[list[dict]] = None) -> Intent:
    """
    Classify patient query into Intent, using conversation history
    to resolve implicit references.

    Args:
        query:   The patient's latest message
        history: Recent conversation turns
                 [{"role": "user"|"assistant", "content": "..."}]
    """
    query = query.strip()
    history = history or []
    has_history = len(history) > 0

    # Fast-path rules
    quick = _quick_check(query, has_history)
    if quick:
        log.info("Fast-path intent: %s", quick.intent)
        return quick

    # Build context string from last 3 turns of history
    context_lines = []
    for msg in history[-6:]:  # last 3 turns = 6 messages
        role    = "Patient"  if msg["role"] == "user"      else "Assistant"
        content = msg["content"][:200]
        context_lines.append(f"{role}: {content}")

    if context_lines:
        context_block = "Recent conversation:\n" + "\n".join(context_lines)
        user_content  = f"{context_block}\n\nNew patient query: {query}"
    else:
        user_content  = f"Patient query: {query}"

    try:
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            temperature=0.0,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()
        intent = _parse_response(raw, query)
        log.info("Classified: %s (%.2f) branch=%s ins=%s spec=%s",
                 intent.intent, intent.confidence,
                 intent.branch, intent.insurance, intent.speciality)
        return intent

    except Exception as e:
        log.error("Intent API error: %s", e)
        return Intent(intent="general_health", confidence=0.1,
                      original_query=query)
