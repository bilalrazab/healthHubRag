"""
chatbot/session.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per-user conversation history management.

In-memory for terminal (V1).
WhatsApp (V2): swap _store dict for Redis in production.
Max 10 turns kept to avoid LLM context overflow.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

MAX_TURNS = 10   # 10 turns = 20 messages

# In-memory store: {session_id: [{"role": ..., "content": ...}]}
_store: dict[str, list[dict]] = {}


def get(session_id: str) -> list[dict]:
    return list(_store.get(session_id, []))


def append(session_id: str, role: str, content: str) -> None:
    if session_id not in _store:
        _store[session_id] = []
    _store[session_id].append({"role": role, "content": content})
    # Keep only last MAX_TURNS turns (2 messages per turn)
    if len(_store[session_id]) > MAX_TURNS * 2:
        _store[session_id] = _store[session_id][-(MAX_TURNS * 2):]


def clear(session_id: str) -> None:
    _store.pop(session_id, None)
