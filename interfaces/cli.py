"""
interfaces/cli.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Terminal REPL — production-grade debug panel.

Commands:
  /debug    — toggle full pipeline trace panel
  /clear    — clear conversation + session
  /eval     — show evaluation summary for this session
  /help     — show commands
  /quit     — exit

Run:
    python -m interfaces.cli
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.bot     import handle_message
from chatbot.prompt  import GREETING
from chatbot.session import clear as session_clear
from chatbot.debug   import render_terminal, EVAL_LOG

# ── ANSI colours ──────────────────────────────────────────────
R  = "\033[0m"
B  = "\033[1m"
T  = "\033[96m"    # teal — assistant
Y  = "\033[93m"    # yellow — user
G  = "\033[90m"    # grey — system
RE = "\033[91m"    # red — error
GR = "\033[92m"    # green


def _banner() -> None:
    print(f"\n{B}{T}{'━'*60}{R}")
    print(f"{B}{T}  🏥  HealthHub by Al-Futtaim — Patient Assistant{R}")
    print(f"{B}{T}{'━'*60}{R}")
    print(f"{G}  /help for commands  |  /debug for pipeline trace  |  /quit to exit{R}\n")


def _print_bot(text: str) -> None:
    print(f"\n{T}{B}🤖 Assistant:{R}")
    for line in text.split("\n"):
        print(f"   {line}")
    print()


def _help() -> None:
    print(f"""
{T}{B}Commands:{R}
  {Y}/debug{R}    — toggle full pipeline trace (intent, SQL, vector, BM25, RRF, LLM cost)
  {Y}/eval{R}     — session evaluation summary (accuracy, cost, latency stats)
  {Y}/clear{R}    — clear conversation history and start fresh
  {Y}/help{R}     — show this message
  {Y}/quit{R}     — exit
""")


def _eval_summary(traces: list) -> None:
    """Print a session-level evaluation summary from collected traces."""
    if not traces:
        print(f"{G}  No turns to evaluate yet.{R}\n")
        return

    real_turns  = [t for t in traces if t.llm.input_tokens > 0]
    fast_turns  = [t for t in traces if t.intent.was_fast_path]
    total_cost  = sum(t.llm.cost_usd for t in traces)
    total_ms    = sum(t.total_ms for t in traces)
    avg_ms      = total_ms / len(traces) if traces else 0

    # Intent distribution
    from collections import Counter
    intent_counts = Counter(t.intent.intent for t in traces)

    # Avg confidence
    conf_values = [t.intent.confidence for t in traces if t.intent.confidence > 0]
    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0

    # SQL hit rate
    sql_hits = sum(1 for t in traces if t.sql_debug.total_rows > 0)

    # Vector hit rate
    vec_hits = sum(1 for t in traces if t.vec_debug.chunks_returned > 0)

    print(f"\n{G}{'═'*60}{R}")
    print(f"{B}{T}  SESSION EVALUATION SUMMARY{R}")
    print(f"{G}{'═'*60}{R}")
    print(f"{G}  Total turns:         {B}{len(traces)}{R}")
    print(f"{G}  API calls made:      {B}{len(real_turns) * 2}{R}{G} ({len(fast_turns)} fast-path, $0){R}")
    print(f"{G}  Total session cost:  {GR}{B}${total_cost:.4f} USD{R}")
    print(f"{G}  Avg cost/turn:       {GR}${total_cost/len(traces):.4f}{R}")
    print(f"{G}  Avg latency/turn:    {B}{avg_ms:.0f}ms{R}")
    print(f"{G}  Avg intent conf:     {B}{avg_conf:.0%}{R}")
    print(f"{G}  SQL hit rate:        {B}{sql_hits}/{len(traces)} turns{R}")
    print(f"{G}  Vector hit rate:     {B}{vec_hits}/{len(traces)} turns{R}")
    print(f"\n{G}  Intent distribution:{R}")
    for intent, count in intent_counts.most_common():
        bar = "█" * count
        print(f"{G}    {intent:<22} {T}{bar}{R} {count}")
    print(f"\n{G}  Eval log:  {EVAL_LOG}{R}")
    print(f"{G}{'═'*60}{R}\n")


def main() -> None:
    _banner()
    _print_bot(GREETING)

    session_id  = str(uuid.uuid4())
    debug_mode  = False
    session_traces = []   # collect PipelineTrace objects for /eval

    while True:
        print(f"{Y}{B}👤 You:{R} ", end="", flush=True)
        try:
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{G}  Goodbye! Stay healthy. 👋{R}\n")
            sys.exit(0)

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Commands ──────────────────────────────────────────
        if cmd in ("/quit", "/exit", "/q"):
            print(f"\n{G}  Goodbye! Stay healthy. 👋{R}\n")
            sys.exit(0)

        if cmd == "/help":
            _help()
            continue

        if cmd == "/clear":
            session_clear(session_id)
            session_id     = str(uuid.uuid4())
            session_traces = []
            print(f"{G}  ↺  Conversation cleared. Starting fresh.{R}\n")
            continue

        if cmd == "/debug":
            debug_mode = not debug_mode
            state = f"{GR}ON{R}" if debug_mode else f"{RE}OFF{R}"
            print(f"{G}  🔧 Debug mode: {state}{R}\n")
            continue

        if cmd == "/eval":
            _eval_summary(session_traces)
            continue

        # ── Normal message ────────────────────────────────────
        print(f"{G}  ⏳ Thinking...{R}", end="\r", flush=True)

        try:
            result = handle_message(
                user_message=user_input,
                session_id=session_id,
                debug=debug_mode,
            )
        except Exception as e:
            print(f"\n{RE}  ❌ Error: {e}{R}\n")
            import traceback
            traceback.print_exc()
            continue

        # Clear the "Thinking..." line
        print(" " * 30, end="\r")

        _print_bot(result["reply"])

        trace = result.get("trace")
        if trace:
            session_traces.append(trace)

        if debug_mode and trace:
            print(render_terminal(trace))


if __name__ == "__main__":
    main()
