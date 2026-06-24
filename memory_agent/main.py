"""Terminal chat loop: guard input -> retrieve memory -> reason -> guard output -> store.

Run with:  python -m memory_agent.main
Type a message, or ``:quit`` to exit.

Reliability layers (each from an installed skill):
- **Guardrails** (NeMo): every user input and every assistant output passes a safety
  rail before it is used / shown.
- **Observability**: every turn writes an OTel-style trace line to traces/agent-trace.jsonl.
- **Memory** (Neo4j graph): short-term + long-term recall, persisted across sessions.

A stable ``session_id`` is reused across runs so both conversation and long-term graph
memory carry over between sessions.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from .guardrails import Guardrails
from .llm import MODEL, LLM
from .memory import Memory
from .observability import Turn

SESSION_ID = os.environ.get("HANGOVER_SESSION_ID", "default-session")

SYSTEM_PROMPT = (
    "You are a helpful assistant with persistent long-term memory about the user.\n"
    "Before each reply you are given a MEMORY CONTEXT block retrieved from a knowledge "
    "graph of past conversations (people, facts, preferences) plus recent messages.\n"
    "Use it to answer questions about the user and to stay consistent across sessions. "
    "If the memory context is empty or irrelevant, just answer normally and never invent "
    "remembered facts."
)


def _build_messages(memory_context: str, user_input: str) -> list[dict]:
    user_block = (
        f"MEMORY CONTEXT (retrieved; may be empty):\n{memory_context or '(none)'}\n\n"
        f"USER MESSAGE:\n{user_input}"
    )
    return [{"role": "user", "content": user_block}]


def main() -> int:
    load_dotenv()
    for var in ("ANTHROPIC_API_KEY", "NEO4J_PASSWORD"):
        if not os.environ.get(var):
            print(f"error: {var} is not set (see .env.example)", file=sys.stderr)
            return 1

    llm = LLM()
    memory = Memory()
    print("Loading guardrails…")
    guard = Guardrails()
    print("Connecting to memory (Neo4j)…")
    memory.connect()
    print(f"Ready. Session: {SESSION_ID}. Type a message, or :quit to exit.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input in (":quit", ":q", ":exit"):
                break

            turn = Turn(SESSION_ID, MODEL)
            try:
                # 1. Input guardrail.
                gin = guard.check_input(user_input)
                turn.set(**{"guardrail.input.allowed": gin.allowed})
                if gin.error:
                    turn.set(**{"guardrail.input.error": gin.error})
                if not gin.allowed:
                    msg = gin.message or "Sorry, I can't help with that."
                    print(f"\nAssistant: {msg}\n")
                    turn.close(status="blocked_input")
                    continue

                # 2. Retrieve memory context.
                context = memory.get_context(user_input, SESSION_ID)
                turn.set(**{"memory.context_chars": len(context or "")})

                # 3. Reason with Claude.
                reply, usage = llm.chat(_build_messages(context, user_input), SYSTEM_PROMPT)
                turn.usage(usage.get("input_tokens"), usage.get("output_tokens"))

                # 4. Output guardrail.
                gout = guard.check_output(user_input, reply)
                turn.set(**{"guardrail.output.allowed": gout.allowed})
                if gout.error:
                    turn.set(**{"guardrail.output.error": gout.error})
                if not gout.allowed:
                    reply = gout.message or "Sorry, I can't share that."

                print(f"\nAssistant: {reply}\n")

                # 5. Persist the turn; long-term extraction builds the graph.
                memory.add_message(SESSION_ID, "user", user_input)
                memory.add_message(SESSION_ID, "assistant", reply)
                turn.close(status="ok")
            except Exception as e:  # never let one bad turn kill the session
                turn.set(**{"error.message": str(e)})
                turn.close(status="error")
                print(f"\n[error] {e}\n", file=sys.stderr)
    finally:
        print("Saving memory…")
        try:
            memory.flush()
        finally:
            memory.close()
        print("Bye.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
