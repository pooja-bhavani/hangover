"""Streamlit web UI for the hangover memory agent.

Same pipeline as the CLI (memory_agent/main.py), in a browser chat:
  input guardrail -> Cognee recall -> Claude -> output guardrail -> Cognee remember

Run:
    source .venv/bin/activate
    streamlit run app.py

The sidebar shows per-turn reliability signals (guardrail verdicts, tokens, latency,
memory-context size) so the memory + guardrail + observability story is visible live.
A stable session id means memory persists across browser sessions and restarts — the
heart of the "AI that doesn't forget" demo.
"""

from __future__ import annotations

import os
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from memory_agent import _config
from memory_agent.guardrails import Guardrails
from memory_agent.llm import MODEL, LLM
from memory_agent.main import SYSTEM_PROMPT, _build_messages
from memory_agent.memory import Memory

load_dotenv()

st.set_page_config(page_title="hangover — memory agent", page_icon="🧠")


@st.cache_resource(show_spinner="Loading guardrails + connecting Cognee memory…")
def _pipeline():
    """Build the heavy, stateful components once and reuse across reruns."""
    required = [] if _config.LOCAL else ["ANTHROPIC_API_KEY", "NEO4J_PASSWORD"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        st.error(f"Missing env vars: {', '.join(missing)} (see .env.example).")
        st.stop()
    llm = LLM()
    guard = Guardrails()
    memory = Memory()
    memory.connect()
    return llm, guard, memory


llm, guard, memory = _pipeline()

st.title("🧠 hangover")
st.caption("An AI that doesn't forget — Cognee memory · Claude · NeMo guardrails")

with st.sidebar:
    st.header("Session")
    session_id = st.text_input("Session id (stable = cross-session memory)", "default-session")
    st.caption(f"Reasoning model: `{MODEL}`")
    st.divider()
    st.header("Last turn")
    metrics = st.empty()

if "history" not in st.session_state:
    st.session_state.history = []  # list[(role, text)]

chat_tab, graph_tab = st.tabs(["💬 Chat", "🕸️ Memory graph"])

with graph_tab:
    st.caption("The Cognee knowledge graph built from your conversation (stored in Neo4j).")
    if st.button("Refresh graph"):
        try:
            html = memory.graph_html()
            components.html(html, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"Couldn't render the graph yet (chat a bit first?): {e}")

chat_box = chat_tab.container()
for role, text in st.session_state.history:
    chat_box.chat_message(role).write(text)

user_input = st.chat_input("Tell me something, or ask what I remember…")
if user_input:
    chat_box.chat_message("user").write(user_input)
    st.session_state.history.append(("user", user_input))
    started = time.monotonic()

    # 1. Input guardrail.
    gin = guard.check_input(user_input)
    if not gin.allowed:
        reply = gin.message or "Sorry, I can't help with that."
        chat_box.chat_message("assistant").write(reply)
        st.session_state.history.append(("assistant", reply))
        metrics.markdown(f"**input guardrail:** ❌ blocked\n\n**latency:** "
                         f"{(time.monotonic()-started)*1000:.0f} ms")
    else:
        with st.spinner("Recalling memory & thinking…"):
            context = memory.get_context(user_input, session_id)
            reply, usage = llm.chat(_build_messages(context, user_input), SYSTEM_PROMPT)
            gout = guard.check_output(user_input, reply)
            if not gout.allowed:
                reply = gout.message or "Sorry, I can't share that."
            memory.add_message(session_id, "user", user_input)
            memory.add_message(session_id, "assistant", reply)

        chat_box.chat_message("assistant").write(reply)
        st.session_state.history.append(("assistant", reply))
        metrics.markdown(
            f"**input guardrail:** ✅ allowed\n\n"
            f"**output guardrail:** {'✅ allowed' if gout.allowed else '❌ blocked'}\n\n"
            f"**memory context:** {len(context or '')} chars\n\n"
            f"**tokens:** in {usage.get('input_tokens')} / out {usage.get('output_tokens')}\n\n"
            f"**latency:** {(time.monotonic()-started)*1000:.0f} ms"
        )
