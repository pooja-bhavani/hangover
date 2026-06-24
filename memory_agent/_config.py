"""Runtime mode selection.

Default = the hackathon stack: Cognee memory + Claude + Neo4j (needs API credits + Docker).

Set ``HANGOVER_LOCAL=1`` for a fully offline demo with **no API cost and no Docker**:
- LLM (reasoning, Cognee extraction, guardrail checks) → local **Ollama**
- Graph store → embedded **Kuzu** (no Neo4j)
- Embeddings → local **fastembed** (unchanged)

Quality is bounded by the local model; this mode is for "does it run / demo offline",
not for the strongest results.
"""

from __future__ import annotations

import os

LOCAL = os.environ.get("HANGOVER_LOCAL", "").strip().lower() in ("1", "true", "yes", "on")
OLLAMA_MODEL = os.environ.get("HANGOVER_OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

if LOCAL:
    # Single-user offline agent; the embedded Kuzu handler doesn't support Cognee's
    # default multi-user access control, so turn it off.
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
