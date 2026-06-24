"""Lightweight, OTel-aligned tracing for reliability (the ``observability-llm-obs`` skill).

Every turn emits a structured JSON event to ``traces/agent-trace.jsonl`` using
OpenTelemetry **GenAI semantic-convention** attribute names (``gen_ai.*``). That keeps
local runs dependency-free while remaining ingestible later by an OpenTelemetry
collector → Elastic (the Elastic skill queries exactly these ``traces*`` signals).

Captured per turn: latency, token usage, model, guardrail verdicts, memory-context
size, and any error — so failures are visible instead of silent.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRACE_DIR = Path(os.environ.get("HANGOVER_TRACE_DIR", "traces"))
_TRACE_FILE = _TRACE_DIR / "agent-trace.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Turn:
    """Accumulates attributes for one user turn and writes one trace line on close."""

    def __init__(self, session_id: str, model: str) -> None:
        self._start = time.monotonic()
        self.attrs: dict[str, Any] = {
            "@timestamp": _now_iso(),
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": model,
            "session.id": session_id,
        }

    def set(self, **kwargs: Any) -> None:
        self.attrs.update(kwargs)

    def usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is not None:
            self.attrs["gen_ai.usage.input_tokens"] = input_tokens
        if output_tokens is not None:
            self.attrs["gen_ai.usage.output_tokens"] = output_tokens

    def close(self, status: str = "ok") -> None:
        self.attrs["duration_ms"] = round((time.monotonic() - self._start) * 1000, 1)
        self.attrs["status"] = status
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        with _TRACE_FILE.open("a") as f:
            f.write(json.dumps(self.attrs) + "\n")
