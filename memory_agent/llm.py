"""Claude reasoning layer.

Thin wrapper over the official Anthropic SDK. The model id is exactly
``claude-opus-4-8`` (Anthropic's most capable Opus-tier model). The Messages API
is stateless, so callers pass the full message history each turn; we stream and
return the assembled final message to stay under SDK HTTP timeouts.
"""

from __future__ import annotations

import anthropic
import httpx

from . import _config

MODEL = _config.OLLAMA_MODEL if _config.LOCAL else "claude-opus-4-8"
MAX_TOKENS = 16_000


class LLM:
    def __init__(self) -> None:
        # Reads ANTHROPIC_API_KEY from the environment (default mode only).
        self._client = None if _config.LOCAL else anthropic.Anthropic()

    def _chat_ollama(self, messages: list[dict], system: str) -> tuple[str, dict]:
        """Local reasoning via Ollama's /api/chat (no API cost)."""
        payload = {
            "model": _config.OLLAMA_MODEL,
            "stream": False,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        r = httpx.post(f"{_config.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "")
        usage = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        }
        return text, usage

    def chat(self, messages: list[dict], system: str) -> tuple[str, dict]:
        """Send the conversation to Claude; return ``(text, usage)``.

        ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.
        ``system`` is the system prompt (includes retrieved long-term memory).
        ``usage`` is ``{"input_tokens": int, "output_tokens": int}`` for tracing.
        """
        if _config.LOCAL:
            return self._chat_ollama(messages, system)

        with self._client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        ) as stream:
            final = stream.get_final_message()

        text = "".join(b.text for b in final.content if b.type == "text")
        usage = {
            "input_tokens": getattr(final.usage, "input_tokens", None),
            "output_tokens": getattr(final.usage, "output_tokens", None),
        }
        return text, usage
