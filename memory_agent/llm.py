"""Claude reasoning layer.

Thin wrapper over the official Anthropic SDK. The model id is exactly
``claude-opus-4-8`` (Anthropic's most capable Opus-tier model). The Messages API
is stateless, so callers pass the full message history each turn; we stream and
return the assembled final message to stay under SDK HTTP timeouts.
"""

from __future__ import annotations

import anthropic

MODEL = "claude-opus-4-8"
MAX_TOKENS = 16_000


class LLM:
    def __init__(self) -> None:
        # Reads ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic()

    def chat(self, messages: list[dict], system: str) -> tuple[str, dict]:
        """Send the conversation to Claude; return ``(text, usage)``.

        ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.
        ``system`` is the system prompt (includes retrieved long-term memory).
        ``usage`` is ``{"input_tokens": int, "output_tokens": int}`` for tracing.
        """
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
