"""Graph-native long-term memory, backed by Neo4j via ``neo4j-agent-memory`` (0.5.x).

The 0.5 API is synchronous. We configure it for a local, key-light setup:

- **Embeddings:** local ``sentence-transformers`` (MiniLM, 384-dim) — no embeddings API.
- **Extraction:** the pure-LLM extractor using **Anthropic** (Claude) — so the only
  external credential needed is ``ANTHROPIC_API_KEY``. This avoids the default
  OpenAI/spaCy/GLiNER pipeline and its extra model downloads.

``add_message`` stores a turn in short-term memory and (by default) extracts entities,
facts, and preferences into the long-term knowledge graph. ``get_context`` returns a
ready-to-inject text blob combining short-term and long-term recall for a query.
"""

from __future__ import annotations

import os

from neo4j_agent_memory import (
    EmbeddingConfig,
    EmbeddingProvider,
    ExtractionConfig,
    ExtractorType,
    LLMConfig,
    LLMProvider,
    MemoryClient,
    MemorySettings,
    Neo4jConfig,
)

# Cheap Claude model for the frequent extraction calls; reasoning uses Opus (see llm.py).
EXTRACTION_MODEL = "claude-haiku-4-5"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384


def _settings() -> MemorySettings:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    return MemorySettings(
        neo4j=Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ["NEO4J_PASSWORD"],
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMS,
            device="cpu",
        ),
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model=EXTRACTION_MODEL,
            api_key=anthropic_key,
        ),
        extraction=ExtractionConfig(
            extractor_type=ExtractorType.LLM,
            enable_spacy=False,
            enable_gliner=False,
            enable_llm_fallback=True,
            llm_model=EXTRACTION_MODEL,
        ),
    )


class Memory:
    """Thin facade over MemoryClient with connect/close lifecycle."""

    def __init__(self) -> None:
        self._client = MemoryClient(_settings())

    def connect(self) -> None:
        self._client.connect()

    def close(self) -> None:
        self._client.close()

    def get_context(self, query: str, session_id: str) -> str:
        """Retrieved short-term + long-term context for the current query."""
        return self._client.get_context(query, session_id=session_id, max_items=10)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Persist a turn; entity/fact/preference extraction runs from it."""
        self._client.short_term.add_message(session_id, role, content)

    def flush(self) -> None:
        """Block until pending background extraction finishes (call before exit)."""
        self._client.wait_for_pending()
