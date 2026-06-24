"""Long-term memory powered by **Cognee** — the hackathon's required memory layer.

Cognee builds a semantic knowledge graph from what the agent is told and recalls it
later. We configure it for a local, key-light setup:

- **LLM** (graph extraction + recall): **Anthropic / Claude** (`cognee[anthropic]`),
  so the only credential needed is ``ANTHROPIC_API_KEY``.
- **Graph store**: **Neo4j** (`cognee[neo4j]`), the same local Docker container.
- **Embeddings**: local **fastembed** (`cognee[fastembed]`, BGE-small, 384-dim) —
  no embeddings API call. Vector + relational stores default to local LanceDB/SQLite.

Cognee's API is module-level and async. ``add_message`` → ``cognee.remember`` (ingest +
build graph for a session); ``get_context`` → ``cognee.recall(..., only_context=True)``
(retrieve graph context for a query). Our CLI/eval harness is synchronous, so we drive
the coroutines on a dedicated event loop.
"""

from __future__ import annotations

import asyncio
import os

import cognee

EXTRACTION_MODEL = "claude-haiku-4-5"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # fastembed default; 384-dim
EMBEDDING_DIMS = 384


def _result_to_text(results) -> str:
    """Flatten Cognee recall/search results into a context string for the prompt."""
    if results is None:
        return ""
    if isinstance(results, str):
        return results
    parts: list[str] = []
    for r in results if isinstance(results, (list, tuple)) else [results]:
        for attr in ("context", "text", "answer", "content"):
            val = getattr(r, attr, None)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
        else:
            parts.append(str(r))
    return "\n".join(parts)


class Memory:
    """Synchronous facade over Cognee's async, module-level memory API."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def connect(self) -> None:
        """Configure Cognee: Claude LLM, local fastembed, Neo4j graph backend."""
        key = os.environ["ANTHROPIC_API_KEY"]
        cognee.config.set_llm_config(
            {"llm_provider": "anthropic", "llm_model": EXTRACTION_MODEL, "llm_api_key": key}
        )
        cognee.config.set_embedding_provider("fastembed")
        cognee.config.set_embedding_model(EMBEDDING_MODEL)
        cognee.config.set_embedding_dimensions(EMBEDDING_DIMS)
        cognee.config.set_graph_database_provider("neo4j")
        cognee.config.set_graph_db_config(
            {
                "graph_database_url": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                "graph_database_username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                "graph_database_password": os.environ["NEO4J_PASSWORD"],
            }
        )

    def close(self) -> None:
        self._loop.close()

    def get_context(self, query: str, session_id: str) -> str:
        """Retrieve graph context for the query (no answer generation)."""
        results = self._run(
            cognee.recall(query, session_id=session_id, only_context=True, top_k=10)
        )
        return _result_to_text(results)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Ingest a turn and build the knowledge graph for this session."""
        self._run(
            cognee.remember(
                f"{role}: {content}", session_id=session_id, self_improvement=False
            )
        )

    def flush(self) -> None:
        """Cognee persists synchronously per call; nothing extra to flush."""
        return None
