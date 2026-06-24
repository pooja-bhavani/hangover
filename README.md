# hangover — a reliable long-term-memory agent 

> Built for the **WeMakeDevs × Cognee "The Hangover Part AI: Where's My Context?"**
> hackathon — *build AI that doesn't forget*. **Cognee is the memory layer** (the event's
> one hard rule), with Claude reasoning and production guardrails/observability on top.

A terminal chat agent that **remembers across sessions**, with safety and reliability
built in. Claude does the reasoning; **Cognee** builds a semantic knowledge graph
(stored in Neo4j) that the agent recalls from in later sessions.

Each turn runs through three layers, every one backed by a vetted skill rather than
hand-rolled code:

| Layer | What it does | Skill / tool |
|-------|--------------|--------------|
| **Memory** | Builds a semantic knowledge graph from the conversation and recalls it across sessions | **Cognee** (`cognee`, graph on Neo4j, local fastembed) |
| **Guardrails** | Safety-checks every user input and assistant output (jailbreak, prompt-injection, harmful/illegal, system-prompt extraction) | NVIDIA **NeMo Guardrails** (`nemo-guardrails`) |
| **Observability** | Writes an OTel-GenAI trace line per turn (latency, tokens, guardrail verdicts, errors) | **observability-llm-obs** (Elastic) |
| **Evaluation** | Behavioral/regression tests for guardrails + memory recall | **agent-evaluation** |

```
You ─▶ [input guardrail] ─▶ retrieve memory ─▶ Claude (claude-opus-4-8)
                                                   │
       store turn ◀─ [output guardrail] ◀─────────┘     (every turn traced)
```

## Prerequisites

- Docker (for local Neo4j)
- **Python 3.12 or 3.13** — NeMo Guardrails does **not** import on 3.14
- An Anthropic API key

## Setup

```bash
# 1. Start Neo4j (set a password first)
cp .env.example .env
#   edit .env: set ANTHROPIC_API_KEY and NEO4J_PASSWORD
export NEO4J_PASSWORD=$(grep NEO4J_PASSWORD .env | cut -d= -f2)
docker compose up -d            # Neo4j Browser at http://localhost:7474

# 2. Python env (3.12 — NeMo Guardrails does not support 3.14)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python -m memory_agent.main
```

Commands inside the chat: type a message, or `:quit` to exit.

### Web UI (recommended for demos)

A Streamlit chat app over the same pipeline — a **💬 Chat** tab (live guardrail verdicts,
token usage, latency, and memory-context size in the sidebar) and a **🕸️ Memory graph** tab
that renders the Cognee knowledge graph as it grows. Memory persists across sessions:

```bash
source .venv/bin/activate
streamlit run app.py            # opens http://localhost:8501
```

## Prove the memory works

1. `You: My name is Pooja and I love graph databases.`
2. `You: What do I love?` → answers from the current session.
3. `:quit`, then `python -m memory_agent.main` again.
4. `You: What's my name?` → recalls **Pooja** from long-term memory across the restart.

Open http://localhost:7474 to browse the graph — you'll see the entity nodes and
relationships Cognee extracted from the conversation.

## Reliability

**Run the eval suite** (validates guardrails block jailbreaks and that memory recalls a
stated fact — needs live Neo4j + API key; exits non-zero on failure, so it can gate CI):

```bash
source .venv/bin/activate
python -m evals.run_evals
```

**Traces:** every turn appends one OTel-GenAI line to `traces/agent-trace.jsonl`
(`gen_ai.usage.*`, guardrail verdicts, latency, errors). Inspect with:

```bash
tail -f traces/agent-trace.jsonl | jq .
```

These are the exact `gen_ai.*` signals the Elastic `observability-llm-obs` skill queries,
so you can later ship them to an OpenTelemetry collector → Elastic without changing the app.

**Guardrails:** input/output safety is LLM-self-check via NeMo, powered by Claude
(`claude-haiku-4-5`). It **fails open** on a guard outage (logs the error, lets the turn
proceed) so a transient failure degrades safety rather than bricking the agent — flip
`FAIL_CLOSED` in `memory_agent/guardrails.py` to invert.

**CI:** `.github/workflows/evals.yml` runs the eval suite on every push/PR against a
Neo4j service container on Python 3.12. Add an `ANTHROPIC_API_KEY` repo secret
(Settings → Secrets and variables → Actions) to enforce it; without the secret the job
warns and skips rather than failing forks.

## Notes

- Reasoning model is `claude-opus-4-8` (Anthropic SDK, streamed); Cognee's graph
  extraction/recall uses `claude-haiku-4-5`.
- Cognee builds the knowledge graph in **Neo4j** and embeds locally via **fastembed**
  (BGE-small, no embeddings API call); its vector/relational stores default to local
  LanceDB/SQLite.
- Graph memory persists in the `neo4j_data` Docker volume; `docker compose down -v` wipes it.

## Roadmap

- Document/RAG ingestion (chunk → embed → retrieve) for domain-knowledge Q&A.
- Expose memory read/write as Claude tools (`@beta_tool` + tool runner).
- Reasoning-trace memory; web/API frontend.
