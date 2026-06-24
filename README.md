# hangover — a reliable long-term-memory agent

A terminal chat agent that **remembers across sessions**, with safety and reliability
built in. Claude does the reasoning; a Neo4j knowledge graph
(via [`neo4j-agent-memory`](https://pypi.org/project/neo4j-agent-memory/)) holds the memory.

Each turn runs through three layers, every one backed by a vetted skill rather than
hand-rolled code:

| Layer | What it does | Skill / tool |
|-------|--------------|--------------|
| **Guardrails** | Safety-checks every user input and assistant output (jailbreak, prompt-injection, harmful/illegal, system-prompt extraction) | NVIDIA **NeMo Guardrails** (`nemo-guardrails`) |
| **Memory** | Short-term conversation + long-term POLE+O entities/facts/preferences, recalled across sessions | **neo4j-agent-memory** (graph) |
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

## Prove the memory works

1. `You: My name is Pooja and I love graph databases.`
2. `You: What do I love?` → answers from the current session.
3. `:quit`, then `python -m memory_agent.main` again.
4. `You: What's my name?` → recalls **Pooja** from long-term memory across the restart.

Open http://localhost:7474 to browse the graph — you'll see `PERSON` entity nodes and
preference/fact relationships built from the conversation.

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

- Reasoning model is `claude-opus-4-8` (Anthropic SDK, streamed).
- Recall embeddings run locally via `sentence-transformers` (no embeddings API call).
- Memory persists in the `neo4j_data` Docker volume; `docker compose down -v` wipes it.

## Roadmap

- Document/RAG ingestion (chunk → embed → retrieve) for domain-knowledge Q&A.
- Expose memory read/write as Claude tools (`@beta_tool` + tool runner).
- Reasoning-trace memory; web/API frontend.
