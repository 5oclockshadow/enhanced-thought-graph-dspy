# Graphos — Use Cases & Positioning

Graphos turns a question (or a corpus) into an **auditable reasoning graph**: a
set of typed thoughts and weighted relationships, organized into clusters, with
a plan, background agents, and saved artifacts. This document grounds *what it
is for* in the market, based on a scan of enterprise knowledge-graph / GraphRAG
practice (late 2025 – 2026).

## Where graph intelligence wins (and where it doesn't)

Across vendor and research sources the pattern is consistent: **vector RAG wins
on specific-document lookup; graph approaches win on multi-hop reasoning,
aggregation, cross-document synthesis, and — critically — explainability**,
where a regulator or a reviewer needs the *reasoning path*, not just an answer.
([When to use Graphs in RAG, arXiv 2506.05690](https://arxiv.org/pdf/2506.05690),
[Trantor](https://www.trantorinc.com/blog/knowledge-graphs-enterprise-ai))

The flip side, which Graphos respects: graph maintenance has real operational
cost, and a well-governed vector index can beat a graph for simple queries at a
fraction of the price. The honest guidance is to **start with one narrow,
high-value use case** rather than model an entire enterprise.
([tianpan.co](https://tianpan.co/blog/2026/04/09/graphrag-production-when-vector-search-hits-ceiling),
[Motadata](https://www.motadata.com/blog/enterprise-knowledge-graph))

## Target use cases (mapped to the built-in eval suite)

| Use case | What Graphos produces | Why a graph |
| :-- | :-- | :-- |
| **Customer-360 / churn** | Account → contracts → support → adoption → churn-risk cluster, with a reasoning path per at-risk account | Churn is multi-signal and relational; the graph makes the *why* defensible ([Datahub](https://datahubanalytics.com/graphrag-the-next-phase-of-enterprise-knowledge-retrieval/)) |
| **Fraud / financial crime** | Coordinated activity across accounts, devices, identities | Fraud is a connectivity pattern, invisible to row-by-row analysis ([TigerGraph](https://www.tigergraph.com/blog/knowledge-graph-enterprise-ai/)) |
| **Supply-chain risk** | Supplier → component → fulfillment propagation of a disruption | Blast-radius is a graph-traversal question |
| **Incident postmortem** | Causal graph of a cascading outage, root causes ranked | Multi-hop cause→effect chains with provenance |
| **Competitive / market intel** | Threats, moats, and moves as a connected landscape | Aggregation across many sources |
| **Research synthesis** | Approaches, claims, contradictions across a literature | Cross-document reasoning; pairs with RAPTOR for long docs |

These six are the exact cases in `graphos/evals.py`, so the product's claims are
continuously regression-tested, not just asserted.

## The differentiator: the reasoning trail *is* the product

Graphos's left→right progression (chat → plan → background agents → artifacts)
is not decoration — it is an **audit trail**. Every generation is persisted as a
`run` (exact input, parameters, engine, latency, error) linked to the graph and
its artifacts. That is precisely the "explainability / compliance" edge the
research says graphs hold over vanilla RAG, and the observability + persistence
that separate a production system from a demo.
([LLM observability, tianpan.co](https://tianpan.co/blog/2025-10-17-llm-observability-production),
[Omdena](https://www.omdena.com/blog/why-llm-applications-fail-silently-and-how-observability-makes-them-production-ready))

## RAPTOR vs. GraphRAG — how they fit here

- **GraphRAG** (entity–relationship graphs, community clustering) is what
  Graphos's thought-graph engine resembles: good for multi-hop and aggregation.
- **RAPTOR** (recursive, hierarchical *summary tree* — parent/child summary
  links, not entity relations) is the right **retrieval layer for long,
  multi-section documents**, and is the natural next ingestion component.
  ([RAPTOR guide, Meilisearch](https://www.meilisearch.com/blog/raptor-rag),
  [arXiv 2503.02922](https://arxiv.org/pdf/2503.02922))

The roadmap therefore keeps the entity-graph reasoning core and adds RAPTOR as a
document-ingestion front end, rather than conflating the two.

## Production posture (what "not a toy" means here)

1. **Persistence** — SQLite-backed sessions, graphs, artifacts; survives restart.
2. **Audit & observability** — every run logged with input/params/engine/latency/error; request IDs; structured error envelopes.
3. **Async generation** — background job model so slow model calls never block.
4. **Evaluation before ship** — a 6-case suite with structural + validity metrics, run in CI and against the live model.
5. **Config-driven** — node caps, CORS, timeouts, DB path, live on/off via env.
6. **Graceful degradation** — auto-detects a model from the environment; falls back to deterministic, data-derived scoring when none is present.

> Sources are practitioner and vendor blogs plus a few arXiv papers; headline
> ROI numbers in that literature (e.g. "90% hallucination reduction") are
> vendor-reported and not independently verified. The architectural guidance —
> graphs for multi-hop/explainability, eval-before-ship, observability as the
> production minimum — is consistent across independent sources.
