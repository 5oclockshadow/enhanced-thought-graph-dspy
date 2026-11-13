# Maximillian Skill — Autonomous Agentic Engineering & DSPy Optimization Engine

Maximillian is a closed-loop, graph-guided autonomous software engineering skill built for high-rigor Python, DSPy, and local LLM ecosystems. It orchestrates a strict 14-stage execution pipeline that audits, synthesizes, refactors, executes, tests-till-pass, logs verbose DSPy training telemetry, snapshots state, and generates rich HTML diagnostic reports.

---

## Repository & Workspace Layout

```text
maximillian/
├── __init__.py
├── state_machine.py        # Core 14-stage DAG orchestrator & loop controller + CLI
├── schemas.py              # Pydantic v2 models for all stage inputs/outputs
├── graph_intel.py          # NetworkX AST parser, dependency graph & blast-radius analyzer
├── auditor.py              # Static analyzer, complexity scorer & invariant checker
├── synthesizer.py          # DSPy modules for zero-exec Build & Precision Refactoring
├── test_harness.py         # Subprocess sandbox runner & Test-Till-Pass recovery loop
├── telemetry.py            # Dual-ledger serializer (Verbose YAML/JSONL vs Clean Git)
├── snapshotter.py          # Preflight verifier, Git automator & state snapshotter
├── reporter.py             # Standalone HTML/SVG round report generator
└── optimize_from_ledger.py # Compile DSPy optimizers (MIPROv2) from the ledger
tracking/                   # [GIT-IGNORED] Verbose telemetry for DSPy Optimizers
├── maximillian_ledger.yaml # Human-readable hierarchical round metadata
└── dspy_trainset.jsonl     # Ready-to-load DSPy Example() records for MIPROv2
reports/                    # Standalone HTML diagnostic reports per round
└── round_001_<ts>.html
tests/test_maximillian.py   # Pytest unit, integration & invariant test suites
PRD.md                      # Product Requirements Document
SKILL.md                    # Agent Skill Directive & Execution Protocol
```

## Core Subsystems

### 1. Graph Intelligence (`graph_intel.py`)
Parses the target codebase into a `networkx.DiGraph`:

* **Nodes:** Modules, Classes, Functions.
* **Edges:** `IMPORTS`, `INHERITS`, `CALLS`, `DEFINES`.
* **Metrics:** Computes PageRank centrality to identify architectural bottlenecks and calculates the exact **Blast Radius** subgraph (upstream dependents) before any refactoring patch is applied.

### 2. Zero-Execution Build & Dual-Audit Pipeline (`auditor.py` & `synthesizer.py`)

* **Build Phase:** Generates pure Python source via DSPy Signatures (`BuildSignature`) without executing code. In deterministic mode (no model) the stage is a recorded no-op.
* **Audit → Refactor → Audit → Refactor:** Alternates deterministic static analysis (`ast`, cyclomatic complexity, optional `ruff`, custom invariant rules) with LLM-driven structural refactoring.

### 3. Test-Till-Pass Harness (`test_harness.py`)
Executes `pytest` inside an isolated `python -I` subprocess with a wall-clock cap. On failure it:

1. Parses failing tests into a structured `FailureNode`.
2. Queries `GraphIntel` for the blast radius of the failing symbol.
3. Invokes the `PatchSynthesizer` with that subgraph context and traceback (when a model is configured).
4. Applies the patch, re-runs, and repeats until `exit_code == 0` or `max_retries`.

### 4. Dual-Ledger Persistence (`telemetry.py`)

* **Git Repository:** Clean, atomic commits only after Test-Till-Pass succeeds and Preflight clears (opt-in via `--enable-git`).
* **External Telemetry (`tracking/`):** DSPy signatures used, iteration counts to pass, scalar reward `R ∈ [0.0, 1.0]` (test-pass speed + diff minimality + complexity reduction), directly loadable for `MIPROv2` / `BootstrapFewShotWithRandomSearch`.

## Quickstart & CLI Invocation

```bash
# Run a single full Maximillian pass on a target directory
python -m maximillian.state_machine \
  --target ./graphos \
  --tests ./tests \
  --max-rounds 3 \
  --max-test-retries 8

# With a live local model, the Build/Refactor/Patch stages activate:
MAXIMILLIAN_MODEL="ollama/qwen2.5-coder:32b" \
python -m maximillian.state_machine --target ./src --tests ./tests

# Compile DSPy optimizers from accumulated tracking logs
python -m maximillian.optimize_from_ledger \
  --ledger ./tracking/dspy_trainset.jsonl \
  --optimizer MIPROv2 \
  --output ./models/maximillian_compiled_v5.json
```

## Deterministic vs. live mode

| Stage group | No model (`MAXIMILLIAN_MODEL` unset) | Live model |
| :-- | :-- | :-- |
| Audit, Graph, Invariants, Execute, Test, Triage, Plan, Telemetry, Snapshot, Report | Fully run | Fully run |
| Build / Refactor / Patch (Stages 02, 04, 06, 08-recovery) | Recorded no-op with reason | Real DSPy synthesis |

Every number in the HTML report — findings, graph metrics, test iterations, reward — is computed from real analysis, independent of model availability.
