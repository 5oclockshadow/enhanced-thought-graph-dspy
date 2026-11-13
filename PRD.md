# Product Requirements Document (PRD): The Maximillian Skill
**Version:** 5.2.0-enterprise
**Architecture Paradigm:** Neuro-Symbolic Agentic State Machine & Recursive Optimization Engine
**Target Frameworks:** DSPy (MIPROv2 / BootstrapFewShot / Signature Assertions), Pydantic v2, NetworkX, Pytest, Local Inference (`llama.cpp` / Ollama / OpenInference)

---

## 1. Executive Summary & Architectural Vision
The **Maximillian Skill** is an autonomous, deterministic-first engineering loop designed to execute complex software auditing, static AST graph analysis, zero-execution code synthesis, iterative refactoring, sandboxed test-till-pass validation, and granular telemetry extraction.

Unlike standard conversational coding assistants that generate unverified snippets, Maximillian operates as a closed-loop **Directed Acyclic Graph (DAG) with bounded cyclic recovery**. It enforces strict type contracts via Pydantic v2 schemas, maps structural dependencies into semantic knowledge graphs, tracks verbose multi-dimensional runtime metadata outside standard version control for downstream **DSPy prompt and weight optimization**, and outputs deterministic HTML diagnostic reports per execution round.

> **Implementation status in this repository.** Every *deterministic* stage is fully
> implemented and tested (`maximillian/`): AST → NetworkX graph intelligence, static
> auditor, invariant checks, sandboxed execution, the test-till-pass runner, dual-ledger
> telemetry, preflight/snapshot, and the air-gapped HTML reporter. The three *LLM-synthesis*
> stages (Build / Refactor / Patch) are real `dspy.Module` seams that activate when a model
> is configured via `MAXIMILLIAN_MODEL`; with no model they run in **deterministic no-op
> mode** and record the reason, so the loop never fabricates code.

---

## 2. Core Design Principles & Invariants
1. **Air-Gapped & Deterministic First:** All AST parsing, dependency mapping, linting, and static verification occur locally before invoking LLM synthesis. Code generation ("Build") strictly produces static code without premature side-effect execution.
2. **Dual-Ledger State Persistence:**
   * **Git Ledger:** Clean, atomic, human-readable commits representing verified, passing states (`test-till-pass` green).
   * **Optimizer Telemetry Ledger (`tracking/maximillian_ledger.yaml` & `.jsonl`):** High-verbosity, structured execution traces containing DSPy signature inputs/outputs, assertion pass/fail ratios, token latencies, AST diff metrics, and traceback recovery trajectories. This ledger is explicitly excluded from standard Git history (`.gitignore`) to prevent repository bloat while serving as a high-signal training corpus for `dspy.teleprompt` optimizers.
3. **Bounded Cyclic Convergence (`Test-Till-Pass`):** Execution enters an isolated sandboxed test harness. If assertions fail, tracebacks are parsed into structured error nodes, mapped against the dependency graph, and fed into a targeted refactoring module until all unit/integration tests pass or `max_iterations` triggers a safe rollback.
4. **Triage & Opportunity Scoring:** After a full pass, Maximillian ingests all workspace files and runtime logs, computes technical debt and cyclomatic complexity deltas, and determines whether a high-ROI refactoring or feature-synthesis target warrants initiating an additional round.

---

## 3. Detailed 14-Stage Execution Lifecycle (The Maximillian Loop)

| Stage | Phase Name | Operation & Technical Mechanism | Output Artifact |
| :--- | :--- | :--- | :--- |
| **01** | **Initial Audit** | AST parsing, NetworkX dependency graph construction, cyclomatic complexity scoring, dead-code detection, and schema validation. | `AuditReport` (Pydantic model) |
| **02** | **Build (Static Synthesis)** | Pure code generation (`no-exec` mode). Synthesizes module skeletons, Pydantic v2 data models, and DSPy Signatures without running code. | Staged `.py` source files |
| **03** | **Full-Context Read** | Ingests newly built files, existing modules, interface contracts, and historical logs into a structured context graph. | `ContextGraph` state object |
| **04** | **Structural Refactor** | Applies SOLID decomposition, decouples tight dependencies, injects type hints, and optimizes DSPy module chaining. | Refactored AST & source diffs |
| **05** | **Secondary Audit** | Verifies refactored code against architectural invariants, circular dependency checks, and security/sandbox rules. | `InvariantCheckResult` |
| **06** | **Precision Refactor** | Micro-adjustments resolving any static warnings, mypy/ruff violations, or signature mismatches flagged in Stage 05. | Clean, lint-compliant codebase |
| **07** | **Sandboxed Execution** | Executes target modules in an isolated subprocess/container with resource caps, capturing `stdout`, `stderr`, and OpenInference traces. | `ExecutionTrace` & runtime logs |
| **08** | **Test-Till-Pass Loop** | Runs `pytest` suite. On failure: extracts traceback → localizes fault node → synthesizes patch → re-tests until 100% green. | Verified passing test suite |
| **09** | **Log & File Ingestion** | Reads all workspace files, test coverage reports, and raw runtime logs to evaluate overall system health and emergent anomalies. | `SystemHealthSnapshot` |
| **10** | **Opportunity Triage** | Evaluates if a "decent and worth tackling" high-leverage optimization or architectural upgrade exists for the current round. | `TriageDecision` (Boolean + ROI score) |
| **11** | **Strategic Audit & Plan** | If triage is positive, performs deep-dive root-cause audit and generates a step-by-step DAG execution plan for the next cycle. | `RoundExecutionPlan` |
| **12** | **Telemetry & Git Sync** | Appends verbose DSPy training metadata to `tracking/ledger.yaml` / `.jsonl`. Commits clean code changes to Git (`git commit`). | Updated YAML/JSON ledger + Git SHA |
| **13** | **Preflight & Snapshot** | Runs final environment sanity checks (imports, disk, clean working tree) and creates a tagged state snapshot (`git tag` / archive). | Immutable state snapshot |
| **14** | **HTML Diagnostic Report** | Compiles AST metrics, test-till-pass iterations, DSPy scores, and diffs into a standalone HTML report saved in `/reports/`. | `reports/round_<id>_<ts>.html` |

---

## 4. Functional & Non-Functional Requirements
* **FR-01 (No-Exec Build Gate):** The Build stage must execute with `PYTHONDONTWRITEBYTECODE=1` and AST-only validation; arbitrary `eval`/`exec` or subprocess invocation is strictly prohibited until Stage 07.
* **FR-02 (DSPy Optimizer Compatibility):** Every LLM call within the pipeline must be wrapped in a `dspy.Module` with explicit `dspy.InputField` and `dspy.OutputField` annotations, logging prompts, completions, reasoning traces, and scalar reward metrics (`0.0` to `1.0`) to the external YAML/JSON ledger.
* **FR-03 (Graph Intelligence Routing):** Codebase modules are represented as nodes *V* and imports/calls as directed edges *E* in a `networkx.DiGraph`. Centrality metrics (PageRank / Betweenness) dictate audit priority and blast-radius estimation during refactoring.
* **FR-04 (Self-Contained HTML Reporting):** Generated reports in `/reports/` must require zero external CDN dependencies (inline CSS/SVG charts) so they remain viewable in air-gapped environments.
