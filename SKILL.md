---
name: maximillian
description: Autonomous recursive software engineering skill that executes a strict 14-stage loop: Audit, Build (static no-exec), Read, Refactor, Audit, Refactor, Execute, Test-Till-Pass, Full Log/File Ingestion, Opportunity Triage, Plan, Update External DSPy Telemetry Ledger (YAML/JSON) & Git, Preflight & Snapshot, and HTML Round Reporting.
version: 5.2.0
tags:
  - dspy
  - graph-intelligence
  - autonomous-coding
  - test-till-pass
  - ast-audit
  - pydantic-v2
  - neuro-symbolic
---

# Maximillian Execution Protocol

When the **Maximillian** skill is invoked, you MUST operate as a deterministic, state-machine-driven principal AI systems engineer. Do not skip stages, do not merge steps casually, and never commit unverified code to Git. Execute the exact sequence below for every round.

This skill is backed by the `maximillian/` Python package in this repository; the
orchestrator is `python -m maximillian.state_machine`.

---

## Phase I: Static Analysis & Zero-Execution Synthesis

### Stage 1: Initial Audit (`AUDIT_INIT`)
* Scan the target workspace directory, file tree, AST structures, and existing logs.
* Construct a programmatic dependency graph (modules, classes, functions, Pydantic schemas, DSPy signatures) — `maximillian.graph_intel.GraphIntel`.
* Identify syntax errors, missing type annotations, circular dependencies, dead code, high cyclomatic complexity, and architectural drift.
* Output a structured **Audit Findings Table** categorizing issues by severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).

### Stage 2: Build (`BUILD_NO_EXEC`)
* **CRITICAL INVARIANT:** Generate or scaffold required code modules, Pydantic v2 models, and DSPy Signatures as pure static code. **Do NOT execute code, run scripts, or trigger side effects during this stage.**
* Enforce strict type hints, docstrings, and explicit `dspy.InputField` / `dspy.OutputField` contracts.

### Stage 3: Full-Context Read (`READ_ALL`)
* Read all newly synthesized files alongside existing core modules, configuration files, test suites, and historical runtime logs.
* Verify cross-file symbol resolution, import paths, and interface compatibility across the entire dependency graph.

---

## Phase II: Iterative Refactoring & Verification

### Stage 4: Structural Refactor (`REFACTOR_PRIMARY`)
* Refactor code for modularity, single-responsibility adherence, clean dependency injection, and optimal DSPy module composition.
* Eliminate monolithic functions and decouple I/O from pure business/graph logic.

### Stage 5: Secondary Audit (`AUDIT_POST_REFACTOR`)
* Re-audit all modified and newly built files against architectural invariants:
  * Are all Pydantic v2 validators properly typed?
  * Are all DSPy modules free of hardcoded prompt strings where structured Signatures belong?
  * Are there any unhandled exceptions or uncovered edge cases?

### Stage 6: Precision Refactor (`REFACTOR_PRECISION`)
* Apply surgical patches to resolve every finding surfaced in Stage 5. Ensure zero linter, type-checker, or static invariant warnings remain.

---

## Phase III: Execution & Bounded Cyclic Validation

### Stage 7: Sandboxed Execution (`EXEC_SANDBOX`)
* Execute the target entrypoint or module harness in a controlled environment.
* Capture complete `stdout`, `stderr`, execution latency, memory footprint, and structured trace logs.

### Stage 8: Test-Till-Pass Loop (`TEST_TILL_PASS`)
* Run the test suite (`pytest` / unit & integration assertions).
* **If ANY test fails:**
  1. Parse the exact traceback and failing assertion.
  2. Trace the fault to its root node in the dependency graph.
  3. Apply a minimal, targeted code fix.
  4. Re-run the test suite.
  5. Repeat until **100% of tests pass** (or escalate with a root-cause diagnostic if `max_iterations` is exhausted).

---

## Phase IV: Triage, Planning & Dual-Ledger Persistence

### Stage 9: Full File & Log Ingestion (`INGEST_ALL_LOGS`)
* Read all updated source files, test coverage metrics, and runtime execution logs generated during Stages 7–8.
* Evaluate system stability, latency bottlenecks, and architectural cleanliness.

### Stage 10: Opportunity Triage (`TRIAGE_ROUND`)
* Determine if there is a high-value, concrete engineering improvement ("something decent and worth tackling") remaining in scope for this cycle.
* Score candidate improvements by **Impact vs. Blast Radius**.

### Stage 11: Strategic Audit & Plan (`AUDIT_AND_PLAN`)
* If Stage 10 identifies a worthy target, conduct a focused mini-audit on that subsystem and formulate a concrete, step-by-step DAG plan for the subsequent round.
* If no high-ROI target remains, mark the convergence flag `CONVERGED = True`.

### Stage 12: Update Telemetry Ledger & Git (`SYNC_LEDGERS`)
1. **External DSPy Optimizer Ledger (YAML/JSON):**
   * Update `tracking/maximillian_ledger.yaml` (and/or `.jsonl`) — kept **outside** Git tracking — with verbose metadata required for future `dspy.teleprompt` optimization (`MIPROv2`, `BootstrapFewShot`):
     * `round_id`, `timestamp`, `dspy_signatures_used`, `prompt_traces`, `reasoning_chains`, `test_iterations_to_pass`, `ast_complexity_delta`, `token_metrics`, and scalar `reward_score` (`0.0`–`1.0`).
2. **Git Repository:**
   * Stage clean production and test code changes (`git add`).
   * Commit with a concise, descriptive conventional commit message (`git commit -m "..."`). Never commit verbose telemetry dumps to Git.

---

## Phase V: Preflight, Snapshot, Report & Repeat

### Stage 13: Preflight & Snapshot (`PREFLIGHT_SNAPSHOT`)
* Execute preflight checks: verify clean `git status`, validate all imports, confirm zero temporary/orphan files, and check disk/environment health.
* Create an immutable state snapshot (`git tag -a maximillian-r<N>-<timestamp>` or workspace archive snapshot).

### Stage 14: HTML Round Report (`REPORT_HTML`)
* Generate a comprehensive, self-contained HTML diagnostic report and save it to `reports/round_<N>_<YYYYMMDD_HHMMSS>.html`.
* The HTML report must include:
  * Executive Round Summary & Convergence Status
  * Stage-by-Stage Execution Timeline & Latencies
  * AST & Graph Intelligence Metrics (Nodes, Edges, Complexity Delta)
  * Test-Till-Pass Convergence Log (Iterations, Tracebacks Resolved)
  * DSPy Telemetry & Reward Score Breakdown
  * Git Commit SHA & Snapshot Tag Reference

### Stage 15: Loop Continuation (`REPEAT_OR_HALT`)
* If `CONVERGED == False` and `current_round < max_rounds`, immediately initiate **Round N + 1** starting back at Stage 1 (`AUDIT_INIT`) using the plan generated in Stage 11.
* Otherwise, conclude execution and present a concise summary linking to the saved HTML report and snapshot tag.
