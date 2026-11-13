# SaaS Readiness — Council Audit & Hardening

A six-lens architecture/ops/security audit (Architect, Craftsman, Skeptic,
Guardian, Operator, Tuner) was run read-only, then a hardening + deepening pass
was implemented. This records both the findings and the honest status.

## Fixed in this pass

### Security (Guardian, Skeptic)
- **Code-execution endpoints gated.** `/api/foundry/propose` and
  `/api/foundry/self-improve` are **OFF by default** (`GRAPHOS_ENABLE_FOUNDRY`,
  default `0`) → `403` unless explicitly enabled in a trusted environment.
- **Subprocess hardening.** The Node task-validator now runs with a **scrubbed
  env** (no inherited secrets), its **own process group** (orphaned children are
  `SIGKILL`ed with it), and a bounded wall-clock → `504` on timeout.
- **Self-improve gate hardened.** Immutable pristine `*.orig` (written once,
  never overwritten) + timestamped rolling backups (no more self-destructing
  `.bak`); the gate now also rejects `@import`, external `url()`, `javascript:`,
  `</style`/`<script`, and `expression()` injection.
- **Optional auth + CORS.** `GRAPHOS_API_KEY` enforces `X-API-Key` on mutating
  routes when set; CORS never combines `*` with `allow_credentials`.

### Multi-tenancy (Guardian)
- **Owner scoping.** `sessions` gained an `owner_id`; create/list/get/delete are
  scoped to the caller principal (`X-Tenant-Id`/API key, else `public`). No more
  global session enumeration; cross-tenant reads return `404`.
- **Right-to-erasure.** `DELETE /api/sessions/{id}` cascades session → runs,
  graphs, artifacts.

### Reliability (Operator, Skeptic, Tuner)
- **Startup reaper** fails runs left `pending`/`running` by a prior restart, so
  pollers fail fast instead of hanging.
- **Bounded job queue** (`GRAPHOS_JOB_QUEUE_MAX`) → `503` when full; **per-job
  timeout** (`GRAPHOS_JOB_TIMEOUT`) so a hung model call can't hold a worker.
- **Health split:** `/api/health/live` (deps-free liveness), `/api/health/ready`
  and `/api/health` return **503** when the store is down (probe no longer
  mutates state).
- **Store hardening:** `PRAGMA busy_timeout`, `user_version` migrations.
- **Event loop:** the 30–150s sync `/api/graph` is now a threadpool `def`, not
  an `async` handler blocking the loop.

### Correctness / durability (Craftsman)
- **Portfolio data-loss bug fixed:** corrupt/transient reads are **quarantined**
  (never silently overwritten); writes are **atomic** (`os.replace`); ids use
  UUIDs (no `len()+1` collisions); reads hold the lock.

### Deepening (Architect, Craftsman)
- **One backend port** (`graphos/backend.py`) — single cached model detection,
  replacing 3 divergent call sites.
- **One LLM-parse helper** (`graphos/llm_parse.py`) — replacing 4 copies.
- **One ingest prelude** (`graphos/repo_ingest.py`) — de-duped across modules.
- **One run lifecycle** (`run_generation`) shared by the async worker and the
  sync endpoint (was duplicated, already divergent).

### Ops (Operator)
- **Compose** now mounts a named volume for `graphos_data/` (state survives
  restart), pins `GRAPHOS_DB_PATH` into it, adds a memory limit, and **removes
  the two unused Redis services**. Dockerfile liveness probe → `/api/health/live`;
  data dir pre-created for the non-root user.

Tests: **46 passing**, including new coverage for the self-improve gate, foundry
gating, portfolio durability, tenant isolation, erasure, and the reaper.

## Deliberately deferred (documented, not done)

These are real multi-day items; they are called out rather than faked:

1. **True sandbox for `propose`.** Even scrubbed, `validate_task.js` still
   `eval`s model JS in a full Node runtime. Before enabling in any shared
   environment it needs `isolated-vm`/a no-network container with dropped caps.
   Until then it stays flag-off.
2. **Postgres + Row-Level Security.** SQLite + a single connection behind one
   lock caps write concurrency and only gives app-layer tenant checks. The
   Store/run contract is shaped for the swap; owner scoping is the interim.
3. **Durable job broker.** In-process jobs are single-replica and lost on crash
   (the reaper cleans up rows, but work is not resumed). Move to RQ/Celery for
   HA + N replicas.
4. **Rate limiting / per-tenant quotas** on model-invoking routes.
5. **Encryption at rest + retention TTL + actor-attributed audit log** for the
   free-text (likely PII) prompts.
6. **Typed graph wire contract + single `GraphScorer`.** `quality` is still
   computed differently on the live vs deterministic path; unifying the scorer
   and returning Pydantic response models is the next deepening step.

## Config surface (ops)

| Env var | Default | Purpose |
| :-- | :-- | :-- |
| `GRAPHOS_DB_PATH` | `graphos_data/graphos.db` | SQLite location (mount a volume) |
| `GRAPHOS_API_KEY` | _(unset = open)_ | require `X-API-Key` on writes |
| `GRAPHOS_CORS_ORIGINS` | `*` | explicit allowlist in production |
| `GRAPHOS_ENABLE_FOUNDRY` | `0` | enable code-exec/self-modify endpoints |
| `GRAPHOS_JOB_QUEUE_MAX` | `64` | reject (503) beyond this many in-flight |
| `GRAPHOS_JOB_TIMEOUT` | `240` | per-job wall-clock cap (s) |
| `GRAPHOS_DISABLE_LIVE` | _(unset)_ | force deterministic engine |
