# Graphos — Graph Intelligence Workspace

A production application layer over the Enhanced Thought Graph engine. It turns
the existing DSPy reactor into a dark, neon-graph workspace that mirrors the
Claude-format left→right progression: **conversation → plan → background agents
→ artifacts**.

## What was scaffolded

```
graphos/
  __init__.py        package metadata
  telemetry.py       ISO-8601 pre/post telemetry JSON + Markdown execution log
  service.py         typed request model, generation, Hebbian + Q-table edge reinforcement
  app.py             FastAPI app: /api/graph, /api/presets, /api/health, SPA + static
graphos_app.py       entrypoint (python graphos_app.py  |  uvicorn graphos.app:app)
frontend/
  index.html         3-column workspace shell
  graphos.css        neon-graph design tokens, fully responsive
  graphos.js         force-directed SVG graph, live API wiring, plan/tasks/artifacts rail
tools/
  build_preview.py   bakes a real graph into one offline, timestamped HTML file
tests/
  test_graphos.py    13 tests: success paths, boundary edge-cases, error injection
Dockerfile           non-root production image + healthcheck
docker-compose.yml   app + graph-redis + raw-redis fallback
requirements-dev.txt  test/dev extras
```

A latent engine bug was fixed in `models.py`: `generate_fake_thought` crashed on
the `business`/`creative`/`technology` templates because several `{placeholders}`
were never supplied. Template filling is now resolver-based (`_fill_template`),
so every domain produces a graph.

## Run it

```bash
pip install -r requirements.txt            # runtime deps
python graphos_app.py                       # http://localhost:8080
```

```bash
pip install -r requirements-dev.txt         # dev + test deps
python -m pytest tests/ -q                   # 13 passing
```

```bash
docker compose up --build                   # app + redis stack on :8080
```

```bash
python tools/build_preview.py "your topic"  # writes dist/graphos_preview_<ISO>.html
```

## Observability

Every `/api/graph` call emits a **pre** (planning) and **post** (validation)
telemetry record under `telemetry/` and appends a human-readable entry to
`GRAPHOS_EXECUTION_LOG.md`. In production these sinks front Arize Phoenix over
`graph-redis`, falling back to raw Redis — the env wiring is in
`docker-compose.yml`.

## Reinforcement layer

`service._reinforce` applies a deterministic Hebbian + Q-table pass to edges:
co-activation (node degree) raises each edge's reinforced weight, and a
discounted one-step Q-value propagates downstream confidence back along the
edge. The aggregate `hebbian_plasticity` and `q_reward` drive the UI readout and
edge thickness — real computed values, not decoration.
