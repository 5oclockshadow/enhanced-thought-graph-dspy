/* Graph Foundry · Universal Builder — engine (foundry-builder.js)
 *
 * One evolutionary core that, given a goal, WRITES real executable JS source
 * from a typed genome, RUNS it for real on generated cases, and evolves a
 * population toward maximum measured fitness — then compiles the lead into a
 * program-graph and emits a serialisable engine spec.
 *
 * The named libraries are MODELLED in-browser (not imported), per the PRD:
 *   dspy   — signature / compile            gym    — make (env + metric)
 *   deap   — evolve (GA operators)          di     — DataInterpreter write+run
 *   mcp    — tool wrapper                    scholar— references (computed)
 *
 * Runs in both the browser (window.FoundryBuilder) and Node (module.exports),
 * so the search is unit-testable headlessly.  Nothing here is mocked: every
 * fitness number comes from executing generated code.
 */
(function (root) {
  "use strict";

  // Deterministic RNG so rounds are reproducible and tests are stable.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : (isFinite(x) ? x : 0));
  const f3 = (x) => (Math.round(x * 1000) / 1000);

  function pearson(a, b) {
    const n = a.length; if (!n) return 0;
    let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
    const d = Math.sqrt(da * db); return d ? num / d : 0;
  }

  /* ======================= TASKS ARE DATA ============================ *
   * Each task: {name, label, keywords, genomeSize, box:[lo,hi],
   *   decode(g)->{src, fn}, cases(rng)->cases, score(fn, cases)->[0,1]} */
  const TASKS = {
    linear_classifier: {
      name: "linear_classifier", label: "Linear classifier (accuracy)",
      keywords: ["classify", "classifier", "separate", "label", "decision", "spam"],
      genomeSize: 3, box: [-5, 5],
      decode(g) {
        const body = `(${f3(g[0])})*x[0]+(${f3(g[1])})*x[1]+(${f3(g[2])})`;
        return { src: `function predict(x){\n  return (${body}) >= 0 ? 1 : 0;\n}`,
                 fn: new Function("x", `return (${body}) >= 0 ? 1 : 0;`) };
      },
      cases(rng) {
        const tw = [1.5, -2.0], tb = 0.4, pts = [];
        for (let i = 0; i < 80; i++) {
          const x = [(rng() * 6 - 3), (rng() * 6 - 3)];
          let y = (tw[0] * x[0] + tw[1] * x[1] + tb) >= 0 ? 1 : 0;
          if (rng() < 0.08) y = 1 - y; // label noise -> accuracy caps < 1
          pts.push({ x, y });
        }
        return pts;
      },
      score(fn, cases) { let ok = 0; for (const c of cases) if (fn(c.x) === c.y) ok++; return clamp01(ok / cases.length); }
    },

    symbolic_regressor: {
      name: "symbolic_regressor", label: "Symbolic regressor (fit)",
      keywords: ["regress", "fit", "curve", "predict value", "function", "approximate"],
      genomeSize: 4, box: [-3, 3],
      decode(g) {
        const body = `(${f3(g[0])}) + (${f3(g[1])})*x + (${f3(g[2])})*x*x + (${f3(g[3])})*Math.sin(x)`;
        return { src: `function f(x){\n  return ${body};\n}`, fn: new Function("x", `return ${body};`) };
      },
      cases(rng) {
        const t = [2, 0.5, -0.3, 1.0], pts = [];
        for (let i = 0; i < 60; i++) {
          const x = rng() * 6 - 3;
          const y = t[0] + t[1] * x + t[2] * x * x + t[3] * Math.sin(x) + (rng() - 0.5) * 0.3;
          pts.push({ x, y });
        }
        return pts;
      },
      score(fn, cases) { let se = 0; for (const c of cases) { const e = fn(c.x) - c.y; se += e * e; } return clamp01(1 / (1 + se / cases.length)); }
    },

    job_scheduler: {
      name: "job_scheduler", label: "Job scheduler (makespan)",
      keywords: ["schedule", "scheduler", "jobs", "machine", "makespan", "allocate"],
      genomeSize: 12, box: [0, 1],
      decode(g) {
        const order = g.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).map((z) => z[1]);
        const logic = `const o=[${order}];const load=Array(M).fill(0);for(const j of o){let m=0;for(let k=1;k<M;k++)if(load[k]<load[m])m=k;load[m]+=dur[j];}return Math.max.apply(null,load);`;
        return { src: `// assign jobs in priority order to the least-loaded machine\nfunction schedule(dur, M){\n  const order=[${order}];\n  const load=Array(M).fill(0);\n  for(const j of order){ let m=0; for(let k=1;k<M;k++) if(load[k]<load[m]) m=k; load[m]+=dur[j]; }\n  return Math.max(...load);\n}`,
                 fn: new Function("dur", "M", logic) };
      },
      cases(rng) {
        const dur = []; let sum = 0;
        for (let i = 0; i < 12; i++) { const d = 1 + Math.floor(rng() * 9); dur.push(d); sum += d; }
        const M = 3, lb = sum / M, worst = sum;
        return { dur, M, lb, worst };
      },
      score(fn, c) { const mk = fn(c.dur, c.M); return clamp01((c.worst - mk) / (c.worst - c.lb)); }
    },

    route_planner: {
      name: "route_planner", label: "Route planner (TSP length)",
      keywords: ["route", "tour", "tsp", "travel", "path", "delivery", "logistics"],
      genomeSize: 11, box: [0, 1],
      decode(g) {
        const order = g.map((k, i) => [k, i]).sort((a, b) => a[0] - b[0]).map((z) => z[1]);
        const logic = `const o=[${order}];let L=0;for(let i=0;i<o.length;i++){const a=P[o[i]],b=P[o[(i+1)%o.length]];L+=Math.hypot(a[0]-b[0],a[1]-b[1]);}return L;`;
        return { src: `// visit cities in the gene-ordered sequence, return tour length\nfunction tour(P){\n  const order=[${order}];\n  let L=0;\n  for(let i=0;i<order.length;i++){ const a=P[order[i]], b=P[order[(i+1)%order.length]]; L+=Math.hypot(a[0]-b[0],a[1]-b[1]); }\n  return L;\n}`,
                 fn: new Function("P", logic) };
      },
      cases(rng) {
        const n = 11, P = [];
        for (let i = 0; i < n; i++) P.push([rng() * 10, rng() * 10]);
        // nearest-neighbour baseline length
        const seen = [0]; let cur = 0, nn = 0;
        while (seen.length < n) {
          let bi = -1, bd = 1e9;
          for (let j = 0; j < n; j++) if (seen.indexOf(j) < 0) { const d = Math.hypot(P[cur][0] - P[j][0], P[cur][1] - P[j][1]); if (d < bd) { bd = d; bi = j; } }
          nn += bd; seen.push(bi); cur = bi;
        }
        nn += Math.hypot(P[cur][0] - P[0][0], P[cur][1] - P[0][1]);
        return { P, nn };
      },
      score(fn, c) { const L = fn(c.P); return clamp01((2 * c.nn - L) / (2 * c.nn - 0.7 * c.nn)); }
    },

    graph_ranker: {
      name: "graph_ranker", label: "Graph ranker (influence correlation)",
      keywords: ["rank", "influence", "centrality", "important", "graph", "node", "spread"],
      genomeSize: 3, box: [-2, 2],
      decode(g) {
        const logic = `return F.map(function(f){return (${f3(g[0])})*f.deg+(${f3(g[1])})*f.out+(${f3(g[2])})*f.clust;});`;
        return { src: `// score each node from structural features, then rank\nfunction rank(F){\n  return F.map(f => (${f3(g[0])})*f.deg + (${f3(g[1])})*f.out + (${f3(g[2])})*f.clust);\n}`,
                 fn: new Function("F", logic) };
      },
      cases(rng) {
        const n = 16, adj = Array.from({ length: n }, () => new Set());
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (rng() < 0.22) { adj[i].add(j); adj[j].add(i); }
        // features
        const F = [];
        for (let i = 0; i < n; i++) {
          const deg = adj[i].size; let tri = 0; const nb = [...adj[i]];
          for (let a = 0; a < nb.length; a++) for (let b = a + 1; b < nb.length; b++) if (adj[nb[a]].has(nb[b])) tri++;
          const possible = deg * (deg - 1) / 2;
          F.push({ deg, out: deg, clust: possible ? tri / possible : 0 });
        }
        // ground-truth influence via independent-cascade simulation
        const truth = [];
        for (let s = 0; s < n; s++) {
          let total = 0, trials = 20;
          for (let t = 0; t < trials; t++) {
            const active = new Set([s]), frontier = [s];
            while (frontier.length) {
              const u = frontier.pop();
              for (const v of adj[u]) if (!active.has(v) && rng() < 0.25) { active.add(v); frontier.push(v); }
            }
            total += active.size;
          }
          truth.push(total / trials);
        }
        return { F, truth };
      },
      score(fn, c) { return clamp01(pearson(fn(c.F), c.truth)); }
    }
  };

  function routeGoal(text) {
    const t = (text || "").toLowerCase();
    let best = "linear_classifier", bestN = 0;
    for (const k in TASKS) {
      let n = 0; for (const kw of TASKS[k].keywords) if (t.indexOf(kw) >= 0) n++;
      if (n > bestN) { bestN = n; best = k; }
    }
    return best;
  }

  /* ======================= DEAP-style evolution ===================== */
  function createEvolve(task, cases, opts) {
    opts = opts || {};
    const rng = mulberry32((opts.seed || 1) ^ 0x9e3779b9);
    const popSize = opts.popSize || 60, gens = opts.gens || 24, elite = opts.elite || 2;
    const cx = opts.cx != null ? opts.cx : 0.6, mut = opts.mut != null ? opts.mut : 0.3;
    const [lo, hi] = task.box, G = task.genomeSize;
    const randGene = () => lo + rng() * (hi - lo);
    const clip = (v) => (v < lo ? lo : v > hi ? hi : v);
    const fitness = (g) => { try { return task.score(task.decode(g).fn, cases); } catch (e) { return 0; } };

    let pop = Array.from({ length: popSize }, () => Array.from({ length: G }, randGene));
    let fit = pop.map(fitness);
    let evals = popSize, gen = 0;
    let bestEver = { genome: pop[0].slice(), fitness: fit[0] };

    function tournament() {
      let bi = Math.floor(rng() * popSize);
      for (let k = 0; k < 2; k++) { const c = Math.floor(rng() * popSize); if (fit[c] > fit[bi]) bi = c; }
      return pop[bi];
    }
    function snapshot() {
      const idx = fit.map((f, i) => [f, i]).sort((a, b) => b[0] - a[0]);
      const bestI = idx[0][1], mean = fit.reduce((a, b) => a + b, 0) / popSize;
      if (fit[bestI] > bestEver.fitness) bestEver = { genome: pop[bestI].slice(), fitness: fit[bestI] };
      return {
        gen, evals, bestFit: fit[bestI], mean, bestGenome: pop[bestI].slice(),
        bestEver: bestEver.fitness,
        individuals: pop.map((g, i) => ({ genome: g, fit: fit[i], best: i === bestI })),
        source: task.decode(pop[bestI]).src
      };
    }

    return {
      get done() { return gen >= gens; },
      get bestEver() { return bestEver; },
      snapshot,
      step() {
        if (gen >= gens) return snapshot();
        const idx = fit.map((f, i) => [f, i]).sort((a, b) => b[0] - a[0]);
        const next = [];
        for (let e = 0; e < elite; e++) next.push(pop[idx[e][1]].slice());
        while (next.length < popSize) {
          let child;
          const p1 = tournament(), p2 = tournament();
          if (rng() < cx) { // BLX-alpha crossover
            child = p1.map((a, i) => { const b = p2[i], d = Math.abs(a - b), mn = Math.min(a, b) - 0.3 * d, mx = Math.max(a, b) + 0.3 * d; return clip(mn + rng() * (mx - mn)); });
          } else child = p1.slice();
          for (let i = 0; i < G; i++) if (rng() < mut) child[i] = clip(child[i] + (rng() - 0.5) * (hi - lo) * 0.25); // Gaussian-ish
          next.push(child);
        }
        pop = next; fit = pop.map(fitness); evals += popSize; gen++;
        return snapshot();
      },
      runAll() { while (gen < gens) this.step(); return snapshot(); }
    };
  }

  /* ======================= modelled library surfaces =============== */
  const dspy = {
    signature(task) {
      const inputs = { linear_classifier: "x:float[2]", symbolic_regressor: "x:float", job_scheduler: "dur:int[],M:int", route_planner: "P:float[][]", graph_ranker: "F:feature[]" }[task.name];
      const outputs = { linear_classifier: "y:int", symbolic_regressor: "y:float", job_scheduler: "makespan:int", route_planner: "length:float", graph_ranker: "scores:float[]" }[task.name];
      return { module: task.name, inputs, outputs, metric: "fitness∈[0,1]" };
    },
    compile(task, best) {
      const d = task.decode(best.genome);
      return {
        nodes: [
          { id: "in", kind: "input", label: dspy.signature(task).inputs },
          { id: "decode", kind: "module", label: "decode(genome)" },
          { id: "run", kind: "module", label: "execute(src)" },
          { id: "out", kind: "output", label: dspy.signature(task).outputs }
        ],
        edges: [["in", "decode"], ["decode", "run"], ["run", "out"]],
        source: d.src, fitness: best.fitness
      };
    }
  };
  const gym = { make(task, seed) { const rng = mulberry32((seed || 1) ^ 0x1234567); return { cases: task.cases(rng), metric: task.score.bind(task) }; } };
  const di = { write(task, genome) { return task.decode(genome); } };
  const scholar = {
    search(task) {
      const refs = {
        linear_classifier: ["Rosenblatt 1958 — Perceptron", "Cortes & Vapnik 1995 — SVM"],
        symbolic_regressor: ["Koza 1992 — Genetic Programming", "Schmidt & Lipson 2009 — Distilling natural laws"],
        job_scheduler: ["Graham 1969 — LPT bounds", "Pinedo 2016 — Scheduling"],
        route_planner: ["Bellmore & Nemhauser 1968 — TSP survey", "Lin & Kernighan 1973 — TSP heuristic"],
        graph_ranker: ["Page & Brin 1998 — PageRank", "Kempe et al. 2003 — Influence maximization"]
      };
      return refs[task.name] || [];
    }
  };
  let _tid = 0;
  const mcp = { tool(name, fn) { return function () { const id = ++_tid; const out = fn.apply(null, arguments); return { tool: name, call: id, result: out }; }; } };

  /* ======================= ReAct round ============================= */
  function engineSpec(task, best, pg, sig) {
    return {
      engine: "universal-builder/" + task.name, version: "1.0",
      task: task.name, signature: sig, genome: best.genome.map(f3),
      fitness: f3(best.fitness), program_graph: pg,
      source: pg.source, created_at: new Date().toISOString()
    };
  }

  // Build the ReAct step list for a round (narration computed; a Claude
  // runtime may replace the `thought` strings when available).
  function plan(task, goal) {
    const sig = dspy.signature(task);
    return [
      { stage: "Signature", tool: "dspy.signature", thought: `Declare typed I/O for ${task.label}.`, observation: `${sig.inputs} → ${sig.outputs}` },
      { stage: "Environment", tool: "gym.make", thought: "Generate evaluation cases + scoring metric.", observation: "env ready · metric fitness∈[0,1]" },
      { stage: "Write", tool: "di.write", thought: "Emit executable JS from a candidate genome and run it.", observation: "source compiled via new Function()" },
      { stage: "Evolve", tool: "deap.evolve", thought: "Tournament + BLX crossover + mutation + elitism.", observation: "population climbing measured fitness" },
      { stage: "Compile", tool: "dspy.compile", thought: "Freeze the lead genome into a program-graph.", observation: "4-node program-graph" },
      { stage: "Ship", tool: "engineSpec", thought: "Emit a serialisable engine spec; write back as a skill.", observation: "spec ready to launch" }
    ];
  }

  // Headless full round (used by the self-test and by callers that don't
  // need per-generation animation). Returns the engine spec + telemetry.
  function runRound(taskName, opts) {
    opts = opts || {};
    const task = TASKS[taskName] || TASKS[routeGoal(taskName)];
    const seed = opts.seed || 1;
    const env = gym.make(task, seed);
    const ev = createEvolve(task, env.cases, { seed, popSize: opts.popSize, gens: opts.gens });
    const initial = ev.snapshot();
    const final = ev.runAll();
    const best = { genome: ev.bestEver.genome, fitness: ev.bestEver.fitness };
    const pg = dspy.compile(task, best);
    const spec = engineSpec(task, best, pg, dspy.signature(task));
    return {
      task: task.name, references: scholar.search(task),
      initialMean: initial.mean, initialBest: initial.bestFit,
      finalBest: best.fitness, evals: final.evals, spec, programGraph: pg
    };
  }

  root.FoundryBuilder = {
    TASKS, routeGoal, createEvolve, runRound, plan, engineSpec,
    dspy, gym, di, scholar, mcp, mulberry32, pearson
  };
  if (typeof module !== "undefined" && module.exports) module.exports = root.FoundryBuilder;
})(typeof window !== "undefined" ? window : globalThis);
