/* Graphos workspace — persistent, async, resilient.
   Live server: bootstraps a session, enqueues generation as a background run,
   polls the run to completion, renders the stored graph, and lists real
   artifacts from the store. Offline preview: renders window.__GRAPHOS_DATA__. */
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var TYPE_COLOR = {
    core_concept: "#A855F7", sub_concept: "#A855F7", synthesis: "#A855F7",
    question: "#22D3EE", hypothesis: "#22D3EE", association: "#22D3EE",
    solution: "#34D399", opportunity: "#34D399", alternative: "#34D399",
    challenge: "#F472B6", contradiction: "#F472B6", critique: "#F472B6",
    insight: "#FBBF24", pattern: "#FBBF24", trend: "#FBBF24", cause: "#FBBF24", effect: "#FBBF24"
  };
  function colorFor(t) { return TYPE_COLOR[t] || "#67E8F9"; }
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, a) { var e = document.createElementNS(SVGNS, tag); for (var k in a) if (a.hasOwnProperty(k)) e.setAttribute(k, a[k]); return e; }
  function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }

  var state = { sessionId: null, busy: false };

  /* ---------- force-directed layout ---------- */
  function layout(nodes, edges, w, h, iters) {
    var idx = {}, i, n, m;
    for (i = 0; i < nodes.length; i++) {
      nodes[i].x = w / 2 + (Math.random() - 0.5) * w * 0.6;
      nodes[i].y = h / 2 + (Math.random() - 0.5) * h * 0.6;
      idx[nodes[i].id] = nodes[i];
    }
    var k = Math.sqrt((w * h) / Math.max(1, nodes.length)) * 0.75;
    for (var it = 0; it < iters; it++) {
      var cool = 1 - it / iters;
      for (n = 0; n < nodes.length; n++) {
        var a = nodes[n]; a.fx = 0; a.fy = 0;
        for (m = 0; m < nodes.length; m++) {
          if (m === n) continue;
          var b = nodes[m], dx = a.x - b.x, dy = a.y - b.y, d = Math.sqrt(dx * dx + dy * dy) || 0.01;
          var rep = (k * k) / d; a.fx += (dx / d) * rep; a.fy += (dy / d) * rep;
        }
      }
      for (var e = 0; e < edges.length; e++) {
        var s = idx[edges[e].source], t = idx[edges[e].target]; if (!s || !t) continue;
        var ex = t.x - s.x, ey = t.y - s.y, ed = Math.sqrt(ex * ex + ey * ey) || 0.01;
        var st = edges[e].reinforced_weight || edges[e].strength || 0.5;
        var attr = (ed * ed) / k * (0.4 + st), ax = (ex / ed) * attr, ay = (ey / ed) * attr;
        s.fx += ax; s.fy += ay; t.fx -= ax; t.fy -= ay;
      }
      for (n = 0; n < nodes.length; n++) {
        var p = nodes[n]; p.fx += (w / 2 - p.x) * 0.02; p.fy += (h / 2 - p.y) * 0.02;
        var disp = Math.sqrt(p.fx * p.fx + p.fy * p.fy) || 0.01, step = Math.min(disp, 24 * cool);
        p.x = Math.max(28, Math.min(w - 28, p.x + (p.fx / disp) * step));
        p.y = Math.max(28, Math.min(h - 28, p.y + (p.fy / disp) * step));
      }
    }
    return idx;
  }

  function renderGraph(data) {
    var svg = $("graph-svg"), rect = svg.getBoundingClientRect();
    var w = Math.max(360, rect.width || 640), h = Math.max(240, rect.height || 360);
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var defs = el("defs", {});
    defs.innerHTML = '<filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
    svg.appendChild(defs);
    var nodes = data.thoughts.map(function (t) { return { id: t.id, t: t }; });
    var idx = layout(nodes, data.connections, w, h, 180);
    var ge = el("g", { "stroke-width": "1.2", opacity: "0.72", filter: "url(#glow)" });
    data.connections.forEach(function (c) {
      var s = idx[c.source], t = idx[c.target]; if (!s || !t) return;
      ge.appendChild(el("line", { x1: s.x, y1: s.y, x2: t.x, y2: t.y, stroke: colorFor(s.t.type),
        "stroke-width": (0.6 + (c.reinforced_weight || c.strength || 0.5) * 2.4).toFixed(2) }));
    });
    svg.appendChild(ge);
    var gn = el("g", { filter: "url(#glow)" });
    nodes.forEach(function (nd) {
      var r = 5 + (nd.t.confidence || 0.5) * 11;
      var c = el("circle", { cx: nd.x, cy: nd.y, r: r.toFixed(1), fill: colorFor(nd.t.type) });
      var ti = el("title", {}); ti.textContent = nd.t.content + "  [" + nd.t.type + " · conf " + nd.t.confidence + "]";
      c.appendChild(ti); gn.appendChild(c);
    });
    svg.appendChild(gn);
    $("graph-empty").style.display = "none";
  }

  /* ---------- panels ---------- */
  function renderPlan(d, query) {
    var done = '<span style="color:#34D399">✅</span>', run = '<span style="color:#22D3EE">◐</span>';
    var steps = [
      [done, "Ingest & validate input", 'Pydantic · "' + esc(query) + '"'],
      [done, "Build graph (" + esc(d.engine || "engine") + ")", d.metrics.total_nodes + " nodes · " + d.metrics.total_edges + " edges"],
      [done, "Hebbian + Q-table reinforcement", "plasticity " + d.learning.hebbian_plasticity + " · reward " + d.learning.q_reward],
      [run, "Rank & explain clusters", d.clusters.length + " clusters"]
    ];
    var html = '<div class="sec-head"><b>Current plan</b><span class="rule"></span><span class="count" style="color:#34D399">3 / 4</span></div>';
    steps.forEach(function (s) { html += '<div class="plan-item"><span>' + s[0] + '</span><div><div class="plan-item__title">' + esc(s[1]) + '</div><div class="plan-item__meta">' + s[2] + '</div></div></div>'; });
    $("panel-plan").innerHTML = html;
  }
  function renderTasks(d) {
    var tasks = [
      ["Scout · cluster ranking", "done", "#34D399", "Ranked " + d.metrics.total_nodes + " nodes by relevance", "100%"],
      ["Alchemist · reinforcement", "done", "#34D399", "Hebbian " + d.learning.hebbian_plasticity + " · Q " + d.learning.q_reward, "100%"],
      ["Guardian · validation", "done", "#34D399", "All scores in range · quality " + d.metrics.quality_score, "100%"]
    ];
    var html = '<div class="sec-head"><b>Background agents</b><span class="rule"></span><span class="count" style="color:#34D399">complete</span></div>';
    tasks.forEach(function (t) {
      html += '<div class="task"><div class="task__head"><span class="dot" style="background:' + t[2] + ';box-shadow:0 0 9px ' + t[2] + '"></span><span class="task__name">' + esc(t[0]) + '</span><span style="font-size:11px;color:' + t[2] + '">' + t[1] + '</span></div><div class="task__detail">' + esc(t[3]) + '</div><div class="bar"><i style="width:' + t[4] + '"></i></div></div>';
    });
    $("panel-tasks").innerHTML = html;
  }
  function renderArtifactsList(items) {
    var tints = ["rgba(168,85,247,.18)", "rgba(34,211,238,.18)", "rgba(52,211,153,.18)", "rgba(251,191,36,.18)"];
    var icons = { graph: "▣", doc: "⛁", telemetry: "∿" };
    var html = '<div class="sec-head"><b>Artifacts</b><span class="rule"></span><span class="count" style="color:#A855F7">' + items.length + ' items</span></div>';
    if (!items.length) html += '<div style="font-size:12px;color:#5E6B7E">No artifacts yet — generate a graph.</div>';
    items.forEach(function (a, i) {
      html += '<a class="artifact" href="#"><span class="artifact__icon" style="background:' + tints[i % 4] + '">' + (icons[a.kind] || "▣") + '</span><span><span class="artifact__name">' + esc(a.name) + '</span><br><span class="artifact__meta">' + esc(a.kind) + ' · ' + esc((a.created_at || "").slice(0, 16).replace("T", " ")) + '</span></span></a>';
    });
    $("panel-artifacts").innerHTML = html;
  }
  function renderThread(query, d) {
    var top = (d.clusters[0] && d.clusters[0].name) || "the core cluster";
    var html = '<div class="msg msg--user"><div class="bubble">' + esc(query) + '</div></div>';
    html += '<div class="msg"><div class="msg__avatar"></div><div class="msg__body">' +
      "Built a graph via <b>" + esc(d.engine || "engine") + "</b> with Hebbian + Q-table reinforcement. " +
      "<b>" + d.metrics.total_nodes + "</b> nodes across <b>" + d.clusters.length + "</b> clusters; strongest is <b>" + esc(top) + "</b>." +
      '<div class="msg__chips"><span class="outchip">⚑ Plan · 4 steps</span><span class="outchip">▣ Artifact saved</span><span class="outchip">quality ' + d.metrics.quality_score + '</span><span class="outchip">engine: ' + esc(d.engine || "?") + '</span></div></div></div>';
    $("thread").innerHTML = html;
  }

  function render(query, d) {
    $("session-title").textContent = query;
    $("session-meta").textContent = (d.engine || "engine") + " · " + d.metrics.total_nodes + " nodes · " +
      d.metrics.total_edges + " edges · quality " + d.metrics.quality_score;
    $("learn-readout").textContent = "Hebbian " + d.learning.hebbian_plasticity + " · Q " + d.learning.q_reward;
    $("graph-cluster").textContent = (d.clusters[0] && d.clusters[0].name) || "graph";
    renderThread(query, d); renderPlan(d, query); renderTasks(d); renderGraph(d);
  }

  function toast(msg, bad) {
    var t = document.createElement("div"); t.className = "toast"; t.textContent = msg;
    if (!bad) t.style.borderColor = "#34D39955", t.style.color = "#6EE7B7";
    document.body.appendChild(t); setTimeout(function () { t.remove(); }, 4500);
  }
  function setBusy(on, label) {
    state.busy = on; $("send").disabled = on;
    $("trace-chip").textContent = on ? ("● " + (label || "working…")) : "● tracing → Phoenix";
    $("graph-empty").style.display = on ? "grid" : $("graph-empty").style.display;
    if (on) $("graph-empty").textContent = label || "Generating…";
  }

  /* ---------- API ---------- */
  function api(path, opts) {
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) throw new Error((body.error && body.error.message) || ("HTTP " + r.status));
        return body;
      });
    });
  }
  function presetBody(query, nodes) {
    var map = {
      churn_intelligence: { generation_mode: "analytical", clustering_method: "causal", domain_focus: "business", enable_risk_analysis: true },
      creative_brainstorm: { generation_mode: "creative", clustering_method: "thematic", domain_focus: "creative", creativity_level: 0.9 },
      technical_planning: { generation_mode: "practical", clustering_method: "functional", domain_focus: "technology", min_confidence: 0.6 }
    };
    var body = { input_text: query, max_nodes: nodes }, p = map[$("preset").value] || {};
    for (var k in p) if (p.hasOwnProperty(k)) body[k] = p[k];
    return body;
  }
  function ensureSession() {
    if (state.sessionId) return Promise.resolve(state.sessionId);
    return api("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "Workspace" }) }).then(function (s) { state.sessionId = s.id; return s.id; });
  }
  function refreshArtifacts() {
    if (!state.sessionId) return;
    api("/api/sessions/" + state.sessionId + "/artifacts").then(function (r) {
      renderArtifactsList(r.artifacts || []);
    }).catch(function () {});
  }
  function pollRun(runId, query, deadline) {
    return api("/api/runs/" + runId).then(function (run) {
      if (run.status === "succeeded") {
        return api("/api/graphs/" + run.graph_id).then(function (g) {
          var d = g.payload; setBusy(false);
          render(query, d); refreshArtifacts();
          toast("Graph ready · " + (d.engine || "engine") + " · " + (run.latency_ms || "?") + "ms");
        });
      }
      if (run.status === "failed") { setBusy(false); throw new Error(run.error || "generation failed"); }
      if (Date.now() > deadline) { setBusy(false); throw new Error("timed out waiting for the model"); }
      return new Promise(function (res) { setTimeout(res, 1200); }).then(function () { return pollRun(runId, query, deadline); });
    });
  }
  function generate(query) {
    if (state.busy) return;
    setBusy(true, "Enqueuing…");
    var nodes = parseInt($("nodes").value, 10) || 16;
    ensureSession().then(function (sid) {
      setBusy(true, "Generating graph…");
      return api("/api/sessions/" + sid + "/graph", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify(presetBody(query, nodes)) });
    }).then(function (job) {
      return pollRun(job.run_id, query, Date.now() + 200000);
    }).catch(function (err) {
      setBusy(false); toast("Failed: " + err.message, true);
      $("trace-chip").textContent = "● error";
    });
  }

  /* ---------- wire up ---------- */
  document.addEventListener("DOMContentLoaded", function () {
    var tabs = document.querySelectorAll(".tab");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (t) { t.classList.remove("is-active"); });
        tab.classList.add("is-active");
        ["plan", "tasks", "artifacts"].forEach(function (name) {
          $("panel-" + name).classList.toggle("is-hidden", name !== tab.dataset.tab);
        });
      });
    });
    $("composer").addEventListener("submit", function (e) {
      e.preventDefault(); var q = $("composer-input").value.trim(); if (q) generate(q);
    });
    $("btn-new").addEventListener("click", function () {
      state.sessionId = null; $("composer-input").value = ""; $("composer-input").focus();
      $("thread").innerHTML = ""; toast("New session");
    });

    if (window.__GRAPHOS_DATA__) {
      var baked = window.__GRAPHOS_DATA__;
      setTimeout(function () { render(baked.query, baked.data); }, 30);
      renderArtifactsList([{ kind: "graph", name: "Baked snapshot", created_at: "" }]);
    } else {
      refreshArtifacts();
    }
  });
})();
