/* Graphos workspace — vanilla JS, no build step.
   Renders a reinforced thought graph from /api/graph (or a baked dataset
   exposed as window.__GRAPHOS_DATA__ for the offline HTML preview). */
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
  function colorFor(type) { return TYPE_COLOR[type] || "#67E8F9"; }

  var $ = function (id) { return document.getElementById(id); };
  function el(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) { if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]); }
    return e;
  }
  function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }

  /* ---------- force-directed layout (spring-electrical) ---------- */
  function layout(nodes, edges, w, h, iters) {
    var i, n, m, idx = {};
    for (i = 0; i < nodes.length; i++) {
      nodes[i].x = w / 2 + (Math.random() - 0.5) * w * 0.6;
      nodes[i].y = h / 2 + (Math.random() - 0.5) * h * 0.6;
      nodes[i].vx = 0; nodes[i].vy = 0;
      idx[nodes[i].id] = nodes[i];
    }
    var k = Math.sqrt((w * h) / Math.max(1, nodes.length)) * 0.75;
    for (var it = 0; it < iters; it++) {
      var cool = 1 - it / iters;
      for (n = 0; n < nodes.length; n++) {
        var a = nodes[n]; a.fx = 0; a.fy = 0;
        for (m = 0; m < nodes.length; m++) {
          if (m === n) continue;
          var b = nodes[m];
          var dx = a.x - b.x, dy = a.y - b.y;
          var d2 = dx * dx + dy * dy || 0.01;
          var d = Math.sqrt(d2);
          var rep = (k * k) / d;
          a.fx += (dx / d) * rep; a.fy += (dy / d) * rep;
        }
      }
      for (var e = 0; e < edges.length; e++) {
        var s = idx[edges[e].source], t = idx[edges[e].target];
        if (!s || !t) continue;
        var ex = t.x - s.x, ey = t.y - s.y;
        var ed = Math.sqrt(ex * ex + ey * ey) || 0.01;
        var strength = edges[e].reinforced_weight || edges[e].strength || 0.5;
        var attr = (ed * ed) / k * (0.4 + strength);
        var ax = (ex / ed) * attr, ay = (ey / ed) * attr;
        s.fx += ax; s.fy += ay; t.fx -= ax; t.fy -= ay;
      }
      for (n = 0; n < nodes.length; n++) {
        var p = nodes[n];
        // pull toward center a touch
        p.fx += (w / 2 - p.x) * 0.02; p.fy += (h / 2 - p.y) * 0.02;
        var disp = Math.sqrt(p.fx * p.fx + p.fy * p.fy) || 0.01;
        var step = Math.min(disp, 24 * cool);
        p.x += (p.fx / disp) * step; p.y += (p.fy / disp) * step;
        p.x = Math.max(28, Math.min(w - 28, p.x));
        p.y = Math.max(28, Math.min(h - 28, p.y));
      }
    }
    return idx;
  }

  /* ---------- render graph into SVG ---------- */
  function renderGraph(data) {
    var svg = $("graph-svg");
    var rect = svg.getBoundingClientRect();
    var w = Math.max(360, rect.width || 640), h = Math.max(240, rect.height || 360);
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var defs = el("defs", {});
    defs.innerHTML = '<filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/>' +
      '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
    svg.appendChild(defs);

    var nodes = data.thoughts.map(function (t) { return { id: t.id, t: t }; });
    var idx = layout(nodes, data.connections, w, h, 180);

    var gEdges = el("g", { "stroke-width": "1.2", opacity: "0.72", filter: "url(#glow)" });
    data.connections.forEach(function (c) {
      var s = idx[c.source], t = idx[c.target];
      if (!s || !t) return;
      var line = el("line", { x1: s.x, y1: s.y, x2: t.x, y2: t.y,
        stroke: colorFor(s.t.type), "stroke-width": (0.6 + (c.reinforced_weight || c.strength || 0.5) * 2.4).toFixed(2) });
      gEdges.appendChild(line);
    });
    svg.appendChild(gEdges);

    var gNodes = el("g", { filter: "url(#glow)" });
    nodes.forEach(function (nd) {
      var r = 5 + (nd.t.confidence || 0.5) * 11;
      var c = el("circle", { cx: nd.x, cy: nd.y, r: r.toFixed(1), fill: colorFor(nd.t.type) });
      var title = el("title", {});
      title.textContent = nd.t.content + "  [" + nd.t.type + " · conf " + nd.t.confidence + "]";
      c.appendChild(title);
      gNodes.appendChild(c);
    });
    svg.appendChild(gNodes);
    $("graph-empty").style.display = "none";
  }

  /* ---------- right-rail panels ---------- */
  function renderPlan(data, query) {
    var done = '<span style="color:#34D399">✅</span>';
    var run = '<span style="color:#22D3EE">◐</span>';
    var steps = [
      [done, "Ingest & validate input", 'Pydantic · "' + esc(query) + '"'],
      [done, "Build thought graph", data.metrics.total_nodes + " nodes · " + data.metrics.total_edges + " edges"],
      [done, "Hebbian + Q-table reinforcement",
        "plasticity " + data.learning.hebbian_plasticity + " · reward " + data.learning.q_reward],
      [run, "Rank & explain clusters", (data.clusters.length) + " clusters · agent Scout"]
    ];
    var html = '<div class="sec-head"><b>Current plan</b><span class="rule"></span>' +
      '<span class="count" style="color:#34D399">3 / 4</span></div>';
    steps.forEach(function (s) {
      html += '<div class="plan-item"><span>' + s[0] + '</span><div>' +
        '<div class="plan-item__title">' + esc(s[1]) + '</div>' +
        '<div class="plan-item__meta">' + s[2] + '</div></div></div>';
    });
    $("panel-plan").innerHTML = html;
  }

  function renderTasks(data) {
    var tasks = [
      ["Scout · cluster ranking", "running", "#22D3EE", "Scoring " + data.metrics.total_nodes + " nodes by relevance", "68%"],
      ["Alchemist · feature sync", "running", "#FBBF24", "Materializing reinforced edges → graph-redis", "41%"],
      ["Guardian · PII scan", "queued", "#4C586B", "Waiting on Scout output", "0%"]
    ];
    var html = '<div class="sec-head"><b>Background agents</b><span class="rule"></span>' +
      '<span class="count" style="color:#FBBF24">2 running</span></div>';
    tasks.forEach(function (t) {
      html += '<div class="task"><div class="task__head">' +
        '<span class="dot" style="background:' + t[2] + ';box-shadow:0 0 9px ' + t[2] + '"></span>' +
        '<span class="task__name">' + esc(t[0]) + '</span>' +
        '<span style="font-size:11px;color:' + t[2] + '">' + t[1] + '</span></div>' +
        '<div class="task__detail">' + esc(t[3]) + '</div>' +
        '<div class="bar"><i style="width:' + t[4] + '"></i></div></div>';
    });
    $("panel-tasks").innerHTML = html;
  }

  function renderArtifacts(data) {
    var tints = ["rgba(168,85,247,.18)", "rgba(34,211,238,.18)", "rgba(52,211,153,.18)", "rgba(251,191,36,.18)"];
    var items = [
      ["▣", "Graph snapshot", data.metrics.total_nodes + " nodes · just now"],
      ["◈", "Risk cluster explorer", data.clusters.length + " clusters"],
      ["⛁", "Reinforcement report", "Hebbian " + data.learning.hebbian_plasticity],
      ["∿", "Phoenix trace", (data.telemetry ? data.telemetry.elapsed_seconds + "s" : "traced")]
    ];
    var html = '<div class="sec-head"><b>Artifacts</b><span class="rule"></span>' +
      '<span class="count" style="color:#A855F7">' + items.length + ' items</span></div>';
    items.forEach(function (a, i) {
      html += '<a class="artifact" href="#"><span class="artifact__icon" style="background:' + tints[i] + '">' + a[0] + '</span>' +
        '<span><span class="artifact__name">' + a[1] + '</span><br><span class="artifact__meta">' + esc(a[2]) + '</span></span></a>';
    });
    $("panel-artifacts").innerHTML = html;
  }

  function renderThread(query, data) {
    var top = (data.clusters[0] && data.clusters[0].name) || "the core cluster";
    var html = '<div class="msg msg--user"><div class="bubble">' + esc(query) + '</div></div>';
    html += '<div class="msg"><div class="msg__avatar"></div><div class="msg__body">' +
      "I built a thought graph from your prompt and reinforced its edges with a Hebbian + Q-table pass. " +
      "It has <b>" + data.metrics.total_nodes + "</b> nodes across <b>" + data.clusters.length + "</b> clusters; " +
      "the strongest is <b>" + esc(top) + "</b>." +
      '<div class="msg__chips">' +
      '<span class="outchip">⚑ Plan · 4 steps</span>' +
      '<span class="outchip">⟳ 2 agents dispatched</span>' +
      '<span class="outchip">▣ Artifact: Graph snapshot</span>' +
      '<span class="outchip">quality ' + data.metrics.quality_score + '</span>' +
      "</div></div></div>";
    $("thread").innerHTML = html;
  }

  /* ---------- orchestration ---------- */
  function render(query, data) {
    $("session-title").textContent = query;
    $("session-meta").textContent = "RAPTOR · " + data.metrics.total_nodes + " nodes · " +
      data.metrics.total_edges + " edges · quality " + data.metrics.quality_score;
    $("learn-readout").textContent = "Hebbian " + data.learning.hebbian_plasticity +
      " · Q " + data.learning.q_reward;
    $("graph-cluster").textContent = (data.clusters[0] && data.clusters[0].name) || "graph";
    renderThread(query, data);
    renderPlan(data, query);
    renderTasks(data);
    renderArtifacts(data);
    renderGraph(data);
  }

  function toast(msg) {
    var t = document.createElement("div");
    t.className = "toast"; t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(function () { t.remove(); }, 4000);
  }

  function presetToBody(presetKey, query, nodes) {
    // Minimal client-side preset → request body mapping.
    var map = {
      churn_intelligence: { generation_mode: "analytical", clustering_method: "causal", domain_focus: "business", enable_risk_analysis: true },
      creative_brainstorm: { generation_mode: "creative", clustering_method: "thematic", domain_focus: "creative", creativity_level: 0.9 },
      technical_planning: { generation_mode: "practical", clustering_method: "functional", domain_focus: "technology", min_confidence: 0.6 }
    };
    var body = { input_text: query, max_nodes: nodes };
    var p = map[presetKey] || {};
    for (var k in p) if (p.hasOwnProperty(k)) body[k] = p[k];
    return body;
  }

  function generate(query) {
    var btn = $("send"); btn.disabled = true;
    $("trace-chip").textContent = "● building…";
    var body = presetToBody($("preset").value, query, parseInt($("nodes").value, 10) || 16);
    fetch("/api/graph", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (data) {
      render(query, data);
      $("trace-chip").textContent = "● tracing → Phoenix";
    }).catch(function (err) {
      toast("Generation failed: " + err.message);
      $("trace-chip").textContent = "● trace error";
    }).finally(function () { btn.disabled = false; });
  }

  /* ---------- wire up ---------- */
  document.addEventListener("DOMContentLoaded", function () {
    // tab switching
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
      e.preventDefault();
      var q = $("composer-input").value.trim();
      if (q) generate(q);
    });
    $("btn-new").addEventListener("click", function () { $("composer-input").focus(); });

    // Offline preview: a baked dataset renders immediately without the API.
    if (window.__GRAPHOS_DATA__) {
      var baked = window.__GRAPHOS_DATA__;
      // Give the SVG a moment to get real dimensions before layout.
      setTimeout(function () { render(baked.query, baked.data); }, 30);
    }
  });
})();
