/* Graphos — multi-view graph-intelligence workspace (vanilla, no build step).
   Interactive animated graph (drag / zoom / pan / inspect), client-side router,
   and views wired to the persistent async backend. Offline preview renders
   window.__GRAPHOS_DATA__ without a server. */
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
  var LEGEND = [["Concept", "#A855F7"], ["Question", "#22D3EE"], ["Solution", "#34D399"], ["Risk", "#F472B6"], ["Signal", "#FBBF24"]];
  function colorFor(t) { return TYPE_COLOR[t] || "#67E8F9"; }
  function $(s, r) { return (r || document).querySelector(s); }
  function svg(tag, a) { var e = document.createElementNS(SVGNS, tag); for (var k in a) if (a.hasOwnProperty(k)) e.setAttribute(k, a[k]); return e; }
  function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
  function fmtTime(s) { return (s || "").slice(0, 19).replace("T", " "); }

  var state = { sessionId: null, sessions: [], graph: null, view: "workspace", busy: false, offline: !!window.__GRAPHOS_DATA__ };

  /* =================== interactive graph ==================== */
  function GraphCanvas(root) {
    this.root = root;
    this.svg = $("svg", root); this.vp = $(".vp", root);
    this.inspector = $(".inspector", root); this.empty = $(".gc__empty", root);
    this.k = 1; this.tx = 0; this.ty = 0; this.alpha = 0; this.raf = null;
    this.nodes = []; this.edges = []; this.idx = {}; this.lineEls = []; this.nodeEls = [];
    this.drag = null; this.moved = 0;
    this._bind();
  }
  GraphCanvas.prototype.size = function () {
    var r = this.svg.getBoundingClientRect();
    return { w: Math.max(320, r.width || 640), h: Math.max(240, r.height || 420) };
  };
  GraphCanvas.prototype.setData = function (data) {
    var self = this, sz = this.size();
    while (this.vp.firstChild) this.vp.removeChild(this.vp.firstChild);
    this.nodes = (data.thoughts || []).map(function (t, i) {
      var ang = (i / Math.max(1, data.thoughts.length)) * Math.PI * 2;
      return { id: t.id, t: t, x: sz.w / 2 + Math.cos(ang) * sz.w * 0.28,
        y: sz.h / 2 + Math.sin(ang) * sz.h * 0.28, vx: 0, vy: 0, fixed: false };
    });
    this.idx = {}; this.nodes.forEach(function (n) { self.idx[n.id] = n; });
    this.edges = (data.connections || []).filter(function (c) { return self.idx[c.source] && self.idx[c.target]; });
    var defs = svg("defs", {});
    defs.innerHTML = '<filter id="glow"><feGaussianBlur stdDeviation="2.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
    this.vp.appendChild(defs);
    var gE = svg("g", { "stroke-width": "1.2", opacity: "0.6", filter: "url(#glow)" });
    this.lineEls = this.edges.map(function (c) {
      var ln = svg("line", { stroke: colorFor(self.idx[c.source].t.type),
        "stroke-width": (0.6 + (c.reinforced_weight || c.strength || .5) * 2.6).toFixed(2) });
      gE.appendChild(ln); return { el: ln, c: c };
    });
    this.vp.appendChild(gE);
    var gN = svg("g", { filter: "url(#glow)" });
    this.nodeEls = this.nodes.map(function (n) {
      var r = 5 + (n.t.confidence || .5) * 12;
      var c = svg("circle", { r: r.toFixed(1), fill: colorFor(n.t.type), "data-id": n.id,
        stroke: "#070A10", "stroke-width": "1.5", style: "cursor:pointer" });
      var ti = svg("title", {}); ti.textContent = n.t.content + " · " + n.t.type + " · conf " + n.t.confidence;
      c.appendChild(ti); gN.appendChild(c); n.r = r; return { el: c, n: n };
    });
    this.vp.appendChild(gN);
    this.empty.style.display = "none";
    this.k = 1; this.tx = 0; this.ty = 0;
    this.alpha = 0.9; this._loop();
    clearTimeout(this._fitT);
    this._fitT = setTimeout(function () { self.fit(); }, 1500);
  };
  GraphCanvas.prototype.fit = function () {
    if (!this.nodes.length) return;
    var xs = this.nodes.map(function (n) { return n.x; }), ys = this.nodes.map(function (n) { return n.y; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var sz = this.size(), pad = 70;
    var gw = (maxX - minX) || 1, gh = (maxY - minY) || 1;
    var k = Math.max(0.3, Math.min(1.6, Math.min((sz.w - pad) / gw, (sz.h - pad) / gh)));
    this.k = k;
    this.tx = (sz.w - (minX + maxX) * k) / 2;
    this.ty = (sz.h - (minY + maxY) * k) / 2;
    this._draw();
  };
  GraphCanvas.prototype._loop = function () {
    var self = this;
    cancelAnimationFrame(this.raf);
    var sz = this.size(), K = Math.sqrt((sz.w * sz.h) / Math.max(1, this.nodes.length)) * 0.72;
    function tick() {
      if (self.alpha > 0.004) {
        var i, j, n, m;
        for (i = 0; i < self.nodes.length; i++) { n = self.nodes[i]; n._fx = 0; n._fy = 0; }
        for (i = 0; i < self.nodes.length; i++) {
          n = self.nodes[i];
          for (j = i + 1; j < self.nodes.length; j++) {
            m = self.nodes[j]; var dx = n.x - m.x, dy = n.y - m.y, d = Math.sqrt(dx * dx + dy * dy) || .01;
            var rep = (K * K) / d / d * 18; var ux = dx / d * rep, uy = dy / d * rep;
            n._fx += ux; n._fy += uy; m._fx -= ux; m._fy -= uy;
          }
        }
        self.edges.forEach(function (c) {
          var s = self.idx[c.source], t = self.idx[c.target];
          var dx = t.x - s.x, dy = t.y - s.y, d = Math.sqrt(dx * dx + dy * dy) || .01;
          var st = c.reinforced_weight || c.strength || .5, f = (d - K * 0.9) * 0.02 * (0.5 + st);
          var ux = dx / d * f, uy = dy / d * f; s._fx += ux; s._fy += uy; t._fx -= ux; t._fy -= uy;
        });
        for (i = 0; i < self.nodes.length; i++) {
          n = self.nodes[i]; if (n.fixed) continue;
          n._fx += (sz.w / 2 - n.x) * 0.012; n._fy += (sz.h / 2 - n.y) * 0.012;
          n.vx = (n.vx + n._fx) * 0.82; n.vy = (n.vy + n._fy) * 0.82;
          n.x += Math.max(-20, Math.min(20, n.vx)); n.y += Math.max(-20, Math.min(20, n.vy));
        }
        self.alpha *= 0.985;
      }
      self._draw();
      self.raf = requestAnimationFrame(tick);
    }
    this.raf = requestAnimationFrame(tick);
  };
  GraphCanvas.prototype._draw = function () {
    this.vp.setAttribute("transform", "translate(" + this.tx + "," + this.ty + ") scale(" + this.k + ")");
    for (var i = 0; i < this.lineEls.length; i++) {
      var o = this.lineEls[i], s = this.idx[o.c.source], t = this.idx[o.c.target];
      o.el.setAttribute("x1", s.x); o.el.setAttribute("y1", s.y);
      o.el.setAttribute("x2", t.x); o.el.setAttribute("y2", t.y);
    }
    for (i = 0; i < this.nodeEls.length; i++) {
      var ne = this.nodeEls[i]; ne.el.setAttribute("cx", ne.n.x); ne.el.setAttribute("cy", ne.n.y);
    }
  };
  GraphCanvas.prototype._pt = function (ev) {
    var r = this.svg.getBoundingClientRect();
    return { px: ev.clientX - r.left, py: ev.clientY - r.top };
  };
  GraphCanvas.prototype._graphPt = function (p) { return { x: (p.px - this.tx) / this.k, y: (p.py - this.ty) / this.k }; };
  GraphCanvas.prototype._hit = function (p) {
    var g = this._graphPt(p), best = null, bd = 1e9;
    for (var i = 0; i < this.nodes.length; i++) {
      var n = this.nodes[i], dx = n.x - g.x, dy = n.y - g.y, d = Math.sqrt(dx * dx + dy * dy);
      if (d < (n.r + 8) && d < bd) { bd = d; best = n; }
    }
    return best;
  };
  GraphCanvas.prototype._bind = function () {
    var self = this;
    this.svg.addEventListener("pointerdown", function (ev) {
      self.svg.setPointerCapture(ev.pointerId); self.moved = 0;
      var p = self._pt(ev), hit = self._hit(p);
      self.drag = { hit: hit, lastpx: p.px, lastpy: p.py, startpx: p.px, startpy: p.py };
      if (hit) { hit.fixed = true; }
      self.svg.classList.add("dragging");
    });
    this.svg.addEventListener("pointermove", function (ev) {
      if (!self.drag) return;
      var p = self._pt(ev), ddx = p.px - self.drag.lastpx, ddy = p.py - self.drag.lastpy;
      self.moved += Math.abs(ddx) + Math.abs(ddy);
      if (self.drag.hit) {
        var g = self._graphPt(p); self.drag.hit.x = g.x; self.drag.hit.y = g.y;
        self.alpha = Math.max(self.alpha, 0.25);
      } else { self.tx += ddx; self.ty += ddy; self._draw(); }
      self.drag.lastpx = p.px; self.drag.lastpy = p.py;
    });
    this.svg.addEventListener("pointerup", function (ev) {
      self.svg.classList.remove("dragging");
      if (self.drag && self.drag.hit) {
        if (self.moved < 5) self._select(self.drag.hit);
        else self.drag.hit.fixed = false;
      }
      self.drag = null;
    });
    this.svg.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var p = self._pt(ev), f = ev.deltaY < 0 ? 1.12 : 0.89, nk = Math.max(0.3, Math.min(3, self.k * f));
      self.tx = p.px - (p.px - self.tx) * (nk / self.k); self.ty = p.py - (p.py - self.ty) * (nk / self.k);
      self.k = nk; self._draw();
    }, { passive: false });
    var close = $(".inspector__x", this.root);
    if (close) close.addEventListener("click", function () { self.inspector.classList.remove("show"); });
  };
  GraphCanvas.prototype._select = function (n) {
    var t = n.t;
    this.inspector.innerHTML = '<button class="inspector__x" aria-label="Close">×</button>' +
      '<div class="inspector__type" style="color:' + colorFor(t.type) + '">' + esc(t.type.replace(/_/g, " ")) + '</div>' +
      '<div class="inspector__c">' + esc(t.content) + '</div>' +
      metricRow("confidence", t.confidence) + metricRow("relevance", t.relevance) +
      metricRow("novelty", t.novelty) + metricRow("practicality", t.practicality);
    var self = this;
    $(".inspector__x", this.root).addEventListener("click", function () { self.inspector.classList.remove("show"); });
    this.inspector.classList.add("show");
  };
  GraphCanvas.prototype.destroy = function () { cancelAnimationFrame(this.raf); };
  function metricRow(k, v) { return '<div class="metric"><span>' + k + '</span><b>' + (v == null ? "—" : v) + '</b></div>'; }

  function gcMarkup(chips) {
    return '<div class="gc"><svg><g class="vp"></g></svg>' +
      '<div class="gc__bar">' + (chips || "") + '</div>' +
      '<div class="gc__legend">' + LEGEND.map(function (l) { return '<span class="lg"><i style="background:' + l[1] + '"></i>' + l[0] + '</span>'; }).join("") + '</div>' +
      '<div class="inspector"></div>' +
      '<div class="gc__empty">Generate a graph to explore it here.</div></div>';
  }

  /* =================== API ==================== */
  function api(path, opts) {
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (b) {
        if (!r.ok) throw new Error((b.error && b.error.message) || ("HTTP " + r.status));
        return b;
      });
    });
  }
  function toast(msg) { var t = document.createElement("div"); t.className = "toast"; t.textContent = msg; document.body.appendChild(t); setTimeout(function () { t.remove(); }, 4000); }

  /* =================== views ==================== */
  var canvas = null;
  function setTitle(t, s) { $("#view-title").textContent = t; $("#view-sub").textContent = s || ""; }

  function renderWorkspace() {
    setTitle("Workspace", "chat · live graph · plan → agents → artifacts");
    var g = state.graph;
    $("#view").innerHTML =
      '<div class="ws"><div class="ws__chat">' +
        '<div class="thread" id="thread"></div>' +
        '<form class="composer" id="composer"><div class="composer__box">' +
          '<input class="composer__input" id="ci" autocomplete="off" placeholder="Ask Graphos — e.g. why enterprise accounts churn after a champion leaves">' +
          '<div class="composer__row">' +
            '<label class="pill">Preset<select id="preset"><option value="churn_intelligence">Churn</option><option value="creative_brainstorm">Brainstorm</option><option value="technical_planning">Technical</option></select></label>' +
            '<label class="pill">Nodes<input id="nodes" type="number" min="5" max="40" value="16"></label>' +
            '<span class="grow"></span>' +
            '<button class="send" id="send" aria-label="Generate"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#070A10" stroke-width="2.4"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button>' +
          '</div></div></form></div>' +
        '<div class="ws__graph">' + gcMarkup('<span class="gc__chip" id="gchip">no graph yet</span>') + '</div>' +
        '<aside class="rail"><div class="tabs"><button class="tab is-active" data-t="plan">Plan</button><button class="tab" data-t="tasks">Agents</button><button class="tab" data-t="artifacts">Artifacts</button></div>' +
          '<div class="panel" id="p-plan"></div><div class="panel is-hidden" id="p-tasks"></div><div class="panel is-hidden" id="p-artifacts"></div></aside></div>';
    canvas = new GraphCanvas($(".gc", $("#view")));
    wireTabs(); wireComposer();
    if (g) { paintWorkspace(g); } else { renderEmptyChat(); refreshArtifactsPanel(); }
  }
  function renderEmptyChat() {
    $("#thread").innerHTML = '<div class="empty-chat"><h2>Build a reasoning graph</h2>' +
      '<div>Pick a question. Graphos builds a typed, reinforced graph you can explore, with a plan, agents and saved artifacts.</div>' +
      '<div class="suggests">' + ["why enterprise accounts churn after a champion leaves", "detecting coordinated fraud across accounts", "root causes of a cascading production outage"].map(function (s) { return '<button class="suggest" type="button">' + esc(s) + '</button>'; }).join("") + '</div></div>';
    Array.prototype.forEach.call(document.querySelectorAll(".suggest"), function (b) {
      b.addEventListener("click", function () { $("#ci").value = b.textContent; generate(b.textContent); });
    });
  }
  function paintWorkspace(g) {
    var top = (g.clusters[0] && g.clusters[0].name) || "the core cluster";
    $("#thread").innerHTML = '<div class="msg msg--user"><div class="bubble">' + esc(g._query || "graph") + '</div></div>' +
      '<div class="msg"><div class="msg__av"></div><div class="msg__body">Built via <b>' + esc(g.engine || "engine") + '</b> with Hebbian + Q-table reinforcement. <b>' + g.metrics.total_nodes + '</b> nodes across <b>' + g.clusters.length + '</b> clusters; strongest is <b>' + esc(top) + '</b>.' +
      '<div class="chips"><span class="chip">⚑ Plan · 4 steps</span><span class="chip">▣ Artifact saved</span><span class="chip">quality ' + g.metrics.quality_score + '</span><span class="chip">engine: ' + esc(g.engine || "?") + '</span></div></div></div>';
    $("#gchip").textContent = g.metrics.total_nodes + " nodes · " + g.metrics.total_edges + " edges · " + (g.engine || "");
    canvas.setData(g);
    paintPlan(g); paintTasks(g); refreshArtifactsPanel();
  }
  function paintPlan(g) {
    var rows = [["✅", "Ingest & validate", 'Pydantic · "' + esc(g._query || "") + '"'],
      ["✅", "Build graph (" + esc(g.engine || "") + ")", g.metrics.total_nodes + " nodes · " + g.metrics.total_edges + " edges"],
      ["✅", "Hebbian + Q-table reinforcement", "plasticity " + g.learning.hebbian_plasticity + " · reward " + g.learning.q_reward],
      ["◐", "Rank & explain clusters", g.clusters.length + " clusters"]];
    $("#p-plan").innerHTML = '<div class="sec-head"><b>Current plan</b><span class="rule"></span></div>' +
      rows.map(function (r) { return '<div class="plan-item"><span>' + r[0] + '</span><div><div class="plan-item__t">' + esc(r[1]) + '</div><div class="plan-item__m">' + r[2] + '</div></div></div>'; }).join("");
  }
  function paintTasks(g) {
    var t = [["Scout · cluster ranking", "Ranked " + g.metrics.total_nodes + " nodes by relevance"],
      ["Alchemist · reinforcement", "Hebbian " + g.learning.hebbian_plasticity + " · Q " + g.learning.q_reward],
      ["Guardian · validation", "All scores valid · quality " + g.metrics.quality_score]];
    $("#p-tasks").innerHTML = '<div class="sec-head"><b>Background agents</b><span class="rule"></span></div>' +
      t.map(function (x) { return '<div class="task"><div class="task__h"><span class="dot ok"></span><span class="task__n">' + esc(x[0]) + '</span><span style="font-size:11px;color:#34D399">done</span></div><div class="plan-item__m">' + esc(x[1]) + '</div><div class="bar"><i style="width:100%"></i></div></div>'; }).join("");
  }
  function refreshArtifactsPanel() {
    var host = $("#p-artifacts"); if (!host) return;
    if (state.offline || !state.sessionId) { host.innerHTML = artifactList([{ kind: "graph", name: "Current graph" }]); return; }
    api("/api/sessions/" + state.sessionId + "/artifacts").then(function (r) { host.innerHTML = artifactList(r.artifacts || []); }).catch(function () {});
  }
  function artifactList(items) {
    var tints = ["rgba(168,85,247,.18)", "rgba(34,211,238,.18)", "rgba(52,211,153,.18)", "rgba(251,191,36,.18)"];
    return '<div class="sec-head"><b>Artifacts</b><span class="rule"></span><span style="font-size:11px;color:#A855F7">' + items.length + '</span></div>' +
      (items.length ? items.map(function (a, i) { return '<div class="art"><span class="art__ic" style="background:' + tints[i % 4] + '">▣</span><div><div class="art__n">' + esc(a.name) + '</div><div class="art__m">' + esc(a.kind) + ' · ' + esc(fmtTime(a.created_at)) + '</div></div></div>'; }).join("") : '<div style="color:#5E6B7E;font-size:12px">No artifacts yet.</div>');
  }
  function wireTabs() {
    var tabs = document.querySelectorAll(".tab");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (t) { t.classList.remove("is-active"); }); tab.classList.add("is-active");
        ["plan", "tasks", "artifacts"].forEach(function (n) { $("#p-" + n).classList.toggle("is-hidden", n !== tab.dataset.t); });
      });
    });
  }
  function wireComposer() {
    $("#composer").addEventListener("submit", function (e) { e.preventDefault(); var q = $("#ci").value.trim(); if (q) generate(q); });
  }

  function renderExplorer() {
    setTitle("Graph Explorer", "drag nodes · scroll to zoom · click a node to inspect");
    if (!state.graph) { $("#view").innerHTML = '<div class="gc"><div class="gc__empty">No graph yet — generate one in the Workspace.</div></div>'; return; }
    $("#view").innerHTML = '<div style="height:100%">' + gcMarkup(
      '<span class="gc__chip">' + esc(state.graph._query || "graph") + '</span>' +
      '<span class="gc__chip">' + state.graph.metrics.total_nodes + ' nodes · ' + state.graph.metrics.total_edges + ' edges</span>' +
      '<span class="gc__chip">engine: ' + esc(state.graph.engine || "?") + '</span>') + '</div>';
    canvas = new GraphCanvas($(".gc", $("#view")));
    canvas.setData(state.graph);
  }

  function renderRuns() {
    setTitle("Runs & Telemetry", "every generation, logged — the audit trail");
    if (state.offline) { $("#view").innerHTML = '<div class="page"><p style="color:#5E6B7E">Run history is available when connected to the server.</p></div>'; return; }
    $("#view").innerHTML = '<div class="page"><div id="runs-body">Loading…</div></div>';
    if (!state.sessionId) { $("#runs-body").innerHTML = "No session."; return; }
    api("/api/sessions/" + state.sessionId).then(function (d) {
      var runs = d.runs || [], ok = runs.filter(function (r) { return r.status === "succeeded"; });
      var lat = ok.map(function (r) { return r.latency_ms || 0; });
      var avg = lat.length ? Math.round(lat.reduce(function (a, b) { return a + b; }, 0) / lat.length) : 0;
      var kpis = '<div class="kpis"><div class="kpi"><b>' + runs.length + '</b><small>total runs</small></div>' +
        '<div class="kpi"><b>' + ok.length + '</b><small>succeeded</small></div>' +
        '<div class="kpi"><b>' + avg + 'ms</b><small>avg latency</small></div>' +
        '<div class="kpi"><b>' + (d.graphs || []).length + '</b><small>graphs</small></div></div>';
      var rows = runs.map(function (r) {
        var b = r.status === "succeeded" ? "ok" : (r.status === "failed" ? "fail" : "run");
        return '<tr><td><code>' + esc(r.id) + '</code></td><td>' + esc(r.input_text.slice(0, 44)) + '</td>' +
          '<td><span class="badge ' + b + '">' + r.status + '</span></td><td>' + esc(r.engine || "—") + '</td>' +
          '<td>' + (r.latency_ms || "—") + 'ms</td><td>' + esc(fmtTime(r.created_at)) + '</td></tr>';
      }).join("");
      $("#runs-body").innerHTML = kpis + '<table><thead><tr><th>Run</th><th>Input</th><th>Status</th><th>Engine</th><th>Latency</th><th>Created</th></tr></thead><tbody>' + (rows || '<tr><td colspan=6>No runs yet.</td></tr>') + '</tbody></table>';
    }).catch(function (e) { $("#runs-body").innerHTML = "Error: " + e.message; });
  }

  function renderArtifacts() {
    setTitle("Artifacts", "saved outputs, persisted per session");
    if (state.offline || !state.sessionId) { $("#view").innerHTML = '<div class="page"><p style="color:#5E6B7E">Artifacts appear here once connected and generating.</p></div>'; return; }
    $("#view").innerHTML = '<div class="page"><div class="grid" id="arts">Loading…</div></div>';
    api("/api/sessions/" + state.sessionId + "/artifacts").then(function (r) {
      var items = r.artifacts || [];
      $("#arts").innerHTML = items.length ? items.map(function (a) {
        return '<div class="kpi"><div class="art__ic" style="background:rgba(168,85,247,.18);margin-bottom:10px">▣</div><b style="font-size:14px">' + esc(a.name) + '</b><small>' + esc(a.kind) + ' · ' + esc(fmtTime(a.created_at)) + '</small></div>';
      }).join("") : '<p style="color:#5E6B7E">No artifacts yet.</p>';
    }).catch(function (e) { $("#arts").innerHTML = "Error: " + e.message; });
  }

  function renderModels() {
    setTitle("Models", "route each step to the right Claude model");
    var M = [
      ["Frontier reasoning", "Claude Opus", "claude-opus", "#A855F7", "Deepest reasoning and long-horizon agentic work.", [["Reasoning", 97], ["Agentic", 95], ["Speed", 55]]],
      ["Balanced", "Claude Sonnet", "claude-sonnet", "#22D3EE", "Near-frontier quality at production throughput.", [["Reasoning", 88], ["Agentic", 86], ["Speed", 78]]],
      ["High-volume", "Claude Haiku", "claude-haiku", "#34D399", "Fast, cheap classification and enrichment at scale.", [["Reasoning", 72], ["Agentic", 66], ["Speed", 95]]],
      ["Lightweight", "Claude Fable", "claude-fable", "#FBBF24", "Rapid drafting; keeps the UI responsive.", [["Reasoning", 64], ["Agentic", 58], ["Speed", 90]]]
    ];
    $("#view").innerHTML = '<div class="page"><div class="grid">' + M.map(function (m) {
      return '<div class="mcard"><div style="font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:' + m[3] + '">' + m[0] + '</div>' +
        '<h3>' + m[1] + '</h3><div class="id">' + m[2] + '</div><div style="font-size:12.5px;color:#AFBECF;line-height:1.5;margin-bottom:14px;min-height:40px">' + m[4] + '</div>' +
        m[5].map(function (b) { return '<div class="mbar"><div class="l"><span>' + b[0] + '</span><span style="color:' + m[3] + '">' + b[1] + '</span></div><div class="bar"><i style="width:' + b[1] + '%;background:' + m[3] + '"></i></div></div>'; }).join("") + '</div>';
    }).join("") + '</div></div>';
  }

  var VIEWS = { workspace: renderWorkspace, explorer: renderExplorer, runs: renderRuns, artifacts: renderArtifacts, models: renderModels };
  function setView(name) {
    if (canvas) { canvas.destroy(); canvas = null; }
    state.view = name;
    document.querySelectorAll(".nav__item").forEach(function (b) { b.classList.toggle("is-active", b.dataset.view === name); });
    (VIEWS[name] || renderWorkspace)();
  }

  /* =================== generation ==================== */
  function presetBody(q, nodes) {
    var map = { churn_intelligence: { generation_mode: "analytical", clustering_method: "causal", domain_focus: "business", enable_risk_analysis: true },
      creative_brainstorm: { generation_mode: "creative", clustering_method: "thematic", domain_focus: "creative", creativity_level: 0.9 },
      technical_planning: { generation_mode: "practical", clustering_method: "functional", domain_focus: "technology", min_confidence: 0.6 } };
    var body = { input_text: q, max_nodes: nodes }, p = map[($("#preset") || {}).value] || map.churn_intelligence;
    for (var k in p) if (p.hasOwnProperty(k)) body[k] = p[k]; return body;
  }
  function loading(on, label) {
    var host = $(".ws__graph") || $("#view"); if (!host) return;
    var ex = $(".loader", host); if (ex) ex.remove();
    if (on) { var l = document.createElement("div"); l.className = "loader"; l.innerHTML = '<div class="spin"></div><span>' + esc(label || "Generating…") + '</span>'; host.style.position = "relative"; host.appendChild(l); }
  }
  function generate(query) {
    if (state.busy) return; state.busy = true; $("#send") && ($("#send").disabled = true);
    loading(true, "Enqueuing…");
    var nodes = parseInt(($("#nodes") || {}).value, 10) || 16;
    ensureSession().then(function (sid) {
      loading(true, "Generating graph…");
      return api("/api/sessions/" + sid + "/graph", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(presetBody(query, nodes)) });
    }).then(function (job) { return poll(job.run_id, query, Date.now() + 220000); })
      .catch(function (e) { loading(false); state.busy = false; $("#send") && ($("#send").disabled = false); toast("Failed: " + e.message); });
  }
  function poll(runId, query, deadline) {
    return api("/api/runs/" + runId).then(function (run) {
      if (run.status === "succeeded") {
        return api("/api/graphs/" + run.graph_id).then(function (g) {
          var data = g.payload; data._query = query; state.graph = data;
          loading(false); state.busy = false; $("#send") && ($("#send").disabled = false);
          if (state.view === "workspace") paintWorkspace(data); else setView(state.view);
          toast("Graph ready · " + (data.engine || "engine") + " · " + (run.latency_ms || "?") + "ms");
        });
      }
      if (run.status === "failed") throw new Error(run.error || "generation failed");
      if (Date.now() > deadline) throw new Error("timed out");
      return new Promise(function (r) { setTimeout(r, 1200); }).then(function () { return poll(runId, query, deadline); });
    });
  }

  /* =================== sessions / bootstrap ==================== */
  function ensureSession() {
    if (state.sessionId) return Promise.resolve(state.sessionId);
    return api("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: "Workspace" }) })
      .then(function (s) { state.sessionId = s.id; loadSessions(); return s.id; });
  }
  function loadSessions() {
    api("/api/sessions").then(function (r) {
      state.sessions = r.sessions || [];
      var sel = $("#session-select"); if (!sel) return;
      sel.innerHTML = state.sessions.map(function (s) { return '<option value="' + s.id + '"' + (s.id === state.sessionId ? " selected" : "") + '>' + esc(s.title) + '</option>'; }).join("") || '<option>new…</option>';
    }).catch(function () {});
  }
  function health() {
    api("/api/health").then(function (h) {
      $("#status-dot").className = "dot " + (h.status === "healthy" ? "ok" : "bad");
      $("#status-text").textContent = h.status + " · " + h.store;
      $("#backend-chip").textContent = "● " + (h.model_backend || "offline");
      $("#backend-chip").style.color = h.model_backend && h.model_backend !== "none" ? "#34D399" : "#5E6B7E";
    }).catch(function () { $("#status-dot").className = "dot bad"; $("#status-text").textContent = "offline"; });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".nav__item").forEach(function (b) { b.addEventListener("click", function () { setView(b.dataset.view); }); });
    $("#btn-new").addEventListener("click", function () { state.sessionId = null; state.graph = null; ensureSession().then(function () { setView("workspace"); }); });
    $("#session-select").addEventListener("change", function (e) { state.sessionId = e.target.value; state.graph = null; setView("workspace"); loadRunGraph(); });

    if (state.offline) {
      var baked = window.__GRAPHOS_DATA__; baked.data._query = baked.query; state.graph = baked.data;
      $("#status-dot").className = "dot ok"; $("#status-text").textContent = "offline preview";
      $("#backend-chip").textContent = "● " + (baked.data.engine || "snapshot");
      setView("workspace");
    } else {
      health(); loadSessions(); ensureSession().then(function () { setView("workspace"); });
    }
  });
  function loadRunGraph() { /* pick latest graph for the session, if any */
    if (!state.sessionId) return;
    api("/api/sessions/" + state.sessionId).then(function (d) {
      if (d.graphs && d.graphs.length) {
        api("/api/graphs/" + d.graphs[0].id).then(function (g) { state.graph = g.payload; state.graph._query = (d.graphs[0] && "saved graph"); if (state.view === "workspace") paintWorkspace(state.graph); });
      }
    }).catch(function () {});
  }
})();
