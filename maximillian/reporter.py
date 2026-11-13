"""Standalone HTML diagnostic report generator (Stage 14).

Produces an air-gapped report (inline CSS + SVG, no external CDN per FR-04) in
a neon-graph dark theme, one file per round:
``reports/round_<N>_<YYYYMMDD_HHMMSS>.html``.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from .schemas import RoundResult

_CSS = """
:root{--bg:#070A10;--panel:#0B0F17;--card:#0D1420;--line:#1A2433;--text:#E8EEF7;
--dim:#9AA7B8;--mute:#5E6B7E;--cyan:#22D3EE;--violet:#A855F7;--green:#34D399;
--pink:#F472B6;--amber:#FBBF24;}
*{box-sizing:border-box;}
body{margin:0;background:radial-gradient(1000px 600px at 20% -10%,rgba(34,211,238,.07),transparent 60%),var(--bg);
color:var(--text);font-family:-apple-system,Segoe UI,Roboto,Inter,sans-serif;padding:32px;}
h1{font-size:24px;margin:0 0 4px;}h2{font-size:14px;letter-spacing:1px;text-transform:uppercase;color:var(--mute);margin:26px 0 12px;}
.sub{color:var(--mute);font-size:13px;margin-bottom:8px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px;}
.kpi{font-size:28px;font-weight:700;}.kpi small{font-size:12px;color:var(--mute);font-weight:400;}
.tag{display:inline-block;font-size:11px;padding:3px 10px;border-radius:999px;border:1px solid var(--line);}
.ok{color:var(--green);border-color:#1c4536;background:rgba(52,211,153,.08);}
.warn{color:var(--amber);border-color:#3a2e12;background:rgba(251,191,36,.08);}
.bad{color:var(--pink);border-color:#3a1d2b;background:rgba(244,114,182,.08);}
table{width:100%;border-collapse:collapse;font-size:13px;}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);}
th{color:var(--mute);font-size:11px;letter-spacing:.5px;text-transform:uppercase;}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--cyan);}
.bar{height:8px;border-radius:999px;background:#141C28;overflow:hidden;}
.bar>i{display:block;height:100%;}
.foot{margin-top:30px;color:var(--mute);font-size:12px;}
"""

_SEV_COLOR = {"CRITICAL": "#F472B6", "HIGH": "#FB923C", "MEDIUM": "#FBBF24", "LOW": "#34D399"}


def _bar(value: float, maximum: float, color: str) -> str:
    pct = 0 if maximum <= 0 else min(100, round(100 * value / maximum))
    return f'<div class="bar"><i style="width:{pct}%;background:{color}"></i></div>'


def _severity_svg(counts: dict) -> str:
    bars, x, bw, gap = [], 10, 54, 18
    mx = max(counts.values()) or 1
    for sev, n in counts.items():
        h = int(120 * n / mx)
        color = _SEV_COLOR.get(sev, "#22D3EE")
        bars.append(
            f'<rect x="{x}" y="{140 - h}" width="{bw}" height="{h}" rx="5" fill="{color}"/>'
            f'<text x="{x + bw/2}" y="158" fill="#9AA7B8" font-size="10" text-anchor="middle">{sev[:4]}</text>'
            f'<text x="{x + bw/2}" y="{134 - h}" fill="#E8EEF7" font-size="12" text-anchor="middle">{n}</text>')
        x += bw + gap
    return f'<svg width="100%" height="170" viewBox="0 0 {x} 170">{"".join(bars)}</svg>'


def _esc(s: object) -> str:
    return html.escape(str(s))


def render_report(result: RoundResult, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
    out = reports_dir / f"round_{result.round_id:03d}_{ts}.html"

    audit = result.audit
    graph = result.graph
    tests = result.tests
    reward = result.reward
    conv = "ok" if result.converged else "warn"
    conv_label = "CONVERGED" if result.converged else "OPEN — more rounds available"

    sev = audit.by_severity() if audit else {}
    central_rows = "".join(
        f"<tr><td><code>{_esc(c.get('node'))}</code></td><td>{_esc(c.get('pagerank'))}</td></tr>"
        for c in (graph.top_central[:6] if graph else []))
    timeline_rows = "".join(
        f"<tr><td>{_esc(t.stage)}</td><td>{_esc(round(t.duration_s,3))}s</td>"
        f"<td>{'✅' if t.ok else '⚠️'} {_esc(t.note)}</td></tr>" for t in result.timeline)
    test_rows = "".join(
        f"<tr><td>#{it.attempt}</td><td>{it.passed}</td><td>{it.failed}</td>"
        f"<td>{it.errors}</td><td>{_esc(round(it.duration_s,3))}s</td>"
        f"<td>{'✅' if it.exit_code==0 else '❌'}</td></tr>"
        for it in (tests.iterations if tests else []))

    llm_badge = (f'<span class="tag ok">autopilot ON · {_esc(result.backend)}</span>'
                 if result.llm_available
                 else '<span class="tag warn">deterministic mode — no model in env</span>')

    proposals_html = ""
    if result.refactor_proposals:
        items = "".join(f"<li>{_esc(p)}</li>" for p in result.refactor_proposals)
        proposals_html = (f'<h2>Autopilot — model refactor proposals</h2>'
                          f'<div class="card"><ol style="margin:0;padding-left:20px;line-height:1.7;font-size:13px">{items}</ol></div>')
    plan_html = ""
    if result.strategic_plan_text:
        plan_html = (f'<h2>Autopilot — strategic plan (next round)</h2>'
                     f'<div class="card" style="line-height:1.6;font-size:13px">{_esc(result.strategic_plan_text)}</div>')

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maximillian · Round {result.round_id}</title><style>{_CSS}</style></head><body>
<h1>Maximillian — Round {result.round_id}</h1>
<div class="sub">Target <code>{_esc(result.target)}</code> · {_esc(result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))} · {llm_badge} · <span class="tag {conv}">{conv_label}</span></div>

<h2>Executive summary</h2>
<div class="grid">
  <div class="card"><div class="kpi">{audit.total_findings if audit else 0}<small> findings</small></div></div>
  <div class="card"><div class="kpi">{graph.nodes if graph else 0}<small> graph nodes</small></div></div>
  <div class="card"><div class="kpi">{graph.edges if graph else 0}<small> edges</small></div></div>
  <div class="card"><div class="kpi">{tests.attempts if tests else 0}<small> test attempts</small></div></div>
  <div class="card"><div class="kpi">{reward.reward_score if reward else 0.0}<small> reward R</small></div></div>
</div>

<h2>Audit findings by severity</h2>
<div class="card">{_severity_svg(sev) if sev else '<div class="sub">No findings.</div>'}</div>

<h2>Graph intelligence — top central nodes (PageRank)</h2>
<div class="card"><table><thead><tr><th>Node</th><th>PageRank</th></tr></thead>
<tbody>{central_rows or '<tr><td colspan=2>—</td></tr>'}</tbody></table></div>

<h2>Test-till-pass convergence</h2>
<div class="card"><table><thead><tr><th>Attempt</th><th>Passed</th><th>Failed</th><th>Errors</th><th>Duration</th><th>Status</th></tr></thead>
<tbody>{test_rows or '<tr><td colspan=6>Tests not run.</td></tr>'}</tbody></table></div>

<h2>DSPy reward breakdown</h2>
<div class="card grid">
  <div><div class="sub">Test-pass speed</div>{_bar(reward.test_pass_speed if reward else 0,1,'#22D3EE')}</div>
  <div><div class="sub">Diff minimality</div>{_bar(reward.diff_minimality if reward else 0,1,'#A855F7')}</div>
  <div><div class="sub">Complexity reduction</div>{_bar(reward.complexity_reduction if reward else 0,1,'#34D399')}</div>
</div>

{proposals_html}
{plan_html}

<h2>Stage timeline</h2>
<div class="card"><table><thead><tr><th>Stage</th><th>Duration</th><th>Status</th></tr></thead>
<tbody>{timeline_rows or '<tr><td colspan=3>—</td></tr>'}</tbody></table></div>

<div class="foot">Git SHA <code>{_esc(result.git_sha or '—')}</code> · snapshot <code>{_esc(result.snapshot_tag or '—')}</code>
· generated by Maximillian v5.2 · air-gapped (no external resources)</div>
</body></html>"""

    out.write_text(doc, encoding="utf-8")
    return out
