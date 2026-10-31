"""Static web UI assets. Served same-origin with a strict CSP; output is rendered with textContent only."""

INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AgentForge</title>
<style>
body{font:15px/1.6 -apple-system,"Segoe UI",system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1f2328}
input,textarea,button{font:inherit;padding:.5rem;border:1px solid #e5e7eb;border-radius:6px}
input,textarea{width:100%;box-sizing:border-box;margin:.25rem 0 .5rem}
button{background:#3b82d4;color:#fff;border:0;cursor:pointer;margin-right:.5rem}
pre{background:#f7f8fa;border:1px solid #e5e7eb;padding:1rem;white-space:pre-wrap;border-radius:6px}
h1{font-size:1.4rem}.muted{color:#57606a;font-size:.85rem}
</style></head><body>
<h1>AgentForge</h1>
<p class="muted">Local only. Your token stays in this page's memory and is never stored.</p>
<label>API token<input id="token" type="password" autocomplete="off"></label>
<label>Goal or company mission<textarea id="text" rows="3" maxlength="2000"></textarea></label>
<button id="agent">Run agent</button><button id="company">Run company</button>
<pre id="out">Ready.</pre>
<script src="/app.js"></script>
</body></html>
"""

APP_JS = """
const out = document.getElementById('out');
async function call(path, payload) {
  out.textContent = 'Working...';
  try {
    const r = await fetch(path, {method: 'POST',
      headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + document.getElementById('token').value},
      body: JSON.stringify(payload)});
    const d = await r.json();
    out.textContent = r.ok ? (d.answer || d.report) + (d.brain ? '\\n\\n[brain: ' + d.brain + (d.degraded ? ', fallback' : '') + ']' : '') : 'Error: ' + (d.error || r.status);
  } catch (e) { out.textContent = 'Request failed.'; }
}
document.getElementById('agent').onclick = () => call('/agent', {goal: document.getElementById('text').value});
document.getElementById('company').onclick = () => call('/company', {mission: document.getElementById('text').value});
"""
