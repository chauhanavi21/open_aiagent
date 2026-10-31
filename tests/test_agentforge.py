import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentforge.agent import Agent, build_runtime
from agentforge.brains import Action, BrainError, LocalBrain, OpenAICompatBrain, make_brains
from agentforge.company import Company, Task
from agentforge.config import Config
from agentforge.netutil import NetworkError, assert_public_https, validate_provider_url
from agentforge.nlp import TfidfIndex, summarize_text
from agentforge.server import make_server
from agentforge.tools import ToolError, safe_eval
from agentforge.workspace import Workspace, WorkspaceError


@pytest.fixture
def cfg(tmp_path):
    return Config(data_dir=str(tmp_path / "data"), workspace_dir=str(tmp_path / "ws"), port=0)


@pytest.fixture
def rt(cfg):
    return build_runtime(cfg)


# ---- core engine -------------------------------------------------------------
def test_calculator_and_safety():
    assert safe_eval("2*(3+4)") == 14
    for bad in ("__import__('os')", "2**9999", "1/0", "open('x')"):
        with pytest.raises(ToolError):
            safe_eval(bad)


def test_tfidf_ranks_relevant_first():
    idx = TfidfIndex()
    idx.add("the cat sat on the mat", "cat")
    idx.add("stock markets rallied on earnings", "stocks")
    assert idx.query("cat mat", k=1)[0][1] == "cat"


def test_summarizer_shorter_than_input():
    text = " ".join(f"Sentence number {i} talks about retention and growth metrics." for i in range(10))
    assert len(summarize_text(text, 2)) < len(text)


def test_memory_persists(cfg):
    rt1 = build_runtime(cfg)
    rt1.agent.run("remember that our pilot customer is Acme Bakery")
    rt2 = build_runtime(cfg)
    assert "Acme" in rt2.agent.run("what do you know about the pilot customer").answer


def test_agent_routes_calc_and_docs(rt):
    assert rt.agent.run("what is 6*7").answer == "42"
    doc = rt.agent.run("write a business plan for a pet-food startup and save it")
    assert doc.answer.startswith("Saved to business_plan-pet-food-startup.md")
    assert "pet-food startup" in rt.workspace.read("business_plan-pet-food-startup.md")


def test_unknown_goal_returns_help(rt):
    assert "I can" in rt.agent.run("blorp").answer


def test_tool_error_is_reported_not_raised(rt):
    assert "file not found" in rt.agent.run("read missing.md").answer


def test_fetch_disabled_by_default(rt):
    assert "disabled" in rt.agent.run("summarize https://example.com").answer


# ---- security ----------------------------------------------------------------
def test_workspace_blocks_escape(tmp_path):
    ws = Workspace(tmp_path / "ws")
    for bad in ("../x.md", "..\\x.md", "/etc/x.md", "C:\\x.md", "a.exe", "", "a\x00.md"):
        with pytest.raises(WorkspaceError):
            ws.write(bad, "x")
    assert ws.write("ok/a.md", "hi") == "ok/a.md"


def test_url_validation():
    assert validate_provider_url("https://api.example.com/v1/") == "https://api.example.com/v1"
    assert validate_provider_url("http://localhost:11434/v1")
    for bad in ("http://evil.example.com", "ftp://x.com", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            validate_provider_url(bad)


def test_ssrf_guard():
    for bad in ("http://example.com", "https://127.0.0.1/", "https://localhost/", "https://169.254.169.254/"):
        with pytest.raises(NetworkError):
            assert_public_https(bad)


def test_config_rules():
    with pytest.raises(ValueError):
        Config(host="0.0.0.0").validate()
    with pytest.raises(ValueError):
        Config(brain="openai_compat").validate()  # model missing
    with pytest.raises(ValueError):
        Config(brain="nope").validate()


def test_secret_not_serialised(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTFORGE_API_KEY", "sk-secret-value")
    path = tmp_path / "c.json"
    Config().save(path)
    assert "sk-secret-value" not in path.read_text()


# ---- third-party brain -------------------------------------------------------
class _FakeLLM(BaseHTTPRequestHandler):
    replies: list = []
    seen_auth: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        _FakeLLM.seen_auth.append(self.headers.get("Authorization"))
        content = _FakeLLM.replies.pop(0)
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_llm():
    srv = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _FakeLLM.seen_auth = []
    yield srv
    srv.shutdown()
    srv.server_close()


def test_third_party_brain_drives_tools(cfg, fake_llm, monkeypatch):
    monkeypatch.setenv("AGENTFORGE_API_KEY", "k-123")
    _FakeLLM.replies = ['{"tool": "calculate", "args": {"expression": "5*5"}}', '{"final": "It is 25"}']
    cfg.brain, cfg.model, cfg.base_url = "openai_compat", "m", f"http://127.0.0.1:{fake_llm.server_port}/v1"
    res = build_runtime(cfg).agent.run("compute five squared")
    assert res.answer == "It is 25" and res.steps[0].observation == "25"
    assert _FakeLLM.seen_auth[0] == "Bearer k-123"


def test_third_party_failure_falls_back_to_local(cfg):
    cfg.brain, cfg.model, cfg.base_url = "openai_compat", "m", "http://127.0.0.1:9/v1"  # nothing listens
    res = build_runtime(cfg).agent.run("what is 2+2")
    assert res.answer == "4" and res.degraded


def test_third_party_failure_without_fallback(cfg):
    cfg.brain, cfg.model, cfg.base_url, cfg.fallback_to_local = "openai_compat", "m", "http://127.0.0.1:9/v1", False
    assert "brains failed" in build_runtime(cfg).agent.run("hello").answer


def test_prompt_injection_text_cannot_add_tools(cfg, fake_llm):
    _FakeLLM.replies = ['{"tool": "delete_everything", "args": {}}', '{"final": "done"}']
    cfg.brain, cfg.model, cfg.base_url = "openai_compat", "m", f"http://127.0.0.1:{fake_llm.server_port}/v1"
    res = build_runtime(cfg).agent.run("x")
    assert res.steps[0].ok is False and res.answer == "done"


# ---- company -----------------------------------------------------------------
def test_company_produces_all_documents(rt):
    run = Company(rt.agent, rt.workspace, rt.memory).run("an AI tutoring service for schools", "EduSpark")
    assert [o.task for o in run.outcomes] == ["research", "strategy", "product", "marketing", "finance", "summary"]
    for o in run.outcomes:
        assert rt.workspace.read(o.file).strip()
    assert "Inputs from earlier work" in run.outcomes[-1].output
    assert (rt.workspace.root / run.folder / "REPORT.md").exists()
    assert "EduSpark" in rt.agent.run("what do you know about EduSpark").answer


def test_company_rejects_bad_graph_and_mission(rt):
    with pytest.raises(ValueError):
        Company(rt.agent, rt.workspace, tasks=(Task("a", "ceo", "executive summary", ("missing",)),))
    with pytest.raises(ValueError):
        Company(rt.agent, rt.workspace).run("x")


# ---- HTTP API ----------------------------------------------------------------
def _post(port, path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(payload).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_api(rt):
    srv = make_server(rt, "t" * 32)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        assert _post(port, "/agent", {"goal": "what is 1+1"})[0] == 401
        assert _post(port, "/agent", {"goal": "what is 1+1"}, "wrong" * 8)[0] == 401
        status, body = _post(port, "/agent", {"goal": "what is 1+1"}, "t" * 32)
        assert status == 200 and body["answer"] == "2"
        assert _post(port, "/agent", {"goal": 5}, "t" * 32)[0] == 400
        assert _post(port, "/company", {"mission": "x"}, "t" * 32)[0] == 400
        status, body = _post(port, "/company", {"mission": "a pet-food startup"}, "t" * 32)
        assert status == 200 and len(body["files"]) == 6
    finally:
        srv.shutdown()
        srv.server_close()


def test_brain_protocol_types():
    assert isinstance(LocalBrain().decide("what is 1+1", "", [], {}, "r"), Action)
    assert BrainError and OpenAICompatBrain and Agent


# ---- provider chain, permissions, parallelism --------------------------------
def test_ollama_cloud_needs_key_and_is_https(cfg, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    cfg.brain = "auto"
    assert [b.name for b in make_brains(cfg)] == ["local"]  # no key -> cloud skipped, offline still works
    monkeypatch.setenv("OLLAMA_API_KEY", "ok-key")
    cfg.ollama_local_model = "llama3"
    names = [b.name for b in make_brains(cfg)]
    assert names == ["ollama_cloud:gpt-oss:120b", "ollama_local:llama3", "local"]  # local Ollama is the last resort
    cfg.brain = "ollama_cloud"
    monkeypatch.delenv("OLLAMA_API_KEY")
    with pytest.raises(ValueError):
        make_brains(cfg)


def test_chain_falls_through_to_next_brain(cfg, fake_llm, monkeypatch):
    monkeypatch.setenv("AGENTFORGE_API_KEY", "k")
    _FakeLLM.replies = ['{"final": "from second"}']
    cfg.brain, cfg.ollama_local_model = "auto", "m"
    cfg.model, cfg.base_url = "m", "http://127.0.0.1:9/v1"  # openai_compat dead
    cfg.ollama_local_url = f"http://127.0.0.1:{fake_llm.server_port}/v1"
    res = build_runtime(cfg).agent.run("hello")
    assert res.answer == "from second" and res.degraded and res.brain.startswith("ollama_local")


def test_role_cannot_use_unlisted_tool(rt):
    scoped = rt.agent.with_role("Intern", frozenset({"recall"}))
    res = scoped.run("what is 2+2")  # local brain wants calculate -> blocked
    assert "not available" in res.answer


def test_parallel_levels_keep_order(rt):
    run = Company(rt.agent, rt.workspace, max_parallel=4).run("a drone delivery startup")
    assert [o.task for o in run.outcomes] == ["research", "strategy", "product", "marketing", "finance", "summary"]


def test_web_ui_served_with_csp(rt):
    srv = make_server(rt, "t" * 32)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
            assert r.status == 200 and "script-src 'self'" in r.headers["Content-Security-Policy"]
            assert b"AgentForge" in r.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/app.js", timeout=10) as r:
            assert b"textContent" in r.read()
    finally:
        srv.shutdown()
        srv.server_close()


def test_openrouter_is_first_in_auto_chain_and_free_by_default(cfg, monkeypatch):
    for k in ("OPENROUTER_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg.brain = "auto"
    assert [b.name for b in make_brains(cfg)] == ["local"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    names = [b.name for b in make_brains(cfg)]
    assert names[0].startswith("openrouter:") and names[0].endswith(":free")  # free before paid Ollama cloud
    assert names.index("ollama_cloud:gpt-oss:120b") > 0 and names[-1] == "local"
