# AgentForge

An AI agent built **from scratch** (no third-party API, no model download, zero runtime dependencies) that scales into an **AI company**: a team of role-specialised agents that turn one mission into a full set of business documents.

A third-party LLM is **optional**: point the agent at any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, ...) and it becomes the agent's brain, with automatic fallback to the built-in brain if the provider fails.

## Strategy

| Layer | What it is | Why |
|---|---|---|
| **Brain** (pluggable) | `LocalBrain` (offline intent router and planner) or `OpenAICompatBrain` (optional) | Works on day one with no keys or cost; upgrades to an LLM with one config change |
| **Tools** | calculator (AST-safe), memory, summarizer, document writer, sandboxed files, opt-in web fetch | Same tool contract for both brains, so behaviour is consistent |
| **Memory** | TF-IDF retrieval written from scratch, persisted to JSON | Agents recall facts and company history |
| **Agent loop** | decide -> act -> observe, bounded steps, fallback brain | The reusable engine |
| **Company** | Roles (analyst, strategist, PM, marketer, CFO, CEO) over a task dependency graph | Same engine, many roles: this is how one agent becomes a company |
| **Interfaces** | CLI and a local authenticated HTTP API | Use it directly or embed it in a product |

Growth path: add roles and tasks in [`company.py`](src/agentforge/company.py), add tools in [`tools.py`](src/agentforge/tools.py), add a brain by implementing `Brain.decide`.

## Quick start

```powershell
python -m venv .venv; .\.venv\Scripts\activate
pip install -e ".[dev]"
agentforge run "write a business plan for a pet-food startup and save it"
agentforge company "an AI tutoring service for schools" --name EduSpark
agentforge chat
pytest
```

Company output goes to `workspace/company-<name>/` (`research.md`, `strategy.md`, `product.md`, `marketing.md`, `finance.md`, `summary.md`, `REPORT.md`).

## Brains: offline first, free hosted models next, paid last

`"brain": "auto"` builds this chain; the first healthy brain answers and failures fall through to the next:

1. `openrouter` - **free**: OpenRouter models whose id ends in `:free` (free account and API key, rate-limited, free models rotate). Key in `OPENROUTER_API_KEY`. Default `google/gemma-4-31b-it:free`.
2. `openai_compat` - any other OpenAI-compatible HTTPS endpoint (needs `model`, key in `AGENTFORGE_API_KEY`).
3. `ollama_cloud` - hosted Ollama, **paid per token** (`OLLAMA_API_KEY`). Skipped unless the key is set.
4. `ollama_local` - Ollama on this machine, last resort; only used if `ollama_local_model` is set.
5. `local` - the built-in offline brain, free and always the final safety net (`fallback_to_local`).

```powershell
agentforge init                          # writes agentforge.json; set "brain": "auto"
agentforge free-models                   # lists currently free OpenRouter models (no key needed)
$env:OPENROUTER_API_KEY = "<free key from openrouter.ai>"   # read from the environment only
agentforge brains                        # shows the active chain
agentforge company "a coffee subscription startup"
```

Pin one provider with `"brain": "openrouter"`, `"ollama_cloud"`, `"openai_compat"`, `"ollama_local"` or `"local"`.
Free-tier prompts go to a third party and may be logged or used by the model provider; use `"brain": "local"` for anything sensitive.

## HTTP API and web UI

```powershell
$env:AGENTFORGE_TOKEN = "<random string, 24+ chars>"   # or one is generated per session
agentforge serve                         # open http://127.0.0.1:8765/ for the web UI
curl -X POST http://127.0.0.1:8765/agent   -H "Authorization: Bearer $env:AGENTFORGE_TOKEN" -H "Content-Type: application/json" -d '{"goal":"what is 6*7"}'
curl -X POST http://127.0.0.1:8765/company -H "Authorization: Bearer $env:AGENTFORGE_TOKEN" -H "Content-Type: application/json" -d '{"mission":"a pet-food startup","name":"PawFuel"}'
```

## Security design

See [`SECURITY.md`](SECURITY.md) for the checklist mapping, trade-offs and items that do not apply. CI runs tests, bandit (SAST) and pip-audit on every push and weekly.


- Binds to loopback only (config rejects other hosts); bearer-token auth with constant-time compare; 64 KB body cap; generic error messages, details only in structured JSON logs.
- API keys come from environment variables only and are never written to config or logs.
- Providers must use HTTPS (plain HTTP only for loopback, e.g. local Ollama). TLS verification stays on; redirects are not followed; timeouts and response-size caps everywhere.
- Web fetch is **off by default**; when on, it is HTTPS-only with an SSRF guard (every resolved address must be public).
- File tools are confined to `workspace/` (no absolute paths, no `..`, extension allowlist, size cap).
- Each company role only sees the tools it needs (least privilege); independent tasks run in parallel by dependency level.
- The web UI is same-origin with a strict Content-Security-Policy and renders output via `textContent` only.
- Calculator parses the AST (no `eval`); tool arguments are type- and length-validated; tool output is treated as untrusted data in LLM prompts, and only registered tools can be invoked.

## Roadmap

Richer planner, pluggable vector embeddings, streaming responses, persisted run history.
