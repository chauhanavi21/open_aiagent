"""Command line interface."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .agent import build_runtime
from .company import Company
from .config import Config

DEFAULT_CONFIG = "agentforge.json"


def _runtime(args):
    return build_runtime(Config.load(args.config))


def cmd_init(args) -> int:
    path = Path(args.config)
    if path.exists():
        print(f"{path} already exists; leaving it unchanged.")
        return 1
    Config().save(path)
    print(f"Wrote {path}. Default brain is 'local' (offline).")
    print("Free option: create a free OpenRouter key, set \"brain\": \"auto\", export OPENROUTER_API_KEY. See README.")
    return 0


def cmd_free_models(args) -> int:
    import json
    import urllib.request

    from .config import OPENROUTER_URL
    from .netutil import NetworkError, send

    try:
        data = json.loads(send(urllib.request.Request(OPENROUTER_URL + "/models"), 20, 5_000_000))["data"]
    except (NetworkError, ValueError, KeyError):
        print("error: could not fetch the model list", file=sys.stderr)
        return 2
    free = [m for m in data if str(m.get("id", "")).endswith(":free")]
    free.sort(key=lambda m: -int(m.get("context_length") or 0))
    for m in free:
        print(f"{m['id']}  (context {m.get('context_length')})")
    print(f"\n{len(free)} free models. Set \"openrouter_model\" in agentforge.json to one of them.")
    return 0


def cmd_brains(args) -> int:
    rt = _runtime(args)
    print("Brain chain (first healthy one answers):")
    for i, b in enumerate(rt.agent.brains, 1):
        print(f"  {i}. {b.name}")
    return 0


def cmd_run(args) -> int:
    res = _runtime(args).agent.run(args.goal)
    print(res.answer)
    if res.degraded:
        print("\n[note] third-party brain failed; answered by the local brain.", file=sys.stderr)
    return 0


def cmd_chat(args) -> int:
    rt = _runtime(args)
    print(f"AgentForge {__version__} | brain: {rt.agent.brain.name} | type 'exit' to quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in ("exit", "quit"):
            return 0
        if line:
            print(rt.agent.run(line).answer, "\n")


def cmd_company(args) -> int:
    rt = _runtime(args)
    run = Company(rt.agent, rt.workspace, rt.memory).run(args.mission, args.name or "")
    print(run.report)
    print(f"Files written to {rt.workspace.root / run.folder}")
    return 0


def cmd_serve(args) -> int:
    from .server import configure_logging, make_server, new_token

    configure_logging()
    rt = _runtime(args)
    token = os.environ.get(rt.config.token_env)
    if not token:
        token = new_token()
        print(f"No {rt.config.token_env} set; generated a one-time token for this session:\n{token}")
    elif len(token) < 24:
        print(f"{rt.config.token_env} must be at least 24 characters.", file=sys.stderr)
        return 2
    server = make_server(rt, token)
    print(f"Serving on http://{rt.config.host}:{rt.config.port} (Authorization: Bearer <token>)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentforge", description="From-scratch AI agent that scales into an AI company.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="config file (default: agentforge.json)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="write a default config").set_defaults(fn=cmd_init)
    r = sub.add_parser("run", help="run the agent on one goal")
    r.add_argument("goal")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("free-models", help="list currently free OpenRouter models").set_defaults(fn=cmd_free_models)
    sub.add_parser("brains", help="show the active brain chain").set_defaults(fn=cmd_brains)
    sub.add_parser("chat", help="interactive session").set_defaults(fn=cmd_chat)
    c = sub.add_parser("company", help="run the AI company on a mission")
    c.add_argument("mission")
    c.add_argument("--name", default="")
    c.set_defaults(fn=cmd_company)
    sub.add_parser("serve", help="start the local HTTP API").set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
