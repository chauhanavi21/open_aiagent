"""Tool registry. Every tool validates its own inputs; failures surface as ToolError."""
from __future__ import annotations

import ast
import html
import operator
import re
import urllib.request
from dataclasses import dataclass
from typing import Callable

from . import docs
from .config import Config
from .memory import Memory
from .netutil import NetworkError, assert_public_https, send
from .nlp import summarize_text
from .workspace import Workspace, WorkspaceError

NO_MEMORY = "No matching memories."
MAX_ARG = 20_000


class ToolError(Exception):
    """Safe-to-display tool failure."""


@dataclass
class ToolContext:
    memory: Memory
    workspace: Workspace
    config: Config


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # arg name -> description
    fn: Callable[..., str]
    required: tuple[str, ...] = ()

    def call(self, args: object) -> str:
        if not isinstance(args, dict):
            raise ToolError("arguments must be an object")
        clean: dict[str, str] = {}
        for key in self.params:
            if key in args and args[key] is not None:
                value = args[key]
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise ToolError(f"argument '{key}' must be text")
                value = str(value)
                if len(value) > MAX_ARG:
                    raise ToolError(f"argument '{key}' is too long")
                clean[key] = value
        missing = [k for k in self.required if k not in clean]
        if missing:
            raise ToolError(f"missing argument: {', '.join(missing)}")
        return self.fn(**clean)


# ---- safe calculator -------------------------------------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expr: str) -> float | int:
    if len(expr) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        raise ToolError("not a valid arithmetic expression") from None

    def ev(node: ast.AST, depth: int = 0) -> float | int:
        if depth > 20:
            raise ToolError("expression too deep")
        if isinstance(node, ast.Expression):
            return ev(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand, depth + 1))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left, depth + 1), ev(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent too large")
            try:
                return _BIN[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ToolError("division by zero") from None
            except OverflowError:
                raise ToolError("number too large") from None
        raise ToolError("only arithmetic is supported")

    return ev(tree)


def _calculate(expression: str) -> str:
    result = safe_eval(expression)
    return str(round(result, 10) if isinstance(result, float) else result)


# ---- fetch -----------------------------------------------------------------
def _fetch(ctx: ToolContext, url: str) -> str:
    if not ctx.config.enable_fetch:
        raise ToolError("web fetching is disabled (set enable_fetch to true in the config)")
    try:
        assert_public_https(url)
        data = send(urllib.request.Request(url, headers={"User-Agent": "AgentForge/0.1"}), 15, 500_000)
    except NetworkError as exc:
        raise ToolError(f"fetch failed: {exc}") from None
    text = data.decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()[:8000] or "(empty page)"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "doc"


def build_tools(ctx: ToolContext) -> dict[str, Tool]:
    def remember(text: str) -> str:
        try:
            return f"Remembered (item {ctx.memory.add(text)})."
        except ValueError as exc:
            raise ToolError(str(exc)) from None

    def recall(query: str) -> str:
        hits = ctx.memory.search(query)
        return "\n".join(f"- {h}" for h in hits) if hits else NO_MEMORY

    def write_doc(template: str, subject: str, context: str = "", save_as: str = "") -> str:
        template = docs.ALIASES.get(template.lower().strip(), template.strip())
        try:
            doc = docs.render(template, subject, context, tuple(ctx.memory.search(subject, k=3)))
            if save_as:
                return f"Saved to {ctx.workspace.write(save_as, doc)}\n\n{doc}"
        except (ValueError, WorkspaceError) as exc:
            raise ToolError(str(exc)) from None
        return doc

    def read_file(path: str) -> str:
        try:
            return ctx.workspace.read(path)
        except WorkspaceError as exc:
            raise ToolError(str(exc)) from None

    def write_file(path: str, content: str) -> str:
        try:
            return f"Wrote {ctx.workspace.write(path, content)}"
        except WorkspaceError as exc:
            raise ToolError(str(exc)) from None

    tools = [
        Tool("calculate", "Evaluate an arithmetic expression.", {"expression": "e.g. 12*(3+4)"}, _calculate, ("expression",)),
        Tool("remember", "Store a fact in long-term memory.", {"text": "the fact"}, remember, ("text",)),
        Tool("recall", "Search long-term memory.", {"query": "what to look for"}, recall, ("query",)),
        Tool("summarize_text", "Summarize a block of text.", {"text": "text to summarize"},
             lambda text: summarize_text(text), ("text",)),
        Tool("write_doc", "Create a business document. Templates: " + ", ".join(sorted(docs.SECTIONS)),
             {"template": "template name", "subject": "what it is about", "context": "optional earlier work",
              "save_as": "optional workspace path, e.g. plan.md"}, write_doc, ("template", "subject")),
        Tool("read_file", "Read a text file from the workspace.", {"path": "relative path"}, read_file, ("path",)),
        Tool("write_file", "Write a text file into the workspace.", {"path": "relative path", "content": "file text"},
             write_file, ("path", "content")),
        Tool("fetch_url", "Fetch a public https page as plain text (must be enabled in config).",
             {"url": "https URL"}, lambda url: _fetch(ctx, url), ("url",)),
    ]
    return {t.name: t for t in tools}
