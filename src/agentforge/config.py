"""Configuration. Secrets are never stored here: only the *name* of the env var holding them."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .netutil import LOOPBACK_HOSTS, validate_provider_url

BRAINS = ("local", "openrouter", "openai_compat", "ollama_cloud", "ollama_local", "auto")
OLLAMA_CLOUD_URL = "https://ollama.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderSpec:
    label: str
    base_url: str
    model: str
    api_key: str | None = None


@dataclass
class Config:
    data_dir: str = ".agentforge"
    workspace_dir: str = "workspace"
    # local = built-in offline engine | openrouter = free ":free" models (free account key) | openai_compat = any other endpoint
    # ollama_cloud = hosted Ollama (PAID per token) | ollama_local = Ollama on this machine (last resort)
    # auto = openrouter, openai_compat, ollama_cloud, ollama_local, then local
    brain: str = "local"
    base_url: str = "https://api.openai.com/v1"  # any OpenAI-compatible endpoint (Ollama: http://localhost:11434/v1)
    model: str = ""
    api_key_env: str = "AGENTFORGE_API_KEY"
    fallback_to_local: bool = True
    openrouter_model: str = "google/gemma-4-31b-it:free"  # free models rotate; list them with: agentforge free-models
    openrouter_key_env: str = "OPENROUTER_API_KEY"
    ollama_cloud_model: str = "gpt-oss:120b"
    ollama_cloud_key_env: str = "OLLAMA_API_KEY"
    ollama_local_url: str = "http://localhost:11434/v1"
    ollama_local_model: str = ""
    enable_fetch: bool = False
    max_steps: int = 6
    host: str = "127.0.0.1"
    port: int = 8765
    token_env: str = "AGENTFORGE_TOKEN"

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None

    def provider_spec(self, name: str) -> ProviderSpec | None:
        """Resolve a provider preset; None when it is not usable (no model or no key)."""
        if name == "openrouter":
            key = os.environ.get(self.openrouter_key_env) or None
            return ProviderSpec("openrouter", OPENROUTER_URL, self.openrouter_model, key) if key and self.openrouter_model else None
        if name == "ollama_cloud":
            key = os.environ.get(self.ollama_cloud_key_env) or None
            return ProviderSpec("ollama_cloud", OLLAMA_CLOUD_URL, self.ollama_cloud_model, key) if key and self.ollama_cloud_model else None
        if name == "ollama_local":
            return ProviderSpec("ollama_local", self.ollama_local_url, self.ollama_local_model) if self.ollama_local_model else None
        if name == "openai_compat":
            return ProviderSpec("openai_compat", self.base_url, self.model, self.api_key) if self.model else None
        return None

    def validate(self) -> "Config":
        if self.brain not in BRAINS:
            raise ValueError(f"brain must be one of {BRAINS}")
        if self.brain == "openai_compat" and not self.model:
            raise ValueError("model is required when brain is openai_compat")
        for url in (self.base_url, self.ollama_local_url):
            validate_provider_url(url)
        if not 1 <= self.max_steps <= 20:
            raise ValueError("max_steps must be between 1 and 20")
        if not 0 <= self.port <= 65535:
            raise ValueError("port out of range")
        if self.host not in LOOPBACK_HOSTS:
            raise ValueError("host must be a loopback address")
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        data: dict = {}
        if path and Path(path).exists():
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config file must contain a JSON object")
            known = {f.name for f in fields(cls)}
            data = {k: v for k, v in raw.items() if k in known}
        for key, env in (("brain", "AGENTFORGE_BRAIN"), ("model", "AGENTFORGE_MODEL"), ("base_url", "AGENTFORGE_BASE_URL")):
            if os.environ.get(env):
                data[key] = os.environ[env]
        return cls(**data).validate()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @property
    def memory_path(self) -> Path:
        return Path(self.data_dir) / "memory.json"
