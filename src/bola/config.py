"""Configuration: 4-layer merge (defaults < user < project < env < CLI overrides)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

USER_CONFIG = Path.home() / ".config" / "bola" / "config.toml"

DEFAULTS: dict[str, Any] = {
    "ollama_url": "http://localhost:11434",
    "planner_model": "qwen3.8:27b",
    "worker_model": "qwen3-coder:30b",
    "num_ctx": 32768,               # total context window budget per request
    "output_reserve": 4096,         # tokens reserved for model output
    "state_note_max": 2500,         # max tokens for the carry-forward state note
    "max_turns": 24,                # max chat/tool turns per subtask session
    "max_sessions": 3,
    "bash_timeout": 120,            # per-command timeout in seconds
    "require_confirm": True,        # ask before executing a task graph
    "allow_network": False,         # bash commands may not use the network
    "bash_allow": [],               # command-prefix allowlist (deny-by-default)
    "bash_deny": ["rm", "sudo", "shut", "reboot", "curl", "wget", "nc", "dd",
                  "mkfs", "chmod 777"],
    "read_max_bytes": 256 * 1024,   # cap per file read
    "bash_max_output": 16384,       # cap bash output captured
    "log_raw": False,               # log full request/response bodies to run.log
}

# env var name per config key
_ENV_MAP = {
    "ollama_url": "BOLA_OLLAMA_URL",
    "planner_model": "BOLA_PLANNER_MODEL",
    "worker_model": "BOLA_WORKER_MODEL",
    "num_ctx": "BOLA_NUM_CTX",
    "output_reserve": "BOLA_OUTPUT_RESERVE",
    "bash_timeout": "BOLA_BASH_TIMEOUT",
    "allow_network": "BOLA_ALLOW_NETWORK",
}


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import tomllib
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _env_value(key: str) -> Any | None:
    name = _ENV_MAP.get(key)
    if not name:
        return None
    raw = os.environ.get(name)
    if raw is None:
        return None
    if key in ("num_ctx", "output_reserve", "bash_timeout"):
        return int(raw)
    if key == "allow_network":
        return raw.strip().lower() in ("1", "true", "yes")
    return raw


@dataclass
class Config:
    ollama_url: str = DEFAULTS["ollama_url"]
    planner_model: str = DEFAULTS["planner_model"]
    worker_model: str = DEFAULTS["worker_model"]
    num_ctx: int = DEFAULTS["num_ctx"]
    output_reserve: int = DEFAULTS["output_reserve"]
    state_note_max: int = DEFAULTS["state_note_max"]
    max_turns: int = DEFAULTS["max_turns"]
    max_sessions: int = DEFAULTS["max_sessions"]
    bash_timeout: int = DEFAULTS["bash_timeout"]
    require_confirm: bool = DEFAULTS["require_confirm"]
    allow_network: bool = DEFAULTS["allow_network"]
    bash_allow: list[str] = field(default_factory=list)
    bash_deny: list[str] = field(default_factory=list)
    read_max_bytes: int = DEFAULTS["read_max_bytes"]
    bash_max_output: int = DEFAULTS["bash_max_output"]
    log_raw: bool = DEFAULTS["log_raw"]
    # set after merge, not user-facing:
    root: Path = field(default_factory=Path.cwd, compare=False)
    project_dir: Path | None = field(default=None, compare=False)

    def validate(self) -> None:
        if self.num_ctx <= 0:
            raise ValueError("num_ctx must be > 0")
        if self.output_reserve >= self.num_ctx:
            raise ValueError("output_reserve must be < num_ctx")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be > 0")
        if self.max_sessions <= 0 or self.output_reserve <= 0 or self.state_note_max <= 0:
            raise ValueError("session and token limits must be positive")
        if self.num_ctx - self.output_reserve < 4000:
            raise ValueError("Leave at least 4000 tokens for prompts and tools")


def load_config(root: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    root = Path(root or Path.cwd())
    merged = dict(DEFAULTS)
    merged.update(_load_toml(USER_CONFIG))
    project_cfg = root / ".bola.toml"
    if project_cfg.is_file():
        merged.update(_load_toml(project_cfg))
        merged.setdefault("project_dir", str(root))
    for key in DEFAULTS:
        env_val = _env_value(key)
        if env_val is not None:
            merged[key] = env_val
    if overrides:
        for key, val in overrides.items():
            if val is None:
                continue
            if key == "root":
                continue
            merged[key] = val
    known = {f.name for f in fields(Config)}
    known.discard("root")
    kwargs = {k: v for k, v in merged.items() if k in known}
    cfg = Config(root=root, **kwargs)
    cfg.project_dir = Path(cfg.project_dir) if cfg.project_dir else (root if project_cfg.is_file() else None)
    cfg.validate()
    return cfg
