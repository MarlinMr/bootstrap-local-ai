"""Planner: one LLM call that turns a big task into a dependency-ordered
task graph. Uses the project's repo layout as context (cheaply gathered),
and enforces output to strict JSON via a `plan` tool call when available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import Config
from .graph import TaskGraph, make_graph, parse_plan_json
from .logging import Logger
from .ollama_client import Ollama, OllamaError

PLAN_TOOL_NAME = "propose_plan"

PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": PLAN_TOOL_NAME,
        "description": "Submit the decomposed plan for the user's task.",
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {"type": "string",
                          "description": "global constraints / conventions to follow"},
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "short id like t01"},
                            "title": {"type": "string"},
                            "description": {"type": "string",
                                           "description": "precise, self-contained instructions "
                                                          "a single agent session can complete"},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "title", "description"],
                    },
                },
            },
            "required": ["tasks"],
        },
    },
}

SYSTEM = (
    "You are a planning agent. Decompose the user's task into the SMALLEST set of "
    "subtasks that a SINGLE local-LLM coding agent session can complete within a "
    "limited context window (only a few thousand prompt tokens of file content per "
    "session plus your output). Rules:\n"
    "1. Each subtask must be completable in ONE session: read a few files, make a "
    "focused change, and finish. No subtask may require holding the whole project "
    "in context.\n"
    "2. Order subtasks as a DAG with 'depends_on'. Prefer small steps over one big "
    "rewrite.\n"
    "3. Include a first subtask for exploring/reporting the relevant existing code "
    "when the layout is unknown.\n"
    "4. Write descriptions as step-by-step instructions listing the files to touch.\n"
    "5. Aim for 2-12 subtasks unless the task clearly needs more.\n"
    "Call the propose_plan tool exactly once with the full plan."
)


def repo_snapshot(root: Path, max_files: int = 200) -> str:
    """Cheap structural snapshot of the repo for the planner (no LLM yet)."""
    files = []
    skip = {".git", "node_modules", ".venv", "__pycache__", ".bola", "target", "dist"}
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for name in sorted(names):
            files.append(str((Path(directory) / name).relative_to(root)))
            if len(files) >= max_files:
                return "\n".join(files + ["… (listing capped)"])
    return "\n".join(files)


def plan(config: Config, task: str, log: Logger, snapshot: str | None = None) -> TaskGraph:
    if snapshot is None:
        snapshot = repo_snapshot(config.root)
    system = SYSTEM
    user = (
        f"# PROJECT\nRoot: {config.root}\n\n"
        f"## Existing files (first {200} shown)\n{snapshot}\n\n"
        f"# USER TASK\n{task}\n\n"
        "Now call propose_plan."
    )
    client = Ollama(config.ollama_url)
    log.section(f"Planning with {config.planner_model}")
    log.dim(f"requesting plan for: {task[:120]}\n")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    opts = {"temperature": 0.2, "num_predict": config.output_reserve}
    if len(user) + len(system) > (config.num_ctx - config.output_reserve - 2500) * 3:
        raise ValueError("Planner input too large; shorten the task or increase --ctx")
    try:
        raw = ""
        plan_args: dict[str, Any] | None = None
        for chunk in client.stream_chat(model=config.planner_model, messages=messages,
                                        tools=[PLAN_TOOL], num_ctx=config.num_ctx,
                                        options=opts):
            msg = chunk.get("message") or {}
            if msg.get("content"):
                log.stream(msg["content"])
                raw += msg["content"]
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                if fn.get("name") == PLAN_TOOL_NAME:
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    plan_args = args
            if chunk.get("done"):
                log.kv("prompt_tokens", chunk.get("prompt_eval_count"))
                log.kv("completion_tokens", chunk.get("eval_count"))
                break
        log.stream_end()
        if plan_args is not None:
            return make_graph(task, plan_args)
        # fallback: parse JSON from text
        log.dim("no plan tool call received; parsing JSON from text")
        return make_graph(task, parse_plan_json(raw))
    except OllamaError as e:
        log.err(str(e))
        raise
