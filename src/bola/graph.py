"""Task graph model, validation, and topological ordering."""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Subtask:
    id: str
    title: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    # execution fields (filled by executor):
    status: str = "pending"            # pending|running|done|failed|skipped
    attempts: int = 0
    result: str = ""
    files: list[str] = field(default_factory=list)
    blockers: str = ""
    next_hints: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Subtask":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TaskGraph:
    id: str
    task: str
    created_at: float
    tasks: list[Subtask] = field(default_factory=list)
    notes: str = ""                    # planner notes / global constraints
    order: list[str] = field(default_factory=list)

    def get(self, tid: str) -> Subtask | None:
        for t in self.tasks:
            if t.id == tid:
                return t
        return None

    def topological(self) -> list[str]:
        by_id = {t.id: t for t in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("duplicate task ids")
        if any(d not in by_id for t in self.tasks for d in t.depends_on):
            unknown = {d for t in self.tasks for d in t.depends_on if d not in by_id}
            raise ValueError(f"unknown depends_on ids: {sorted(unknown)}")
        indeg = {t.id: len(t.depends_on) for t in self.tasks}
        dependents: dict[str, list[str]] = {t.id: [] for t in self.tasks}
        for t in self.tasks:
            for d in t.depends_on:
                dependents[d].append(t.id)
        ready = [tid for tid, n in indeg.items() if n == 0]
        order: list[str] = []
        while ready:
            ready.sort()
            cur = ready.pop(0)
            order.append(cur)
            for nxt in dependents[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        if len(order) != len(self.tasks):
            cycle = sorted(t for t, n in indeg.items() if n > 0)
            raise ValueError(f"dependency cycle involving: {cycle}")
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "task": self.task, "created_at": self.created_at,
            "notes": self.notes,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskGraph":
        return cls(
            id=d.get("id") or new_id(),
            task=d.get("task", ""),
            created_at=float(d.get("created_at") or time.time()),
            notes=d.get("notes", ""),
            tasks=[Subtask.from_dict(t) for t in d.get("tasks", [])],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TaskGraph":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def make_graph(task: str, raw: dict[str, Any]) -> TaskGraph:
    """Build a validated graph from a planner JSON blob."""
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError('planner output must contain a non-empty "tasks" list')
    tasks: list[Subtask] = []
    for i, tr in enumerate(tasks_raw):
        if not isinstance(tr, dict) or not tr.get("title"):
            raise ValueError(f"task #{i} missing 'title'")
        tid = str(tr.get("id") or f"t{i:02d}")
        deps = tr.get("depends_on") or []
        if not isinstance(deps, list):
            raise ValueError(f"task {tid}: depends_on must be a list")
        tasks.append(Subtask(
            id=tid,
            title=str(tr["title"]).strip(),
            description=str(tr.get("description", "")).strip(),
            depends_on=[str(x) for x in deps],
        ))
    g = TaskGraph(id=new_id(), task=task, created_at=time.time(),
                  tasks=tasks, notes=str(raw.get("notes", "")))
    g.order = g.topological()  # raises on cycles/unknown deps
    return g


def parse_plan_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from planner output (tolerates code fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    elif text.startswith("`") and text.endswith("`"):
        text = text.strip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("planner output contains no JSON object")
    return json.loads(text[start:end + 1])
