"""Per-project run store: state.json (machine) + run.log (human), and a
history helper that scans known project dirs for '.bola/runs'."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .graph import TaskGraph


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


@dataclass
class RunState:
    root: Path                # project root
    id: str
    graph: TaskGraph
    state_note: str = ""      # carry-forward note shared across subtasks
    current_task_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    outcome: str = "running"  # running|done|failed
    totals: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0, "turns": 0,
    })

    # -- paths -------------------------------------------------------------
    @property
    def bola_dir(self) -> Path:
        return self.root / ".bola"

    @property
    def run_dir(self) -> Path:
        return self.bola_dir / "runs" / self.id

    @property
    def path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def graph_path(self) -> Path:
        return self.run_dir / "task-graph.json"

    def save(self) -> None:
        self.updated_at = time.time()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.id, "root": str(self.root),
            "created_at": self.created_at, "updated_at": self.updated_at,
            "finished_at": self.finished_at, "outcome": self.outcome,
            "state_note": self.state_note,
            "current_task_id": self.current_task_id,
            "totals": self.totals,
            "graph": self.graph.to_dict(),
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self.graph.save(self.graph_path)

    @classmethod
    def load(cls, path: Path) -> "RunState":
        data = json.loads(path.read_text(encoding="utf-8"))
        g = TaskGraph.from_dict(data["graph"])
        rs = cls(
            root=Path(data.get("root") or Path.cwd()),
            id=data.get("id") or run_id(),
            graph=g,
            state_note=data.get("state_note", ""),
            current_task_id=data.get("current_task_id"),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            finished_at=float(data.get("finished_at") or 0.0),
            outcome=data.get("outcome", "running"),
            totals=dict(data.get("totals", {})),
        )
        return rs

    @classmethod
    def latest(cls, root: Path) -> "RunState | None":
        runs = root / ".bola" / "runs"
        if not runs.is_dir():
            return None
        dirs = sorted((d for d in runs.iterdir() if (d / "state.json").is_file()))
        if not dirs:
            return None
        return cls.load(dirs[-1] / "state.json")

    def add_totals(self, *, prompt_tokens: int = 0, completion_tokens: int = 0,
                   tool_calls: int = 0, turns: int = 0) -> None:
        self.totals["prompt_tokens"] += prompt_tokens
        self.totals["completion_tokens"] += completion_tokens
        self.totals["tool_calls"] += tool_calls
        self.totals["turns"] += turns


def iter_runs(root: Path) -> Iterator[Path]:
    runs = root / ".bola" / "runs"
    if not runs.is_dir():
        return
    for d in sorted(runs.iterdir()):
        if (d / "state.json").is_file():
            yield d


def scan_history(roots: list[Path] | None = None) -> list[dict[str, Any]]:
    """Scan the given roots (or the cwd) for .bola/runs and summarize runs."""
    roots = [Path(r) for r in (roots or [Path.cwd()])]
    out: list[dict[str, Any]] = []
    for root in roots:
        for d in iter_runs(root):
            try:
                rs = RunState.load(d / "state.json")
            except Exception as e:
                continue
            done = sum(1 for t in rs.graph.tasks if t.status == "done")
            failed = sum(1 for t in rs.graph.tasks if t.status == "failed")
            out.append({
                "run_id": rs.id, "root": str(root),
                "task": rs.graph.task[:80],
                "outcome": rs.outcome,
                "tasks_done": done, "tasks_failed": failed, "tasks_total": len(rs.graph.tasks),
                "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(rs.updated_at)),
                "prompt_tokens": rs.totals.get("prompt_tokens", 0),
                "completion_tokens": rs.totals.get("completion_tokens", 0),
                "tool_calls": rs.totals.get("tool_calls", 0),
            })
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out
