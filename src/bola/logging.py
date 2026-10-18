"""Live terminal rendering + append-only run.log for a run.

No daemon, no daemon threads: this is a plain class used from the foreground
CLI process. Thread-safe enough for our single-threaded flow (log writes go
through one lock).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any


class Logger:
    def __init__(self, log_path: Path | None = None, verbose: bool = False,
                 quiet: bool = False):
        self.log_path = log_path
        self.verbose = verbose
        self.quiet = quiet
        self._lock = threading.Lock()
        self._colors_supported = os.environ.get("NO_COLOR") is None and os.isatty(1)

    # -- color helpers ------------------------------------------------------
    def _c(self, s: str, code: str) -> str:
        if not self._colors_supported:
            return s
        return f"\033[{code}m{s}\033[0m"

    def info(self, s: str) -> None:
        if not self.quiet:
            print(self._c("[bola]", "36") + " " + s, flush=True)
        self._log(s)

    def ok(self, s: str) -> None:
        if not self.quiet:
            print(self._c("[ok  ]", "32") + " " + s, flush=True)
        self._log(s)

    def warn(self, s: str) -> None:
        if not self.quiet:
            print(self._c("[warn]", "33") + " " + s, flush=True)
        self._log(s)

    def err(self, s: str) -> None:
        print(self._c("[err ]", "31") + " " + s, file=sys.stderr, flush=True)
        self._log(s)

    def dim(self, s: str) -> None:
        if not self.quiet:
            print(self._c("      ", "2") + s, flush=True)
        self._log(s)

    def stream(self, chunk: str) -> None:
        """Print an incremental assistant-chunk (no newline) to the terminal
        only; the log gets the whole message later."""
        if self.quiet:
            return
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def stream_end(self) -> None:
        if not self.quiet:
            print(flush=True)

    def section(self, title: str) -> None:
        if self.quiet:
            return
        line = "=" * 60
        print(flush=True)
        print(self._c(line, "2"), flush=True)
        print(self._c(title, "1"), flush=True)
        print(self._c(line, "2"), flush=True)
        self._log(f"\n===== {title} =====")

    def kv(self, label: str, value: Any) -> None:
        if self.quiet:
            return
        print(f"{label:<18}: {value}", flush=True)
        self._log(f"{label}: {value}")

    def tool(self, name: str, args_brief: str, ok: bool, duration: float,
             out_brief: str = "") -> None:
        if not self.quiet:
            mark = self._c("+", "32") if ok else self._c("x", "31")
            print(f"{mark} {name} {args_brief}  ({duration:.1f}s) {out_brief}",
                  flush=True)
        self._log(f"TOOL {name} {args_brief} ok={ok} duration={duration:.3f} :: {out_brief}")

    def raw(self, label: str, payload: Any) -> None:
        if not self.verbose:
            return
        text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
        self._log(f"RAW {label}\n{text}")

    # -- log file -----------------------------------------------------------
    def _log(self, s: str) -> None:
        if not self.log_path:
            return
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(_dt.datetime.now().isoformat(timespec="milliseconds") + " " + s + "\n")


def _human(n: int | float) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def fmt_duration(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    return f"{m}m{s:02d}s"


def summarize_log(log_path: Path) -> list[str]:
    """Parse a run.log back into a compact table of events (for `bola report`)."""
    lines: list[str] = []
    if not log_path.is_file():
        return lines
    for line in log_path.read_text(encoding="utf-8").splitlines():
        lines.append(line)
    return lines
