"""Sandboxed tool execution for worker sessions.

All file paths are rooted at the project directory. Bash is deny-by-default:
only command prefixes present in the configured allowlist may run.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ToolError(RuntimeError):
    pass


@dataclass
class ToolResult:
    ok: bool
    output: str
    duration: float = 0.0
    name: str = ""
    args: dict[str, Any] = None

    def to_message(self) -> dict[str, Any]:
        status = "ok" if self.ok else "error"
        return {"status": status, "output": self.output,
                "duration_s": round(self.duration, 3)}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class ToolRegistry:
    def __init__(self, root: Path, *, bash_allow: list[str], bash_deny: list[str],
                 bash_timeout: int, bash_max_output: int, read_max_bytes: int,
                 allow_network: bool):
        self._root = root.resolve()
        self._bash_allow = [a.strip() for a in bash_allow if a.strip()]
        self._bash_deny = [d.strip() for d in bash_deny if d.strip()]
        self._bash_timeout = bash_timeout
        self._bash_max_output = bash_max_output
        self._read_max_bytes = read_max_bytes
        self._allow_network = allow_network
        self._schemes: dict[str, dict[str, Any]] = {
            "read_file": _t("Read a text file from the project (relative to root).",
                            {"path": "str", "limit": "int? lines starting at 1"},
                            {"type": "object", "properties": {
                                "path": {"type": "string", "description": "file path relative to project root"},
                                "offset": {"type": "integer", "description": "starting line, 1-based"},
                                "limit": {"type": "integer", "description": "max lines to return (default 2000)"}},
                             "required": ["path"]}),
            "write_file": _t("Create or overwrite a file with the given content. Parent dirs are created.",
                             {"path": "str", "content": "str"},
                             {"type": "object", "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}},
                             "required": ["path", "content"]}),
            "edit_file": _t("Replace an exact string in a file. old_string must occur exactly once.",
                            {"path": "str", "old_string": "str", "new_string": "str"},
                             {"type": "object", "properties": {
                                 "path": {"type": "string"},
                                 "old_string": {"type": "string"},
                                 "new_string": {"type": "string"}},
                              "required": ["path", "old_string", "new_string"]}),
            "list_dir": _t("List a directory (non-recursive).",
                           {"path": "str? default '.'"},
                           {"type": "object", "properties": {
                               "path": {"type": "string", "description": "dir relative to root, default '.'"}},
                            }),
            "list_files": _t("Recursively list files under a path.",
                             {"path": "str? default '.'"},
                             {"type": "object", "properties": {
                                 "path": {"type": "string", "description": "dir relative to root, default '.'"}},
                            }),
            "find_file": _t("Find files by glob pattern (e.g. 'src/**/*.py'). Returns matching paths.",
                            {"pattern": "str"},
                             {"type": "object", "properties": {
                                 "pattern": {"type": "string", "description": "glob pattern, e.g. '**/*.py'"}},
                              "required": ["pattern"]}),
            "grep": _t("Search file contents with a regex. Optionally filter files by glob.",
                       {"pattern": "str", "path": "str?", "include": "str?"},
                       {"type": "object", "properties": {
                           "pattern": {"type": "string", "description": "regex to search for"},
                           "path": {"type": "string", "description": "dir to search, default project root"},
                           "include": {"type": "string", "description": "glob filter, e.g. '*.py'"}},
                        "required": ["pattern"]}),
            "bash": _t("Run an allowed command directly inside the project directory. "
                       "Only whitelisted commands are allowed (see 'bash_allow' in config; "
                       "by default nothing is allowed without explicit allowlist entry). "
                       "This is a command filter, not OS isolation. Command output is truncated.",
                       {"command": "str", "timeout_s": "int?"},
                       {"type": "object", "properties": {
                           "command": {"type": "string", "description": "shell command to execute"},
                           "timeout_s": {"type": "integer", "description": "override timeout (max 600s)"}},
                        "required": ["command"]}),
            "finish": _t("Call when the subtask is COMPLETE to end the session. "
                         "Provide a concise result summary, list of files created/changed, "
                         "any blockers, and hints for the next subtask(s).",
                         {"status": "done|blocked|incomplete",
                          "result": "str  short summary of what was accomplished",
                          "files": "str[] file paths created or modified",
                          "blockers": "str? what is preventing completion, if any",
                          "next_hints": "str? guidance for follow-up subtasks"},
                         {"type": "object", "properties": {
                             "status": {"type": "string", "enum": ["done", "blocked", "incomplete"]},
                             "result": {"type": "string"},
                             "files": {"type": "array", "items": {"type": "string"}},
                             "blockers": {"type": "string"},
                             "next_hints": {"type": "string"}},
                          "required": ["status", "result"]}),
        }
        self._impls: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "list_dir": self._list_dir,
            "list_files": self._list_files,
            "find_file": self._find_file,
            "grep": self._grep,
            "bash": self._bash,
            "finish": self._finish,
        }

    # -- sandbox helpers -----------------------------------------------------
    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            p = self._root / p
        p = p.resolve()
        if not _is_within(p, self._root):
            raise ToolError(f"path escapes project sandbox: {raw} (root={self._root})")
        return p

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self._root))
        except ValueError:
            return str(p)

    # -- file tools ------------------------------------------------------------
    def _read_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve(args["path"])
        limit = int(args.get("limit") or 2000)
        offset = max(1, int(args.get("offset") or 1))
        if not path.is_file():
            raise ToolError(f"not a file: {args['path']}")
        with path.open("rb") as f:
            data = f.read(self._read_max_bytes + 1)
        if len(data) > self._read_max_bytes:
            data = data[: self._read_max_bytes]
        text = data.decode("utf-8", "replace")
        lines = text.splitlines()
        truncated_lines = len(lines) > limit
        shown = lines[offset - 1:offset - 1 + limit]
        out = "\n".join(f"{i+offset}: {line}" for i, line in enumerate(shown))
        if truncated_lines:
            out += f"\n... ({len(lines) - limit} more lines)"
        if len(data) >= self._read_max_bytes:
            out += "\n... (file truncated at configured byte limit)"
        return ToolResult(True, out)

    def _write_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"wrote {len(content)} chars to {self._rel(path)}")

    def _edit_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve(args["path"])
        if not path.is_file():
            raise ToolError(f"not a file: {args['path']}")
        text = path.read_text(encoding="utf-8")
        old = str(args["old_string"])
        new = str(args["new_string"])
        count = text.count(old)
        if count == 0:
            raise ToolError("old_string not found in file")
        if count > 1:
            raise ToolError(f"old_string occurs {count} times; provide more context")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult(True, f"edited {self._rel(path)}")

    def _list_dir(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve(args.get("path") or ".")
        if not path.is_dir():
            raise ToolError(f"not a directory: {args.get('path', '.')}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return ToolResult(True, "\n".join(entries) or "(empty)")

    def _list_files(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve(args.get("path") or ".")
        if not path.is_dir():
            raise ToolError(f"not a directory: {args.get('path', '.')}")
        files = sorted(str(p.relative_to(self._root)) for p in path.rglob("*") if p.is_file())
        if len(files) > 500:
            files = files[:500]
            files.append("… (truncated at 500)")
        return ToolResult(True, "\n".join(files) or "(no files)")

    def _find_file(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"])
        try:
            matches = sorted(str(p.relative_to(self._root)) for p in self._root.glob(pattern) if p.is_file())
        except (ValueError, NotImplementedError) as e:
            raise ToolError(f"bad glob: {e}")
        if len(matches) > 500:
            matches = matches[:500]
            matches.append("… (truncated at 500)")
        return ToolResult(True, "\n".join(matches) or "(no matches)")

    def _grep(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"])
        include = args.get("include")
        base = self._resolve(args.get("path") or ".")
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ToolError(f"bad regex: {e}")
        if include:
            candidates = [p for p in base.rglob(include) if p.is_file()]
        else:
            candidates = list(base.rglob("*"))
        hits: list[str] = []
        checked = 0
        for p in candidates:
            if not p.is_file() or p.is_symlink():
                continue
            checked += 1
            if checked > 5000:
                raise ToolError("search too broad; narrow include/path")
            if p.stat().st_size > 512 * 1024:
                continue
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append(f"{Path(p).relative_to(self._root)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 500:
                        hits.append("... (truncated at 500 hits)")
                        return ToolResult(True, "\n".join(hits))
        return ToolResult(True, "\n".join(hits) if hits else "(no matches)")

    # -- bash --------------------------------------------------------------
    def _bash(self, args: dict[str, Any]) -> ToolResult:
        command = str(args["command"]).strip()
        if not command:
            raise ToolError("empty command")
        if self._is_denied(command):
            raise ToolError(
                f"command not allowed. Allowlist: {self._bash_allow or '(empty — add entries to bash_allow in .bola.toml)'} "
                f"Network is {'enabled' if self._allow_network else 'disabled'}."
            )
        argv = shlex.split(command)
        timeout = int(args.get("timeout_s") or self._bash_timeout)
        timeout = min(max(1, timeout), 600)
        import time
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self._root),
                capture_output=True, text=True,
                timeout=timeout,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(Path.home()),
                     "LANG": "C.UTF-8"},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"command timed out after {timeout}s")
        duration = time.monotonic() - started
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        if len(out) > self._bash_max_output:
            out = out[: self._bash_max_output] + "\n... (output truncated)"
        body = out.strip() or "(no output)"
        if proc.returncode != 0:
            return ToolResult(False, f"exit {proc.returncode}\n{body}", duration=duration)
        return ToolResult(True, body, duration=duration)

    def _is_denied(self, command: str) -> bool:
        """Deny-by-default: command is allowed only if it matches an
        allowlist entry, and then must not hit any deny pattern."""
        if any(c in command for c in (";", "|", "&", "`", "$", "\n", "\r", "<", ">")):
            return True
        try:
            argv = shlex.split(command)
        except ValueError:
            return True
        if not argv or "/" in argv[0] or argv[0] in ("bash", "sh", "sudo", "env"):
            return True
        lowered = command.lower()
        first = lowered.split()[0].split("/")[-1] if lowered.split() else ""
        # hard denials apply even when a command is allowlisted
        for d in self._bash_deny:
            if d and d in lowered:
                return True
        if not self._allow_network and (
            first in ("curl", "wget", "nc", "ssh", "telnet")
            or (first == "git" and "push" in lowered.split())
        ):
            return True
        if not self._bash_allow:
            return True  # empty allowlist -> deny everything
        for entry in self._bash_allow:
            if _match_allow(entry, command):
                return False
        return True

    def _finish(self, args: dict[str, Any]) -> ToolResult:
        status = str(args.get("status") or "incomplete")
        if status not in ("done", "blocked", "incomplete"):
            raise ToolError("status must be one of done|blocked|incomplete")
        return ToolResult(True, "finished", args={"finish": args})

    # -- dispatch -------------------------------------------------------------
    def schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": name, **spec}}
                for name, spec in self._schemes.items()]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        import time
        impl = self._impls.get(name)
        if impl is None:
            return ToolResult(False, f"unknown tool: {name}", name=name, args=args)
        t0 = time.monotonic()
        try:
            res = impl(args if isinstance(args, dict) else {})
            res.name = name
            res.args = args if isinstance(args, dict) else {}
            res.duration = res.duration or (time.monotonic() - t0)
            return res
        except ToolError as e:
            return ToolResult(False, str(e), duration=time.monotonic() - t0, name=name, args=args)
        except Exception as e:
            return ToolResult(False, f"tool error: {e}", duration=time.monotonic() - t0, name=name, args=args)


def _t(description: str, _params: Any, schema: dict[str, Any]) -> dict[str, Any]:
    return {"description": description, "parameters": schema}


def _match_allow(entry: str, command: str) -> bool:
    """Entry matches when it is the program name, or a prefix of the
    (stripped) command. Trailing '*' is a wildcard prefix (e.g. 'cargo *')."""
    entry = entry.strip()
    cmd = command.strip()
    if not entry:
        return False
    if entry.endswith("*"):
        prefix = entry[:-1].rstrip()
        if not prefix:
            return True
        return cmd == prefix or cmd.startswith(prefix + " ")
    prog = cmd.split()[0].split("/")[-1] if cmd.split() else ""
    return cmd == entry
