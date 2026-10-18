"""One agentic worker session for a single subtask.

The session owns a small chat history. Each turn:
  1. Pack {system, brief, state_note, history} into the context budget
     (oldest items dropped first) -> one /api/chat request.
  2. Consume the stream: assistant text + tool calls.
  3. Execute non-finish tools, append tool results to history, repeat.
  4. Stop on a `finish` tool call, on max_turns, or when the context budget
     is nearly exhausted (in which case we force a wrap-up turn).

Only the compact carry-forward state note + this subtask's own small history
travel between requests — the full project never has to fit in one context.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .context import Budget, assemble, to_messages, estimate_tokens
from .graph import Subtask
from .logging import Logger
from .ollama_client import Ollama, OllamaError
from .tools import ToolRegistry, ToolResult

WORKER_SYSTEM = (
    "You are a focused software-engineering agent working on ONE small subtask "
    "inside a larger project. Use the tools to inspect the code, make the "
    "change, and verify it.\n"
    "Rules:\n"
    "- Work only inside the project directory; prefer minimal, correct edits.\n"
    "- Read only what you need: your context window is small, so be economical "
    "with file reads and long outputs.\n"
    "- When done (or when you are confident the subtask is complete), call the "
    "finish tool with status, a concise result summary, the files you touched, "
    "any blockers, and hints for later subtasks.\n"
    "- If blocked, call finish with status=blocked and explain precisely what "
    "is missing.\n"
    "- Do not invent file contents you have not read."
)


@dataclass
class SessionResult:
    task_id: str
    status: str = "incomplete"      # done|blocked|incomplete
    result: str = ""
    files: list[str] = field(default_factory=list)
    blockers: str = ""
    next_hints: str = ""
    turns: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    high_water: int = 0             # max measured prompt tokens
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    finished: bool = False


class Session:
    def __init__(self, config: Config, state_note: str, log: Logger):
        self.cfg = config
        self.log = log
        self.budget = Budget(total_ctx=config.num_ctx,
                             output_reserve=config.output_reserve + 2500,
                             state_note_max=config.state_note_max)
        self.tools = ToolRegistry(
            config.root,
            bash_allow=config.bash_allow, bash_deny=config.bash_deny,
            bash_timeout=config.bash_timeout, bash_max_output=config.bash_max_output,
            read_max_bytes=config.read_max_bytes, allow_network=config.allow_network,
        )
        self.client = Ollama(config.ollama_url)
        schema_tokens = estimate_tokens(json.dumps(self.tools.schemas()))
        self.budget.output_reserve = config.output_reserve + schema_tokens + 512
        self._state_note = state_note
        self._history: list[dict[str, Any]] = []
        self._high_water = 0

    # -- internals -----------------------------------------------------------
    def _append_assistant(self, content: str, tool_calls: list[dict[str, Any]]) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for i, tc in enumerate(tool_calls)
            ]
        self._history.append(msg)

    def _append_tool_results(self, results: list[ToolResult]) -> None:
        for i, r in enumerate(results):
            self._history.append({
                "role": "tool",
                "content": json.dumps(r.to_message()),
            })

    def _overflow_flag(self, measured: int) -> bool:
        return measured > self.budget.input_budget * 0.92

    # -- main loop -----------------------------------------------------------
    def run(self, subtask: Subtask) -> SessionResult:
        res = SessionResult(task_id=subtask.id)
        self.log.section(f"Subtask {subtask.id}: {subtask.title}")
        self.log.kv("brief", (subtask.description or subtask.title)[:400])
        if self._state_note:
            self.log.dim(f"state note: {len(self._state_note)} chars")
        max_turns = self.cfg.max_turns
        overflow = False
        for turn in range(1, max_turns + 2):
            forced = overflow or turn > max_turns
            if forced:
                self._history.append({
                    "role": "user",
                    "content": ("SYSTEM: context budget is nearly exhausted. "
                                "STOP working. Immediately call the finish tool "
                                "with your best summary of where things stand "
                                "(status, result, files, blockers, next_hints)."),
                })
            brief = self._brief(subtask)
            system = WORKER_SYSTEM + "\nAllowed command patterns: " + repr(self.cfg.bash_allow)
            if not self.cfg.bash_allow:
                system += "\nNo commands are allowed. Use file tools; do not call bash."
            pc = assemble(self.budget, system=system,
                          brief=brief, state_note=self._state_note,
                          history=self._history)
            messages = to_messages(pc)
            if pc.dropped:
                self.log.dim(f"context assembled: dropped {pc.dropped} oldest history "
                             f"messages (est {pc.est_tokens} tok)")
            self.log.dim(f"turn {turn}/{max_turns}" + (" (wrap-up)" if forced else "") +
                         " — requesting model …")
            try:
                content = ""
                tool_calls: list[dict[str, Any]] = []
                measured_prompt = 0
                measured_completion = 0
                for chunk in self.client.stream_chat(
                        model=self.cfg.worker_model, messages=messages,
                        tools=[s for s in self.tools.schemas() if not forced or
                               s["function"]["name"] == "finish"],
                        num_ctx=self.cfg.num_ctx,
                        options={"temperature": 0.2, "num_predict": self.cfg.output_reserve}):
                    msg = chunk.get("message") or {}
                    c = msg.get("content")
                    if c:
                        self.log.stream(c)
                        content += c
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function", {})
                        args = fn.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        if not isinstance(args, dict):
                            args = {}
                        tool_calls.append({"name": fn.get("name", ""), "arguments": args})
                    if chunk.get("done"):
                        measured_prompt = int(chunk.get("prompt_eval_count", 0) or 0)
                        measured_completion = int(chunk.get("eval_count", 0) or 0)
                        break
                self.log.stream_end()
            except OllamaError as e:
                res.error = str(e)
                self.log.err(str(e))
                break
            res.turns = turn
            res.prompt_tokens += measured_prompt
            res.completion_tokens += measured_completion
            self._high_water = max(self._high_water, measured_prompt)
            res.high_water = self._high_water

            self._append_assistant(content, tool_calls)
            self.log._log(f"ASSISTANT {content}")
            if forced and not tool_calls:
                res.result = content
                break
            if not tool_calls:
                if content.strip():
                    self.log.dim("model produced no tool call but gave text; "
                                 "nudging it to use the finish tool")
                    self._history.append({
                        "role": "user",
                        "content": "Remember: end the subtask by calling the finish tool.",
                    })
                    continue
                self.log.warn("empty model turn; ending session")
                break

            results: list[ToolResult] = []
            for tc in tool_calls:
                name = tc["name"]
                if forced and name != "finish":
                    continue
                brief_repr = _args_brief(tc["arguments"])
                self.log.dim(f"> tool: {name} {brief_repr}")
                t0 = time.monotonic()
                r = self.tools.execute(name, tc["arguments"])
                res.tool_calls += 1
                out_brief = (r.output[:120].replace("\n", " ")) + ("…" if len(r.output) > 120 else "")
                self.log.tool(name, brief_repr, r.ok, time.monotonic() - t0,
                              out_brief if not self.log.verbose else r.output[:500])
                res.tool_log.append({"tool": name, "ok": r.ok,
                                     "duration": round(time.monotonic() - t0, 3)})
                self.log._log(f"TOOL RESULT {name}: {r.output}")
                if name == "finish" and r.ok:
                    fa = tc["arguments"]
                    res.status = fa.get("status", "incomplete")
                    res.result = fa.get("result", "")
                    res.files = list(fa.get("files") or [])
                    res.blockers = fa.get("blockers", "")
                    res.next_hints = fa.get("next_hints", "")
                    res.finished = True
                    self._append_tool_results([r])
                    return res
                results.append(r)
            self._append_tool_results(results)
            if forced:
                break
            if measured_prompt and self._overflow_flag(measured_prompt):
                self.log.warn(f"prompt tokens {measured_prompt} ≈ budget "
                              f"{self.budget.input_budget}; will force a wrap-up turn")
                overflow = True

        if not res.result:
            res.result = "Session ended without a structured handoff; inspect current files and run log."
        self.log.warn("Session ended without a successful finish call")
        return res

    @staticmethod
    def _brief(subtask: Subtask) -> str:
        base = (f"# YOUR SUBTASK\n## {subtask.id}: {subtask.title}\n\n"
                f"{subtask.description}\n\n"
                "Inspect the relevant code, make the change, verify as feasible, "
                "then call finish with status/result/files/next_hints.")
        return base


def _args_brief(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts = []
    for k, v in list(args.items())[:4]:
        s = str(v)
        if len(s) > 60:
            s = s[:60] + "…"
        s = s.replace("\n", " ")
        parts.append(f"{k}={s}")
    return " ".join(parts)
