"""Minimal Ollama /api/chat client over stdlib urllib, with SSE streaming,
tool calling, and accurate token accounting."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


class OllamaError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ChatReply:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    eval_count: int = 0
    prompt_eval_count: int = 0
    done: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Ollama:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=None if path in ("/api/tags", "/api/version") else data,
            headers={"Content-Type": "application/json"},
            method="GET" if path in ("/api/tags", "/api/version") else "POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise OllamaError(f"Ollama HTTP {e.code} on {path}: {body[:500]}", e.code, body) from e
        except urllib.error.URLError as e:
            raise OllamaError(f"Cannot reach Ollama at {url}: {e.reason}") from e

    # -- non-stream ----------------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        out = self._post("/api/tags", {})
        return out.get("models", []) if isinstance(out, dict) else []

    def version(self) -> str:
        out = self._post("/api/version", {})
        return out.get("version", "?") if isinstance(out, dict) else "?"

    def has_model(self, name: str) -> bool:
        return any(m.get("name") == name or m.get("name").startswith(name + ":")
                   for m in self.list_models())

    # -- streaming chat with tools ------------------------------------------
    def stream_chat(self, *, model: str, messages: Sequence[dict[str, Any]],
                    tools: Sequence[dict[str, Any]] | None = None,
                    num_ctx: int | None = None,
                    options: dict[str, Any] | None = None,
                    ) -> Iterator[dict[str, Any]]:
        """Yield incremental chunks from the Ollama chat endpoint (SSE-like,
        one JSON object per line). Last chunk carries `done: true` and token
        counts."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = list(tools)
        merged_opts = dict(options or {})
        if num_ctx is not None:
            merged_opts["num_ctx"] = num_ctx
        if merged_opts:
            payload["options"] = merged_opts

        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise OllamaError(f"Ollama HTTP {e.code} on /api/chat: {body[:800]}", e.code, body) from e
        except urllib.error.URLError as e:
            raise OllamaError(f"Cannot reach Ollama at {url}: {e.reason}") from e

        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    if obj.get("error"):
                        raise OllamaError(str(obj["error"]))
                    yield obj

    def chat(self, *, model: str, messages: Sequence[dict[str, Any]],
             tools: Sequence[dict[str, Any]] | None = None,
             num_ctx: int | None = None,
             options: dict[str, Any] | None = None) -> ChatReply:
        """Non-streaming convenience: consumes stream_chat into one reply."""
        reply = ChatReply()
        for chunk in self.stream_chat(model=model, messages=messages, tools=tools,
                                      num_ctx=num_ctx, options=options):
            msg = chunk.get("message") or {}
            if msg.get("content"):
                reply.content += msg["content"]
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    reply.tool_calls.append({"name": fn.get("name"),
                                             "arguments": fn.get("arguments", {})})
            if chunk.get("done"):
                reply.done = True
                reply.prompt_tokens = int(chunk.get("prompt_eval_count", 0) or 0)
                reply.completion_tokens = int(chunk.get("eval_count", 0) or 0)
                reply.prompt_eval_count = reply.prompt_tokens
                reply.eval_count = reply.completion_tokens
        return reply
