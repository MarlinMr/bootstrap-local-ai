"""Context-window budgeting.

A session request is assembled in a fixed importance order so that when the
budget is tight, low-value content is dropped first:

    1. system prompt           (never dropped)
    2. subtask brief           (never dropped)
    3. carry-forward state note (never dropped; summarized to fit)
    4. recent tool observations (oldest dropped first)
    5. earlier assistant/user text (dropped before 4)

Sessions record Ollama's prompt_eval_count / eval_count after requests. A local
BPE-free estimate is used for assembly; this is not an exact tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# rough chars-per-token for mixed code/chat text with typical BPE tokenizers
EST_CHARS_PER_TOKEN = 3.2


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / EST_CHARS_PER_TOKEN)) if text else 1


def message_tokens(msg: dict[str, Any]) -> int:
    t = estimate_tokens(str(msg.get("content") or ""))
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        t += estimate_tokens(str(fn.get("name", "")) + str(fn.get("arguments", "")))
    return t


def payload_tokens(messages: Sequence[dict[str, Any]]) -> int:
    return sum(message_tokens(m) for m in messages)


@dataclass
class Budget:
    total_ctx: int
    output_reserve: int
    state_note_max: int

    @property
    def input_budget(self) -> int:
        return self.total_ctx - self.output_reserve


@dataclass
class PackedContext:
    system: str
    brief: str
    state_note: str
    history: list[dict[str, Any]] = field(default_factory=list)
    dropped: int = 0                      # number of history messages dropped
    est_tokens: int = 0
    high_water: int = 0                   # highest measured prompt token count


def assemble(budget: Budget, *, system: str, brief: str, state_note: str,
             history: list[dict[str, Any]]) -> PackedContext:
    """Fit {system, brief, state_note, history} into budget.input_budget,
    dropping the OLDEST history messages first. system/brief/state_note are
    never dropped, but state_note is clipped if it alone would blow the budget."""
    limit = budget.input_budget
    base = estimate_tokens(system) + estimate_tokens(brief)
    if estimate_tokens(state_note) > budget.state_note_max:
        state_note = state_note[-int(budget.state_note_max * EST_CHARS_PER_TOKEN):]
    note_tokens = estimate_tokens(state_note)
    # hard-clip the state note if it alone exceeds budget
    if base + note_tokens > limit:
        allowed_note = max(200, limit - base - 200)
        # clip by estimated tokens
        keep = max(0, int(allowed_note * EST_CHARS_PER_TOKEN))
        if len(state_note) > keep:
            state_note = state_note[:keep] + "\n[...note truncated to fit budget...]"
            note_tokens = estimate_tokens(state_note)
    if base + estimate_tokens(state_note) > limit:
        raise ValueError("Subtask brief and state exceed context budget; split the task further")
    remaining = limit - base - estimate_tokens(state_note)
    packed: list[dict[str, Any]] = []
    used = 0
    # keep newest-first while they fit
    for msg in reversed(history):
        cost = message_tokens(msg)
        if used + cost > remaining:
            break
        packed.append(msg)
        used += cost
    dropped = len(history) - len(packed)
    packed.reverse()
    # Never send orphaned tool results without the assistant's tool calls.
    while packed and packed[0].get("role") == "tool":
        used -= message_tokens(packed.pop(0))
    dropped = len(history) - len(packed)
    return PackedContext(system=system, brief=brief, state_note=state_note,
                         history=packed, dropped=dropped,
                         est_tokens=base + estimate_tokens(state_note) + used)


def to_messages(pc: PackedContext) -> list[dict[str, Any]]:
    sys_text = pc.system if not pc.state_note else (
        f"{pc.system}\n\n# PROJECT STATE (carry-forward note from earlier subtasks)\n{pc.state_note}"
    )
    msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_text}]
    msgs.append({"role": "user", "content": pc.brief})
    msgs.extend(pc.history)
    return msgs
