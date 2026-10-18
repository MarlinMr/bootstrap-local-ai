"""Compact, bounded handoffs; disk state retains full per-task results."""
from .config import Config
from .context import estimate_tokens
from .ollama_client import Ollama


def update_note(config: Config, previous: str, title: str, result) -> str:
    note = (f"{previous}\n\n## {title} ({result.status})\n{result.result}\n"
            f"Files: {', '.join(result.files)}\nBlockers: {result.blockers}\n"
            f"Next: {result.next_hints}").strip()
    if estimate_tokens(note) <= config.state_note_max:
        return note
    output_limit = min(config.state_note_max, config.output_reserve)
    input_chars = max(1, (config.num_ctx - output_limit - 1000) * 3)
    reply = Ollama(config.ollama_url).chat(
        model=config.worker_model, num_ctx=config.num_ctx,
        messages=[{"role": "system", "content":
                   "Compress the project notes. Preserve file paths, decisions, blockers, "
                   "verification results and unfinished work. Do not invent facts."},
                  {"role": "user", "content": note[-input_chars:]}],
        options={"num_predict": output_limit, "temperature": 0.1})
    return (reply.content or note)[-int(config.state_note_max * 3.2):]
