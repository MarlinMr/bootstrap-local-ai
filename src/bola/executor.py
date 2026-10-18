"""Sequential DAG execution with bounded continuation sessions."""
import time

from .logging import fmt_duration
from .session import Session
from .summarizer import update_note


def execute(config, state, log):
    state.outcome = "running"
    state.finished_at = 0
    state.save()
    try:
        order = state.graph.topological()
        for index, tid in enumerate(order, 1):
            task = state.graph.get(tid)
            if task.status == "done":
                continue
            if any(state.graph.get(d).status != "done" for d in task.depends_on):
                break
            state.current_task_id = tid
            log.info(f"[{index}/{len(order)}] {task.title}")
            for _ in range(config.max_sessions):
                task.status = "running"
                task.attempts += 1
                task.started_at = task.started_at or time.time()
                state.save()
                result = Session(config, state.state_note, log).run(task)
                task.result = result.result
                task.files = sorted(set(task.files + result.files))
                task.blockers = result.blockers
                task.next_hints = result.next_hints
                task.error = result.error
                task.turns += result.turns
                task.tool_calls += result.tool_calls
                task.prompt_tokens += result.prompt_tokens
                task.completion_tokens += result.completion_tokens
                state.add_totals(prompt_tokens=result.prompt_tokens,
                                 completion_tokens=result.completion_tokens,
                                 tool_calls=result.tool_calls, turns=result.turns)
                task.status = "done" if result.status == "done" else "failed"
                task.finished_at = time.time()
                state.save()
                state.state_note = update_note(config, state.state_note, task.title, result)
                state.save()
                log.info(f"{task.status}: {fmt_duration(task.finished_at-task.started_at)}, "
                         f"{task.turns} turns, {task.prompt_tokens} prompt / "
                         f"{task.completion_tokens} generated tokens, "
                         f"ctx high-water {result.high_water}/{config.num_ctx}")
                if task.status == "done" or result.status == "blocked" or result.error:
                    break
            if task.status != "done":
                break
        state.outcome = "done" if all(t.status == "done" for t in state.graph.tasks) else "failed"
        state.finished_at = time.time()
    except KeyboardInterrupt:
        state.outcome = "interrupted"
        log.warn("Interrupted. Resume restarts the unfinished subtask against current files.")
    except Exception:
        state.outcome = "failed"
        raise
    finally:
        state.save()
    return state.outcome == "done"
