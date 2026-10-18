# bola — Bootstrap Local AI

A foreground Python CLI that breaks coding tasks into a dependency graph and
executes focused Ollama worker sessions. Designed for limited context windows,
including a 32,768-token setup. Python 3.11+; no runtime dependencies.

## Install once, use in any project

From this checkout:

```sh
pipx install -e .
# Alternatively: uv tool install -e .
```

Then, in an existing or empty project directory:

```sh
bola doctor
bola plan "Build a small Python CLI for tracking expenses" --output plan.json
bola run plan.json
# Or plan and run in one step:
bola run "Add CSV export and verify it" --yes
```

For development without installing:

```sh
PYTHONPATH=/path/to/bootstrap-local-ai/src python3 -m bola --help
```

`--root /path/to/project` also selects the target directory.

## Progress and recovery

Assistant text streams to the terminal. Tool names, results, durations, subtask
progress and prompt high-water counts are shown. Full tool results and assistant
text go to `.bola/runs/<id>/run.log`; state and task graphs are saved alongside it.
Add `.bola/` to each target project's `.gitignore`.

```sh
bola status
bola report
bola resume --yes
bola history
bola history --all
```

`--all` reads a registry of projects previously executed on this machine, stored
in `~/.local/share/bola/projects.json`. There is no daemon. Ctrl-C saves run state;
resume **restarts the unfinished subtask**, using the files already on disk.
It does not replay a partially completed tool call or restore a live conversation.
Use `--quiet` for reduced screen output.

## Configuration and portability

Precedence: CLI flags → `BOLA_*` environment → project `.bola.toml` →
`~/.config/bola/config.toml` → defaults. See `examples/bola.toml`.

```sh
bola run "Implement search" --model another-coder:latest \
  --planner-model another-model:latest --ctx 16384 --output-reserve 2048 \
  --ollama-url http://other-machine:11434 --max-turns 12 --max-sessions 3
```

Supported environment variables: `BOLA_OLLAMA_URL`, `BOLA_PLANNER_MODEL`,
`BOLA_WORKER_MODEL`, `BOLA_NUM_CTX`, `BOLA_OUTPUT_RESERVE`, `BOLA_BASH_TIMEOUT`,
`BOLA_ALLOW_NETWORK`. Model defaults are suggestions; install/configure the models
available on your machine. `doctor` checks connectivity and model presence; it
does not predict whether a context size fits VRAM.

## Context strategy

The planner sees a bounded file listing and proposes small dependent subtasks.
Each worker has file-read/write/edit/search tools and a `finish` tool for a
structured handoff. Sessions reserve output and tool-schema headroom, estimate
input size before sending, and record Ollama's actual token counts afterward.
Near the input limit (or turn limit), one final wrap-up request allows only
`finish`. Incomplete subtasks get fresh sessions, up to `max_sessions` per
invocation. Blocked subtasks stop the run. Completed dependencies are preserved.

Carry-forward notes preserve decisions, file paths, blockers and next steps;
long notes are compressed. Full results remain in run state. Estimates are not
an exact tokenizer, so choose headroom appropriate to your model and workload.
This first version uses a sequential, fixed DAG; it does not automatically replan.

## Command policy

Commands are denied by default. Configure `bash_allow` with exact commands or
explicit trailing-`*` prefixes (e.g. `"pytest *"`). Shell operators, substitutions,
redirections, and shell interpreters are rejected. Commands run directly as
argument vectors, with timeouts and captured output. Built-in file tools check
resolved paths against the project root.

This is **not an OS security sandbox**: an allowed interpreter or test runner can
execute project code, access outside files, or use the network. `allow_network =
false` blocks recognized network commands, not arbitrary network system calls.
Use narrow command entries and an external container if you need isolation.

## Checks

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m bola --help
```
