"""Foreground command-line interface."""
import argparse
import json
from pathlib import Path

from .config import load_config
from .doctor import doctor
from .executor import execute
from .graph import TaskGraph
from .logging import Logger
from .ollama_client import OllamaError
from .planner import plan
from .state import RunState, run_id, scan_history


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bola", description="Bounded Ollama coding sessions")
    parser.add_argument("command", choices=["plan", "run", "resume", "status", "report", "history", "doctor"])
    parser.add_argument("task", nargs="?", help="task text, plan JSON path, or run id")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ollama-url")
    parser.add_argument("--model", dest="worker_model")
    parser.add_argument("--planner-model")
    parser.add_argument("--ctx", dest="num_ctx", type=int)
    parser.add_argument("--output-reserve", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--output", type=Path, help="plan output JSON")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="plan without executing tools")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--all", action="store_true", help="history across registered projects")
    args = parser.parse_args(argv)
    log = Logger(verbose=args.verbose, quiet=args.quiet)
    try:
        root = args.root.resolve()
        if not root.is_dir():
            raise ValueError(f"Project directory does not exist: {root}")
        config = load_config(root, vars(args))
        registry = Path.home() / ".local" / "share" / "bola" / "projects.json"
        if args.command == "doctor":
            return 0 if doctor(config, log) else 1
        if args.command == "history":
            roots = [root]
            if args.all and registry.exists():
                roots = [Path(p) for p in json.loads(registry.read_text())]
            print(json.dumps(scan_history(roots), indent=2))
            return 0
        if args.command in ("resume", "status", "report"):
            if args.task and (Path(args.task).name != args.task or args.task in (".", "..")):
                raise ValueError("Expected a run id")
            state = (RunState.load(root / ".bola" / "runs" / args.task / "state.json")
                     if args.task else RunState.latest(root))
            if state is None:
                raise ValueError("No runs found in this project")
            # A copied project must operate on its current location.
            state.root = root
            if args.command != "resume":
                print(f"Run {state.id}: {state.outcome}\n{state.graph.task}")
                for t in state.graph.tasks:
                    print(f"  {t.id:12} {t.status:10} {t.title}")
                    if args.command == "report":
                        print(f"    {t.result or t.error or t.blockers}")
                print(f"Totals: {state.totals}\nLog: {state.log_path}")
                return 0
        else:
            if not args.task:
                raise ValueError("Provide a task description or plan JSON path")
            candidate = Path(args.task)
            # Only treat .json arguments as plan paths; long task text isn't a filename.
            graph = (TaskGraph.load(candidate) if args.task.endswith(".json")
                     else plan(config, args.task, log))
            graph.topological()
            for tid in graph.topological():
                t = graph.get(tid)
                print(f"  {tid}: {t.title} (after: {', '.join(t.depends_on) or 'none'})")
            if args.command == "plan" or args.dry_run:
                output = args.output or root / "bola-plan.json"
                graph.save(output)
                print(f"Plan saved: {output}")
                return 0
            state = RunState(root=root, id=run_id(), graph=graph)
        if config.require_confirm and not args.yes:
            if input(f"Execute in {root}? [y/N] ").strip().lower() != "y":
                return 0
        registry.parent.mkdir(parents=True, exist_ok=True)
        known = set(json.loads(registry.read_text()) if registry.exists() else [])
        known.add(str(root))
        registry.write_text(json.dumps(sorted(known), indent=2))
        log.log_path = state.log_path
        log.info(f"Run {state.id}; log: {state.log_path}")
        return 0 if execute(config, state, log) else 1
    except (ValueError, OSError, OllamaError) as exc:
        log.err(str(exc))
        return 1
    except (KeyboardInterrupt, EOFError):
        log.warn("Cancelled")
        return 130
