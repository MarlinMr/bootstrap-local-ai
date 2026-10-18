import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bola.config import Config, load_config
from bola.context import Budget, assemble
from bola.executor import execute
from bola.graph import make_graph
from bola.logging import Logger
from bola.session import Session
from bola.state import RunState
from bola.tools import ToolRegistry


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = load_config(self.root)

    def tools(self, allow=None):
        return ToolRegistry(self.root, bash_allow=allow or [], bash_deny=[],
                            bash_timeout=1, bash_max_output=1000,
                            read_max_bytes=1000, allow_network=False)

    def test_command_policy(self):
        tools = self.tools(["python3 -m unittest *"])
        for cmd in ["python3 -c 'print(1)'", "python3 -m unittest; touch /tmp/x",
                    "python3 -m unittest $(id)", "python3 -m unittestx"]:
            self.assertTrue(tools._is_denied(cmd), cmd)
        self.assertFalse(tools._is_denied("python3 -m unittest discover"))
        self.assertFalse(self.tools(["git status"])._is_denied("git status"))
        self.assertTrue(self.tools()._is_denied("git status"))

    def test_file_escape_and_symlink(self):
        tools = self.tools()
        self.assertFalse(tools.execute("write_file", {"path": "../escape", "content": "bad"}).ok)
        (self.root / "link").symlink_to(self.root.parent, target_is_directory=True)
        self.assertFalse(tools.execute("write_file", {"path": "link/escape", "content": "bad"}).ok)

    def test_graph_validation(self):
        for tasks in [
            [{"id": "a", "title": "A", "depends_on": ["b"]}],
            [{"id": "a", "title": "A"}, {"id": "a", "title": "B"}],
            [{"id": "a", "title": "A", "depends_on": ["a"]}],
        ]:
            with self.assertRaises(ValueError):
                make_graph("test", {"tasks": tasks})

    def test_budget_rejects_oversized_brief_and_orphan_results(self):
        with self.assertRaises(ValueError):
            assemble(Budget(100, 20, 10), system="x"*400, brief="y"*400,
                     state_note="", history=[])
        pc = assemble(Budget(100, 20, 10), system="", brief="", state_note="",
                      history=[{"role": "assistant", "content": "x"*1000},
                               {"role": "tool", "content": "ok"}])
        self.assertEqual(pc.history, [])

    def test_forced_wrapup_and_execution_resume(self):
        graph = make_graph("demo", {"tasks": [{"id": "a", "title": "create"},
                         {"id": "b", "title": "verify", "depends_on": ["a"]}]})
        self.cfg.max_turns = 1
        calls = []

        def stream(**kwargs):
            calls.append(kwargs)
            forced = len(kwargs["tools"]) == 1
            name = "finish" if forced else "write_file"
            args = ({"status": "done", "result": "created", "files": ["hello.txt"]}
                    if forced else {"path": "hello.txt", "content": "hello"})
            yield {"message": {"tool_calls": [{"function": {"name": name, "arguments": args}}]},
                   "done": True, "prompt_eval_count": 100, "eval_count": 10}

        state = RunState(self.root, "test", graph)
        with patch("bola.session.Ollama.stream_chat", side_effect=stream):
            self.assertTrue(execute(self.cfg, state, Logger(quiet=True)))
            loaded = RunState.load(state.path)
            self.assertEqual(loaded.outcome, "done")
            self.assertTrue(execute(self.cfg, loaded, Logger(quiet=True)))
        self.assertEqual(len(calls), 4)
        self.assertEqual((self.root / "hello.txt").read_text(), "hello")
        self.assertEqual(loaded.totals["turns"], 4)
        self.assertEqual(calls[1]["tools"][0]["function"]["name"], "finish")

    def test_early_context_wrapup_is_one_turn(self):
        task = make_graph("demo", {"tasks": [{"id": "a", "title": "Inspect"}]}).tasks[0]
        self.cfg.max_turns = 10
        calls = []

        def stream(**kwargs):
            calls.append(kwargs)
            yield {"message": {"tool_calls": [{"function": {
                "name": "list_dir", "arguments": {}}}]}, "done": True,
                "prompt_eval_count": 30000, "eval_count": 10}

        with patch("bola.session.Ollama.stream_chat", side_effect=stream):
            result = Session(self.cfg, "", Logger(quiet=True)).run(task)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result.finished)
        self.assertEqual([s["function"]["name"] for s in calls[-1]["tools"]], ["finish"])

    def test_config_precedence(self):
        (self.root / ".bola.toml").write_text("num_ctx = 12000\n")
        with patch.dict("os.environ", {"BOLA_NUM_CTX": "16000"}):
            self.assertEqual(load_config(self.root).num_ctx, 16000)
            self.assertEqual(load_config(self.root, {"num_ctx": 24000}).num_ctx, 24000)


if __name__ == "__main__":
    unittest.main()
