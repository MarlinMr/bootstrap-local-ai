import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bola.cli import main
from bola.config import Config
from bola.context import payload_tokens
from bola.graph import TaskGraph, make_graph
from bola.logging import Logger
from bola.ollama_client import ChatReply, OllamaError
from bola.planning_chat import PlanningChat


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = Config(root=self.root)
        self.log = Logger(quiet=True)

    def test_discuss_revise_and_save_via_cli(self):
        prompts = []

        def generate(config, brief, log, snapshot=None):
            prompts.append(brief)
            return make_graph("Expense tracker", {"tasks": [
                {"id": "t1", "title": "Build tracker", "description":
                 "Use SQLite" if len(prompts) == 1 else "Use JSON files"}]})

        def stream(**kwargs):
            yield {"message": {"content": "What storage would you prefer?"}, "done": True}

        output = self.root / "plan.json"
        with patch("bola.planning_chat.plan", side_effect=generate), \
                patch("bola.planning_chat.Ollama.stream_chat", side_effect=stream), \
                patch("builtins.input", side_effect=["/plan", "Use JSON files instead of SQLite", "/save"]), \
                contextlib.redirect_stdout(io.StringIO()):
            code = main(["plan", "Build a tracker using SQLite", "--interactive",
                         "--root", str(self.root), "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 2)
        self.assertIn("Build a tracker using SQLite", prompts[1])
        self.assertIn("Use JSON files instead of SQLite", prompts[1])
        self.assertIn("Proposed plan for review", prompts[1])
        self.assertEqual(TaskGraph.load(output).tasks[0].description, "Use JSON files")
        resumed = PlanningChat(self.config, self.log, resume=True)
        self.assertFalse(resumed.data["dirty"])
        self.assertEqual(len(resumed.data["messages"]), 6)

    def test_context_compression_preserves_archive(self):
        self.config.num_ctx = 8192
        self.config.output_reserve = 1024
        chat = PlanningChat(self.config, self.log)
        messages = [{"role": "user" if i % 2 == 0 else "assistant",
                     "content": "Keep SQLite. " + "x" * 2500} for i in range(6)]
        chat.data["messages"] = list(messages)
        chat.data["active"] = list(messages)
        with patch.object(chat.client, "chat", return_value=ChatReply(
                content="Agreed requirement: keep SQLite.", done=True)) as summarize:
            chat.fit({"role": "user", "content": "y" * 2000})
        summarize.assert_called()
        self.assertIn("Keep SQLite", summarize.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(chat.data["messages"], messages)
        self.assertIn("keep SQLite", chat.data["summary"])
        self.assertLessEqual(payload_tokens(chat.messages({"role": "user", "content": "y" * 2000})),
                             chat.limit)
        resumed = PlanningChat(self.config, self.log, resume=True)
        self.assertEqual(resumed.data["summary"], chat.data["summary"])

    def test_failed_response_can_retry_without_duplicate_user_message(self):
        chat = PlanningChat(self.config, self.log)
        with patch.object(chat.client, "stream_chat", side_effect=OllamaError("offline")):
            with self.assertRaises(OllamaError):
                chat.ask("Build a tracker")
        resumed = PlanningChat(self.config, self.log, resume=True)
        self.assertTrue(resumed.data["pending"])
        with patch.object(resumed.client, "stream_chat", return_value=iter([
                {"message": {"content": "Which features?"}, "done": True}])):
            resumed.reply()
        self.assertFalse(resumed.data["pending"])
        self.assertEqual(len(resumed.data["messages"]), 2)

    def test_oversized_message_is_not_saved(self):
        chat = PlanningChat(self.config, self.log)
        with self.assertRaises(ValueError):
            chat.ask("x" * self.config.num_ctx * 4)
        self.assertEqual(chat.data["messages"], [])

    def test_quit_saves_and_does_not_execute_or_generate(self):
        with patch("builtins.input", side_effect=["/quit"]), \
                patch("bola.planning_chat.plan") as generate, \
                patch("bola.cli.execute") as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["plan", "-i", "--root", str(self.root)]), 0)
        generate.assert_not_called()
        execute.assert_not_called()
        self.assertFalse((self.root / "bola-plan.json").exists())
        self.assertTrue(list((self.root / ".bola" / "planning").glob("*/conversation.json")))
