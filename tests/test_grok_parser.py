"""Tests for parse_grok_session and Grok detection in read_session.

Grok session format (Grok CLI):
  - One directory per session: ~/.grok/sessions/<percent-encoded-cwd>/<uuid>/
  - Transcript at chat_history.jsonl; entries are {type, content} with no
    timestamps of their own.
  - Optional sibling summary.json supplies cwd, generated_title and created_at.
  - Entries carrying "synthetic_reason" are harness context injected into the
    turn list, not anything the user or the model said.

Fixtures are generated as strings inside tmpdir — no fixture files committed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

# Make scripts/ importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import recall  # noqa: E402
import read_session  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────


def write_session(root, entries, cwd="/home/u/project", uuid="0199-abc", summary=None):
    """Lay out one Grok session directory and return its transcript path."""
    session_dir = Path(root) / quote(cwd, safe="") / uuid
    session_dir.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (session_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    path = session_dir / "chat_history.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(path)


CONVERSATION = [
    {"type": "user", "content": "why does the build fail on arm"},
    {"type": "assistant", "content": "the toolchain is pinned to x86"},
    {"type": "reasoning", "content": "considering the toolchain"},
    {"type": "tool_result", "content": "exit 1"},
    {"type": "user", "content": "<system-reminder>injected</system-reminder>"},
    {"type": "user", "content": "harness noise", "synthetic_reason": "context"},
    {"type": "assistant", "content": ""},
]


class ParseGrokSession(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def test_keeps_only_user_and_assistant_text(self):
        path = write_session(self.root, CONVERSATION)
        _, messages = recall.parse_grok_session(path)
        self.assertEqual(messages, [
            ("user", "why does the build fail on arm"),
            ("assistant", "the toolchain is pinned to x86"),
        ])

    def test_skips_synthetic_harness_entries(self):
        path = write_session(self.root, CONVERSATION)
        _, messages = recall.parse_grok_session(path)
        self.assertNotIn("harness noise", [text for _, text in messages])

    def test_session_id_and_source_come_from_the_directory(self):
        path = write_session(self.root, CONVERSATION, uuid="0199-def")
        metadata, _ = recall.parse_grok_session(path)
        self.assertEqual(metadata["session_id"], "0199-def")
        self.assertEqual(metadata["source"], "grok")
        self.assertEqual(metadata["file_path"], path)

    def test_project_is_decoded_from_the_parent_directory(self):
        path = write_session(self.root, CONVERSATION, cwd="/home/u/my project")
        metadata, _ = recall.parse_grok_session(path)
        self.assertEqual(metadata["project"], "/home/u/my project")

    def test_summary_supplies_project_title_and_time(self):
        path = write_session(self.root, CONVERSATION, summary={
            "info": {"cwd": "/srv/app"},
            "generated_title": "arm build failure",
            "created_at": "2026-05-01T12:00:00.000Z",
        })
        metadata, _ = recall.parse_grok_session(path)
        self.assertEqual(metadata["project"], "/srv/app")
        self.assertEqual(metadata["slug"], "arm build failure")
        self.assertEqual(metadata["timestamp"],
                         recall.parse_iso_timestamp("2026-05-01T12:00:00.000Z"))

    def test_git_root_is_used_when_summary_has_no_cwd(self):
        path = write_session(self.root, CONVERSATION,
                             summary={"git_root_dir": "/srv/repo"})
        metadata, _ = recall.parse_grok_session(path)
        self.assertEqual(metadata["project"], "/srv/repo")

    def test_slug_falls_back_to_the_session_id(self):
        path = write_session(self.root, CONVERSATION, uuid="0199abcdef123456")
        metadata, _ = recall.parse_grok_session(path)
        self.assertEqual(metadata["slug"], "0199abcdef12")

    def test_a_corrupt_summary_does_not_stop_the_parse(self):
        path = write_session(self.root, CONVERSATION)
        (Path(path).parent / "summary.json").write_text("{not json", encoding="utf-8")
        metadata, messages = recall.parse_grok_session(path)
        self.assertEqual(len(messages), 2)
        self.assertEqual(metadata["project"], "/home/u/project")

    def test_a_corrupt_line_costs_only_that_line(self):
        path = write_session(self.root, CONVERSATION)
        with open(path, "a", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write(json.dumps({"type": "user", "content": "after the corruption"}) + "\n")
        _, messages = recall.parse_grok_session(path)
        self.assertIn("after the corruption", [text for _, text in messages])

    def test_an_empty_transcript_parses(self):
        path = write_session(self.root, [])
        metadata, messages = recall.parse_grok_session(path)
        self.assertEqual(messages, [])
        self.assertEqual(metadata["timestamp"], 0)

    def test_a_missing_file_is_reported_not_raised(self):
        self.assertIsNone(recall.parse_grok_session(
            str(Path(self.root) / "nowhere" / "chat_history.jsonl")))


class DetectGrokFormat(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def test_detects_grok_by_transcript_name(self):
        path = write_session(self.root, CONVERSATION)
        self.assertEqual(read_session.detect_format(path), "grok")

    def test_reads_the_same_turns_recall_indexes(self):
        path = write_session(self.root, CONVERSATION)
        turns = list(read_session.iter_messages(path))
        self.assertEqual(turns, [
            ("user", "why does the build fail on arm"),
            ("assistant", "the toolchain is pinned to x86"),
        ])

    def test_other_formats_are_still_detected(self):
        claude = Path(self.root) / "claude.jsonl"
        claude.write_text(json.dumps(
            {"parentUuid": None, "type": "user",
             "message": {"content": "hi"}}) + "\n", encoding="utf-8")
        self.assertEqual(read_session.detect_format(str(claude)), "claude")


if __name__ == "__main__":
    unittest.main()
