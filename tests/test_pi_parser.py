"""Tests for parse_pi_session and pi detection in read_session.

Pi session format (earendil-works/pi-mono, was badlogic/pi-mono):
  - Header line: {"type": "session", "id": "...", "cwd": "...", "version": 3, ...}
  - Subsequent lines: {"type": "message", "id": ..., "parentId": ..., "timestamp": ...,
                       "message": {"role": "user|assistant|...", "content": ...}}
  - Other top-level types we skip: custom, custom_message, session_info,
    model_change, thinking_level_change, compaction, branch_summary, label.
  - Non-conversational roles we skip: toolResult, bashExecution.

Fixtures are generated as strings inside tmpdir — no fixture files committed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make scripts/ importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import recall  # noqa: E402
import read_session  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────


def write_jsonl(tmpdir: Path, name: str, entries: list[dict]) -> str:
    """Write a list of entries as JSONL into tmpdir and return the path."""
    path = tmpdir / name
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry))
            f.write("\n")
    return str(path)


# A representative pi v3 session: header + user (string content) + assistant
# (mixed text/thinking/toolCall blocks) + toolResult + bashExecution +
# custom + custom_message + session_info + model_change + thinking_level_change.
# The parser should keep only the user text and the assistant text block.
PI_V3_SAMPLE = [
    {
        "type": "session",
        "version": 3,
        "id": "019dfd8d-da36-7552-b0d5-dfa08528cf9b",
        "timestamp": "2026-05-06T13:50:25.335Z",
        "cwd": "/Users/alice/Vaults/blog",
    },
    {
        "type": "message",
        "id": "u1",
        "parentId": None,
        "timestamp": "2026-05-06T13:52:11.384Z",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "Let's do a spring cleaning."},
            ],
            "timestamp": 1778075531338,
        },
    },
    {
        "type": "message",
        "id": "a1",
        "parentId": "u1",
        "timestamp": "2026-05-06T13:52:12.000Z",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should reason about this."},
                {"type": "text", "text": "I will help you clean up."},
                {"type": "toolCall", "id": "t1", "name": "bash", "arguments": {"cmd": "ls"}},
            ],
            "api": "anthropic",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "usage": {"input": 1, "output": 1},
            "stopReason": "toolUse",
        },
    },
    {
        "type": "message",
        "id": "tr1",
        "parentId": "a1",
        "timestamp": "2026-05-06T13:52:13.000Z",
        "message": {
            "role": "toolResult",
            "toolCallId": "t1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "file1\nfile2"}],
            "isError": False,
        },
    },
    {
        "type": "message",
        "id": "be1",
        "parentId": "tr1",
        "timestamp": "2026-05-06T13:52:14.000Z",
        "message": {
            "role": "bashExecution",
            "command": "ls -la",
            "output": "total 0",
            "exitCode": 0,
            "cancelled": False,
            "truncated": False,
        },
    },
    {
        "type": "custom",
        "id": "c1",
        "parentId": "be1",
        "timestamp": "2026-05-06T13:52:15.000Z",
        "customType": "my-extension",
        "data": {"count": 42},
    },
    {
        "type": "custom_message",
        "id": "cm1",
        "parentId": "c1",
        "timestamp": "2026-05-06T13:52:16.000Z",
        "customType": "my-extension",
        "content": "extension-injected note",
        "display": True,
    },
    {
        "type": "session_info",
        "id": "si1",
        "parentId": "cm1",
        "timestamp": "2026-05-06T13:52:17.000Z",
        "name": "spring cleaning",
    },
    {
        "type": "model_change",
        "id": "mc1",
        "parentId": "si1",
        "timestamp": "2026-05-06T13:52:18.000Z",
        "provider": "openai",
        "modelId": "gpt-4o",
    },
    {
        "type": "thinking_level_change",
        "id": "tl1",
        "parentId": "mc1",
        "timestamp": "2026-05-06T13:52:19.000Z",
        "thinkingLevel": "high",
    },
    {
        "type": "message",
        "id": "u2",
        "parentId": "tl1",
        "timestamp": "2026-05-06T13:52:20.000Z",
        "message": {
            "role": "user",
            "content": "second user turn — plain string content",
        },
    },
]


# ── parse_pi_session ─────────────────────────────────────────────────────────


class TestParsePiSession(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="recall-test-"))

    def _write(self, name: str, entries: list[dict]) -> str:
        return write_jsonl(self.tmpdir, name, entries)

    def test_header_metadata(self):
        """Header gives session_id, cwd→project, earliest timestamp."""
        path = self._write(
            "2026-05-06T13-50-25-335Z_019dfd8d-da36-7552-b0d5-dfa08528cf9b.jsonl",
            [PI_V3_SAMPLE[0]],
        )
        metadata, messages, _ = recall.parse_pi_session(path)
        self.assertEqual(metadata["session_id"], "019dfd8d-da36-7552-b0d5-dfa08528cf9b")
        self.assertEqual(metadata["source"], "pi")
        self.assertEqual(metadata["project"], "/Users/alice/Vaults/blog")
        self.assertEqual(metadata["file_path"], path)
        self.assertGreater(metadata["timestamp"], 0)
        self.assertEqual(messages, [])

    def test_extracts_user_and_assistant_text(self):
        """User string content and assistant TextContent are both kept."""
        path = self._write("session.jsonl", PI_V3_SAMPLE)
        _, messages, _ = recall.parse_pi_session(path)

        roles = [m[0] for m in messages]
        texts = [m[1] for m in messages]

        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertIn("spring cleaning", texts[0])
        self.assertEqual(texts[1], "I will help you clean up.")
        self.assertIn("plain string content", texts[2])

    def test_skips_thinking_toolcall_image_blocks(self):
        """Thinking, toolCall, and image blocks are dropped from assistant content."""
        path = self._write("session.jsonl", PI_V3_SAMPLE)
        _, messages, _ = recall.parse_pi_session(path)

        # Assistant turn had thinking + text + toolCall — only text should remain
        assistant_texts = [t for r, t in messages if r == "assistant"]
        self.assertEqual(len(assistant_texts), 1)
        self.assertNotIn("reason about this", assistant_texts[0])
        self.assertNotIn("toolCall", assistant_texts[0])
        self.assertNotIn("bash", assistant_texts[0])
        self.assertEqual(assistant_texts[0], "I will help you clean up.")

    def test_skips_toolresult_and_bashexecution(self):
        """toolResult and bashExecution roles produce no indexed messages."""
        path = self._write("session.jsonl", PI_V3_SAMPLE)
        _, messages, _ = recall.parse_pi_session(path)

        roles = [m[0] for m in messages]
        self.assertNotIn("toolResult", roles)
        self.assertNotIn("bashExecution", roles)

    def test_skips_non_message_top_level_types(self):
        """custom, custom_message, session_info, model_change, thinking_level_change skipped."""
        path = self._write("session.jsonl", PI_V3_SAMPLE)
        _, messages, _ = recall.parse_pi_session(path)

        joined = "\n".join(t for _, t in messages)
        self.assertNotIn("extension-injected note", joined)
        self.assertNotIn("spring cleaning", joined.split("\n")[1] if "\n" in joined else "")
        # We accept "spring cleaning" appearing in the user turn (that's the text).
        # session_info.name is also "spring cleaning" — but it must not be a separate message.
        # Total messages should be 3 (user, assistant, user), not more.
        self.assertEqual(len(messages), 3)

    def test_compaction_branch_summary_label_skipped(self):
        """compaction, branch_summary, label entry types are skipped."""
        entries = [PI_V3_SAMPLE[0]] + [
            {
                "type": "compaction",
                "id": "comp1",
                "parentId": None,
                "timestamp": "2026-05-06T14:00:00.000Z",
                "summary": "earlier turns summarized here",
                "firstKeptEntryId": "x",
                "tokensBefore": 1000,
            },
            {
                "type": "branch_summary",
                "id": "bs1",
                "parentId": None,
                "timestamp": "2026-05-06T14:01:00.000Z",
                "fromId": "y",
                "summary": "abandoned branch summary",
            },
            {
                "type": "label",
                "id": "lb1",
                "parentId": None,
                "timestamp": "2026-05-06T14:02:00.000Z",
                "targetId": "u1",
                "label": "checkpoint-1",
            },
        ]
        path = self._write("session.jsonl", entries)
        _, messages, _ = recall.parse_pi_session(path)
        self.assertEqual(messages, [])

    def test_slug_includes_date_and_short_id(self):
        """Slug derived from filename: YYYY-MM-DD-<uuid8>."""
        name = "2026-05-06T13-50-25-335Z_019dfd8d-da36-7552-b0d5-dfa08528cf9b.jsonl"
        path = self._write(name, [PI_V3_SAMPLE[0]])
        metadata, _, _ = recall.parse_pi_session(path)
        self.assertEqual(metadata["slug"], "2026-05-06-019dfd8d")

    def test_slug_fallback_when_filename_unusual(self):
        """Filename without the expected pattern: slug falls back to a session-id prefix."""
        path = self._write("oddname.jsonl", [PI_V3_SAMPLE[0]])
        metadata, _, _ = recall.parse_pi_session(path)
        # Either short_id or session_id prefix — both are acceptable; just non-empty.
        self.assertTrue(metadata["slug"])

    def test_malformed_json_line_is_skipped(self):
        """A malformed line in the middle of the file does not break parsing."""
        path = self.tmpdir / "session.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(PI_V3_SAMPLE[0]) + "\n")
            f.write("this is not json\n")
            f.write(json.dumps(PI_V3_SAMPLE[1]) + "\n")  # user message
        _, messages, _ = recall.parse_pi_session(str(path))
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0][0], "user")

    def test_empty_lines_are_ignored(self):
        """Blank lines do not affect parsing."""
        path = self.tmpdir / "session.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write("\n")
            f.write(json.dumps(PI_V3_SAMPLE[0]) + "\n")
            f.write("\n\n")
            f.write(json.dumps(PI_V3_SAMPLE[1]) + "\n")
            f.write("\n")
        _, messages, _ = recall.parse_pi_session(str(path))
        self.assertEqual(len(messages), 1)

    def test_string_content_user_message(self):
        """User message with content as a plain string (not array)."""
        entries = [
            PI_V3_SAMPLE[0],
            {
                "type": "message",
                "id": "u1",
                "parentId": None,
                "timestamp": "2026-05-06T13:52:11.000Z",
                "message": {"role": "user", "content": "hello world"},
            },
        ]
        path = self._write("session.jsonl", entries)
        _, messages, _ = recall.parse_pi_session(path)
        self.assertEqual(messages, [("user", "hello world")])

    def test_returns_none_on_unreadable_file(self):
        """Permission errors yield None (warning to stderr), not an exception."""
        path = self.tmpdir / "noperm.jsonl"
        path.write_text(json.dumps(PI_V3_SAMPLE[0]) + "\n")
        os.chmod(str(path), 0o000)
        try:
            result = recall.parse_pi_session(str(path))
            # On some systems root can still read; accept either outcome,
            # but if we did get a result it must be well-formed.
            if result is not None:
                self.assertEqual(result[0]["source"], "pi")
        finally:
            os.chmod(str(path), 0o600)


# ── detect_format (read_session) ─────────────────────────────────────────────


class TestDetectFormat(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="recall-test-"))

    def _write(self, name: str, entries: list[dict]) -> str:
        return write_jsonl(self.tmpdir, name, entries)

    def test_detects_pi_from_session_header(self):
        path = self._write("pi.jsonl", [PI_V3_SAMPLE[0]])
        self.assertEqual(read_session.detect_format(path), "pi")

    def test_detects_pi_with_version_but_no_cwd(self):
        """Pi headers must have at least one of cwd/version."""
        entries = [{"type": "session", "version": 3, "id": "x", "timestamp": "2026-01-01T00:00:00Z"}]
        path = self._write("pi.jsonl", entries)
        self.assertEqual(read_session.detect_format(path), "pi")

    def test_does_not_misclassify_session_typed_entries_without_cwd_or_version(self):
        """An entry that happens to be type='session' but lacks cwd/version is not pi."""
        entries = [{"type": "session", "id": "x"}]  # no cwd, no version
        path = self._write("ambig.jsonl", entries)
        self.assertNotEqual(read_session.detect_format(path), "pi")

    def test_detects_claude_from_parentuuid(self):
        entries = [
            {
                "parentUuid": None,
                "type": "user",
                "uuid": "u1",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "hi"},
            }
        ]
        path = self._write("claude.jsonl", entries)
        self.assertEqual(read_session.detect_format(path), "claude")

    def test_detects_codex_modern_session_meta(self):
        entries = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "type": "session_meta",
                "payload": {"id": "x", "cwd": "/tmp"},
            }
        ]
        path = self._write("codex.jsonl", entries)
        self.assertEqual(read_session.detect_format(path), "codex")

    def test_detects_codex_legacy_state_record(self):
        entries = [{"record_type": "state", "id": "x"}]
        path = self._write("codex-legacy.jsonl", entries)
        self.assertEqual(read_session.detect_format(path), "codex")


# ── read_session.iter_messages on pi ─────────────────────────────────────────


class TestIterMessagesPi(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="recall-test-"))

    def test_iter_pi_yields_user_and_assistant_only(self):
        path = write_jsonl(self.tmpdir, "session.jsonl", PI_V3_SAMPLE)
        out = list(read_session.iter_messages(path))
        roles = [r for r, _ in out]
        self.assertEqual(roles, ["user", "assistant", "user"])
        # Assistant text is the plain TextContent only, not thinking/toolCall
        self.assertEqual(out[1][1], "I will help you clean up.")


# ── list_sessions ────────────────────────────────────────────────────────────


class TestListSessions(unittest.TestCase):
    def setUp(self):
        import sqlite3
        import time

        self.tmpdir = Path(tempfile.mkdtemp(prefix="recall-test-"))
        self.db_path = self.tmpdir / "recall.db"
        self.conn = sqlite3.connect(str(self.db_path))
        recall.create_schema(self.conn)
        recall.migrate_schema(self.conn)

        # Seed with three sessions across three sources, recent first.
        now_ms = int(time.time() * 1000)
        day = 86_400_000
        rows = [
            ("pi-recent", "pi", "/p/recent.jsonl", "/Users/alice/proj-a", "slug-pi-recent", now_ms),
            ("claude-older", "claude", "/c/older.jsonl", "/Users/alice/proj-b", "slug-claude-older", now_ms - 3 * day),
            ("codex-oldest", "codex", "/x/oldest.jsonl", "/Users/alice/proj-a", "slug-codex-oldest", now_ms - 30 * day),
        ]
        for sid, src, fp, proj, slug, ts in rows:
            self.conn.execute(
                "INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, src, fp, proj, slug, ts, 0.0),
            )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_returns_all_sessions_sorted_by_recency(self):
        rows = recall.list_sessions(self.conn, limit=10)
        slugs = [r[4] for r in rows]
        self.assertEqual(slugs, ["slug-pi-recent", "slug-claude-older", "slug-codex-oldest"])

    def test_filters_by_source(self):
        rows = recall.list_sessions(self.conn, source="pi", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "pi")

    def test_filters_by_days(self):
        """--days 1 keeps the recent pi session, drops the older two."""
        rows = recall.list_sessions(self.conn, days=1, limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][4], "slug-pi-recent")

    def test_filters_by_project_prefix(self):
        """--project filters by prefix match against the stored project path."""
        rows = recall.list_sessions(self.conn, project="/Users/alice/proj-a", limit=10)
        self.assertEqual(sorted(r[4] for r in rows), ["slug-codex-oldest", "slug-pi-recent"])

    def test_combines_filters(self):
        """Multiple filters AND together."""
        rows = recall.list_sessions(
            self.conn, source="pi", project="/Users/alice/proj-a", days=1, limit=10
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][4], "slug-pi-recent")

    def test_respects_limit(self):
        rows = recall.list_sessions(self.conn, limit=2)
        self.assertEqual(len(rows), 2)

    def test_empty_db(self):
        """List mode on an empty sessions table returns []."""
        self.conn.execute("DELETE FROM sessions")
        self.conn.commit()
        self.assertEqual(recall.list_sessions(self.conn, limit=10), [])

    def test_row_shape_matches_search_results(self):
        """Returned tuples have the same arity as search() rows (8 fields)."""
        rows = recall.list_sessions(self.conn, limit=1)
        self.assertEqual(len(rows[0]), 8)
        # Last two columns are excerpt and rank — empty / zero for list mode.
        self.assertEqual(rows[0][6], "")
        self.assertEqual(rows[0][7], 0.0)


# ── Integration: round-trip a real pi session through the parser ─────────────


class TestRealPiSession(unittest.TestCase):
    """End-to-end: if the host has any pi sessions on disk, parse one and assert
    the parser produces well-formed output. Skipped on machines without pi.
    """

    def test_parse_any_real_pi_session(self):
        pi_dir = Path.home() / ".pi" / "agent" / "sessions"
        if not pi_dir.is_dir():
            self.skipTest("no pi sessions on this host")

        files = sorted(pi_dir.glob("**/*.jsonl"))
        if not files:
            self.skipTest("no pi session files on this host")

        # Pick the smallest non-empty file to keep the test fast.
        smallest = min((f for f in files if f.stat().st_size > 100), key=lambda f: f.stat().st_size, default=None)
        if smallest is None:
            self.skipTest("no usable pi session files on this host")

        result = recall.parse_pi_session(str(smallest))
        self.assertIsNotNone(result)
        metadata, messages = result
        self.assertEqual(metadata["source"], "pi")
        self.assertTrue(metadata["session_id"])
        self.assertTrue(metadata["slug"])
        self.assertGreater(metadata["timestamp"], 0)
        # Every message must be user or assistant, with non-empty text.
        for role, text in messages:
            self.assertIn(role, ("user", "assistant"))
            self.assertTrue(text)


if __name__ == "__main__":
    unittest.main()
