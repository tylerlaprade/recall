"""Tests for incremental indexing.

One property matters more than the rest: an index built up a piece at a time
must hold exactly what an index built in one pass holds. The rest of this file
is a catalogue of ways for that to break.

Fixtures are generated as strings inside tmpdir — no fixture files committed,
and nothing here reads real sessions or a real index.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

# Make scripts/ importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import recall  # noqa: E402

BASE_MTIME = 1_800_000_000  # a fixed point in time; only the order matters


class Corpus:
    """A synthetic ~/.claude and ~/.codex under one root.

    Every write moves the file's mtime forward a step. The scan decides what to
    look at by mtime, and tests run faster than the clock ticks, so the step is
    what makes them repeatable.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.claude = self.root / "claude" / "projects"
        self.codex = self.root / "codex" / "sessions"
        self.pi = self.root / "pi" / "sessions"
        for directory in (self.claude, self.codex, self.pi):
            directory.mkdir(parents=True, exist_ok=True)
        self._tick = 0

    def write(self, path, entries, mode="a"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        self.stamp(path)
        return path

    def write_raw(self, path, text, mode="a"):
        path = Path(path)
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(text)
        self.stamp(path)
        return path

    def stamp(self, path):
        self._tick += 1
        os.utime(path, (BASE_MTIME + self._tick, BASE_MTIME + self._tick))

    def claude_session(self, session_id, entries, project="proj"):
        return self.write(self.claude / project / f"{session_id}.jsonl", entries)

    def codex_session(self, uuid, entries, day="2026/01/01"):
        name = f"rollout-2026-01-01T00-00-00-{uuid}.jsonl"
        return self.write(self.codex / day / name, entries)


def claude_entry(text, role="user", cwd="/work/project", slug=None, ts=None):
    entry = {"type": role, "cwd": cwd, "message": {"content": text}}
    if slug:
        entry["slug"] = slug
    if ts:
        entry["timestamp"] = ts
    return entry


def codex_meta(uuid, cwd="/work/project", ts="2026-01-01T00:00:00.000Z"):
    return {"timestamp": ts, "type": "session_meta", "payload": {"id": uuid, "cwd": cwd}}


def codex_entry(text, role="user", ts="2026-01-01T00:01:00.000Z"):
    return {"timestamp": ts, "type": "response_item",
            "payload": {"role": role, "content": [{"type": "input_text", "text": text}]}}


CODEX_UUID = "019dff1d-385c-7822-8302-008a34dca659"


class IncrementalCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.corpus = Corpus(self.tmp / "corpus")
        self.db = str(self.tmp / "incremental.db")
        self.rebuild_db = str(self.tmp / "rebuild.db")

        for name, value in (("CLAUDE_DIR", self.corpus.root / "claude"),
                            ("CLAUDE_PROJECTS_DIR", self.corpus.claude),
                            ("CODEX_SESSIONS_DIR", self.corpus.codex),
                            ("PI_SESSIONS_DIR", self.corpus.pi)):
            self.addCleanup(setattr, recall, name, getattr(recall, name))
            setattr(recall, name, value)

    def index(self, db=None, force=False):
        conn = sqlite3.connect(db or self.db)
        try:
            recall.create_schema(conn)
            recall.migrate_schema(conn)
            indexed, _, _, _ = recall.index_sessions(conn, force=force)
            return indexed
        finally:
            conn.close()

    def contents(self, db=None):
        conn = sqlite3.connect(db or self.db)
        try:
            sessions = {row[0]: row[1:] for row in conn.execute(
                "SELECT file_path, session_id, source, project, slug, timestamp FROM sessions")}
            messages = {}
            for session_id, role, text in conn.execute(
                    "SELECT session_id, role, text FROM messages"):
                messages.setdefault(session_id, Counter())[(role, text)] += 1
            return sessions, messages
        finally:
            conn.close()

    def assert_matches_rebuild(self):
        self.index(db=self.rebuild_db, force=True)
        self.assertEqual(self.contents(), self.contents(self.rebuild_db))

    def row(self, path):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(
                "SELECT session_id, byte_offset, tail_hash, parser_version, "
                "project, slug, timestamp FROM sessions WHERE file_path = ?",
                (str(path),)).fetchone()
        finally:
            conn.close()

    def texts(self, session_id):
        conn = sqlite3.connect(self.db)
        try:
            return sorted(row[0] for row in conn.execute(
                "SELECT text FROM messages WHERE session_id = ?", (session_id,)))
        finally:
            conn.close()


class GrowingSessions(IncrementalCase):
    def test_matches_a_full_rebuild(self):
        claude = self.corpus.claude_session("11111111-1111-1111-1111-111111111111", [
            claude_entry("first turn", ts="2026-01-01T00:00:00.000Z"),
            claude_entry("first reply", role="assistant")])
        codex = self.corpus.codex_session(CODEX_UUID,
                                          [codex_meta(CODEX_UUID), codex_entry("codex one")])
        self.index()
        for round_no in range(2, 5):
            self.corpus.write(claude, [claude_entry(f"turn {round_no}")])
            self.corpus.write(codex, [codex_entry(f"codex {round_no}")])
            self.index()
        self.assert_matches_rebuild()

    def test_new_messages_are_not_duplicated(self):
        path = self.corpus.claude_session("22222222-2222-2222-2222-222222222222",
                                          [claude_entry("alpha")])
        self.index()
        self.corpus.write(path, [claude_entry("bravo")])
        self.index()
        self.assertEqual(self.texts(self.row(path)[0]), ["alpha", "bravo"])

    def test_an_unchanged_file_is_not_read_again(self):
        self.corpus.claude_session("33333333-3333-3333-3333-333333333333",
                                   [claude_entry("only turn")])
        self.assertEqual(self.index(), 1)
        self.assertEqual(self.index(), 0)

    def test_a_resume_point_is_recorded(self):
        path = self.corpus.claude_session("44444444-4444-4444-4444-444444444444",
                                          [claude_entry("turn")])
        self.index()
        _, offset, tail_hash, version, *_ = self.row(path)
        self.assertEqual(offset, os.path.getsize(path))
        self.assertIsNotNone(tail_hash)
        self.assertEqual(version, recall.PARSER_VERSION)

    def test_a_half_written_line_waits_for_the_rest(self):
        path = self.corpus.claude_session("55555555-5555-5555-5555-555555555555",
                                          [claude_entry("complete")])
        self.index()
        half = json.dumps(claude_entry("torn"))
        self.corpus.write_raw(path, half[: len(half) // 2])
        self.index()
        self.assertEqual(self.texts(self.row(path)[0]), ["complete"])
        self.corpus.write_raw(path, half[len(half) // 2:] + "\n")
        self.index()
        self.assertEqual(self.texts(self.row(path)[0]), ["complete", "torn"])
        self.assert_matches_rebuild()


class Mutations(IncrementalCase):
    """A file that was not simply appended to must be read again in full."""

    def session(self, count=6):
        return self.corpus.claude_session(
            "66666666-6666-6666-6666-666666666666",
            [claude_entry(f"turn {i}") for i in range(count)])

    def test_an_entry_removed_from_the_middle(self):
        path = self.session()
        self.index()
        lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
        del lines[2]
        self.corpus.write_raw(path, "".join(lines), mode="w")
        self.index()
        self.assertNotIn("turn 2", self.texts(self.row(path)[0]))
        self.assert_matches_rebuild()

    def test_a_removal_hidden_by_later_appends(self):
        """The file ends up longer than the resume point again, so only the
        tail hash can tell that the bytes underneath it moved."""
        path = self.session()
        self.index()
        lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
        del lines[2]
        self.corpus.write_raw(path, "".join(lines), mode="w")
        self.corpus.write(path, [claude_entry(f"turn {i}") for i in range(6, 12)])
        self.index()
        texts = self.texts(self.row(path)[0])
        self.assertNotIn("turn 2", texts)
        self.assertIn("turn 11", texts)
        self.assert_matches_rebuild()

    def test_truncation(self):
        path = self.session()
        self.index()
        self.corpus.write(path, [claude_entry("turn 0")], mode="w")
        self.index()
        self.assertEqual(self.texts(self.row(path)[0]), ["turn 0"])
        self.assert_matches_rebuild()

    def test_replacement_by_rename(self):
        path = self.session()
        self.index()
        replacement = self.tmp / "replacement.jsonl"
        replacement.write_text(json.dumps(claude_entry("fresh")) + "\n", encoding="utf-8")
        os.replace(replacement, path)
        self.corpus.stamp(path)
        self.index()
        self.assertEqual(self.texts(self.row(path)[0]), ["fresh"])
        self.assert_matches_rebuild()

    def test_an_edit_inside_the_tail_window(self):
        path = self.session(4)
        self.index()
        text = Path(path).read_text(encoding="utf-8").replace("turn 3", "edited 3")
        self.corpus.write_raw(path, text, mode="w")
        self.index()
        self.assertIn("edited 3", self.texts(self.row(path)[0]))
        self.assert_matches_rebuild()


class Metadata(IncrementalCase):
    """A tail read sees only the end of a file. What the head supplied has to
    survive rather than be overwritten with nothing."""

    def test_the_earliest_timestamp_is_kept(self):
        path = self.corpus.claude_session("77777777-7777-7777-7777-777777777777",
                                          [claude_entry("open", ts="2026-01-01T00:00:00.000Z")])
        self.index()
        first = self.row(path)[6]
        self.corpus.write(path, [claude_entry("later", ts="2026-06-01T00:00:00.000Z")])
        self.index()
        self.assertEqual(self.row(path)[6], first)

    def test_an_earlier_timestamp_still_wins(self):
        path = self.corpus.claude_session("88888888-8888-8888-8888-888888888888",
                                          [claude_entry("open", ts="2026-06-01T00:00:00.000Z")])
        self.index()
        self.corpus.write(path, [claude_entry("older", ts="2026-01-01T00:00:00.000Z")])
        self.index()
        self.assertEqual(self.row(path)[6],
                         recall.parse_iso_timestamp("2026-01-01T00:00:00.000Z"))
        self.assert_matches_rebuild()

    def test_the_first_slug_wins(self):
        path = self.corpus.claude_session("99999999-9999-9999-9999-999999999999",
                                          [claude_entry("open", slug="first-title")])
        self.index()
        self.corpus.write(path, [claude_entry("later", slug="second-title")])
        self.index()
        self.assertEqual(self.row(path)[5], "first-title")
        self.assert_matches_rebuild()

    def test_the_first_cwd_wins(self):
        path = self.corpus.claude_session("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                                          [claude_entry("open", cwd="/first")])
        self.index()
        self.corpus.write(path, [claude_entry("later", cwd="/second")])
        self.index()
        self.assertEqual(self.row(path)[4], "/first")
        self.assert_matches_rebuild()

    def test_codex_keeps_the_id_from_its_first_line(self):
        """Codex takes its session id from session_meta at the head, so a tail
        read must not fall back to the rollout file name."""
        path = self.corpus.codex_session(CODEX_UUID,
                                         [codex_meta(CODEX_UUID), codex_entry("open")])
        self.index()
        self.assertEqual(self.row(path)[0], CODEX_UUID)
        self.corpus.write(path, [codex_entry("later")])
        self.index()
        self.assertEqual(self.row(path)[0], CODEX_UUID)
        self.assertEqual(self.texts(CODEX_UUID), ["later", "open"])


class PiIsAlwaysReadInFull(IncrementalCase):
    def test_pi_is_not_an_append_only_source(self):
        self.assertNotIn("pi", recall.APPEND_ONLY_SOURCES)

    def test_pi_rows_carry_no_resume_point(self):
        path = self.corpus.write(self.corpus.pi / "s" / "2026-01-01T00_00_00_abc.jsonl", [
            {"type": "session", "id": "abc", "cwd": "/work", "version": 3},
            {"type": "message", "message": {"role": "user", "content": "hello pi"}}])
        self.index()
        self.assertEqual(self.row(path)[1], 0)


class ParserVersion(IncrementalCase):
    def test_a_bump_reaches_a_session_that_has_stopped_growing(self):
        """Without this, a change to what the parsers keep would apply only to
        sessions still being written."""
        path = self.corpus.claude_session("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                                          [claude_entry("written once")])
        self.index()
        self.assertEqual(self.index(), 0)
        original = recall.PARSER_VERSION
        recall.PARSER_VERSION = original + 1
        try:
            self.assertEqual(self.index(), 1)
            self.assertEqual(self.row(path)[3], original + 1)
            self.assertEqual(self.texts(self.row(path)[0]), ["written once"])
        finally:
            recall.PARSER_VERSION = original


class Migration(IncrementalCase):
    OLD_SCHEMA = """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT,
            project TEXT, slug TEXT, timestamp INTEGER, mtime REAL
        );
        CREATE VIRTUAL TABLE messages USING fts5(
            session_id UNINDEXED, role, text, tokenize='porter unicode61');
        CREATE VIRTUAL TABLE messages_cjk USING fts5(
            session_id UNINDEXED, role, text, tokenize='trigram');
    """

    def test_an_older_index_gains_the_columns_in_place(self):
        conn = sqlite3.connect(self.db)
        conn.executescript(self.OLD_SCHEMA)
        conn.execute("INSERT INTO sessions VALUES "
                     "('old', 'claude', '/gone.jsonl', '/work', 'slug', 1700, 1.0)")
        conn.execute("INSERT INTO messages VALUES ('old', 'user', 'kept text')")
        conn.commit()
        recall.migrate_schema(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        self.assertLessEqual({"byte_offset", "tail_hash", "parser_version"}, columns)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT text FROM messages").fetchone()[0],
                         "kept text")
        conn.close()

    def test_existing_rows_are_read_in_full_once_more(self):
        conn = sqlite3.connect(self.db)
        conn.executescript(self.OLD_SCHEMA)
        conn.execute("INSERT INTO sessions VALUES "
                     "('old', 'claude', '/gone.jsonl', '/work', 'slug', 1700, 1.0)")
        conn.commit()
        recall.migrate_schema(conn)
        offset, tail_hash, version = conn.execute(
            "SELECT byte_offset, tail_hash, parser_version FROM sessions").fetchone()
        self.assertEqual(recall.resume_offset("/gone.jsonl", offset, tail_hash, version), 0)
        conn.close()

    def test_migrating_twice_changes_nothing(self):
        conn = sqlite3.connect(self.db)
        conn.executescript(self.OLD_SCHEMA)
        recall.migrate_schema(conn)
        first = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        recall.migrate_schema(conn)
        self.assertEqual({row[1] for row in conn.execute("PRAGMA table_info(sessions)")},
                         first)
        conn.close()


class Reader(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def write(self, text):
        path = self.tmp / "session.jsonl"
        path.write_bytes(text.encode("utf-8"))
        return str(path)

    def read_all(self, path, start=0):
        lines, offset = [], start
        for line, line_end in recall.read_complete_lines(path, start):
            lines.append(line)
            offset = line_end
        return lines, offset

    def test_offsets_land_just_past_each_newline(self):
        path = self.write("aa\nbbb\nc\n")
        self.assertEqual([o for _, o in recall.read_complete_lines(path)], [3, 7, 9])

    def test_a_partial_final_line_is_left_alone(self):
        path = self.write('{"a":1}\n{"b":2}\n{"c":unfin')
        lines, offset = self.read_all(path)
        self.assertEqual(len(lines), 2)
        self.assertEqual(offset, 16)

    def test_resuming_from_a_non_zero_start(self):
        path = self.write("one\ntwo\nthree\n")
        lines, offset = self.read_all(path, 4)
        self.assertEqual(lines, ["two\n", "three\n"])
        self.assertEqual(offset, 14)

    def test_a_line_longer_than_the_read_buffer(self):
        long_line = "x" * (4 << 20)
        path = self.write(f"first\n{long_line}\nlast\n")
        lines, offset = self.read_all(path)
        self.assertEqual(lines[1], long_line + "\n")
        self.assertEqual(offset, os.path.getsize(path))

    def test_multibyte_characters_do_not_disturb_the_offset(self):
        path = self.write("a" * 100 + "中文\n")
        lines, offset = self.read_all(path)
        self.assertNotIn("�", lines[0])
        self.assertEqual(offset, os.path.getsize(path))

    def test_an_empty_file(self):
        self.assertEqual(self.read_all(self.write("")), ([], 0))

    def test_resume_refuses_a_truncated_file(self):
        path = self.write("alpha\nbravo\ncharlie\n")
        offset = os.path.getsize(path)
        stored = recall.tail_hash_at(path, offset)
        Path(path).write_bytes(b"alpha\n")
        self.assertEqual(recall.resume_offset(path, offset, stored,
                                              recall.PARSER_VERSION), 0)

    def test_resume_accepts_an_appended_file(self):
        path = self.write("alpha\nbravo\n")
        offset = os.path.getsize(path)
        stored = recall.tail_hash_at(path, offset)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("charlie\n")
        self.assertEqual(recall.resume_offset(path, offset, stored,
                                              recall.PARSER_VERSION), offset)

    def test_resume_refuses_without_a_stored_hash(self):
        path = self.write("alpha\n")
        self.assertEqual(recall.resume_offset(path, 6, None, recall.PARSER_VERSION), 0)


if __name__ == "__main__":
    unittest.main()
