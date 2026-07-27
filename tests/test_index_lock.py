"""Tests for index_lock.

Indexing is a single write transaction spanning every file it parses. Without
a lock, a second run started during a long index waits out SQLite's busy
timeout and then dies with "database is locked".

Fixtures are generated inside tmpdir — no fixture files committed.
"""
from __future__ import annotations

import fcntl
import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path

# Make scripts/ importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import recall  # noqa: E402


@contextmanager
def lock_held_elsewhere(lock_path):
    """Hold the lock the way another run would.

    flock belongs to the open file description rather than the process, so a
    second handle on the same path contends with the first even from here.
    """
    Path(lock_path).touch()
    handle = open(lock_path, "a", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        yield
    finally:
        handle.close()


class IndexLock(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lock_path = str(Path(tmp.name) / "recall.db.lock")

        for name, value in (("DB_LOCK_PATH", Path(self.lock_path)),
                            ("LOCK_WAIT_SECONDS", 0.3)):
            self.addCleanup(setattr, recall, name, getattr(recall, name))
            setattr(recall, name, value)

    def test_an_uncontended_run_takes_the_lock(self):
        with recall.index_lock() as have_lock:
            self.assertTrue(have_lock)

    def test_the_lock_is_released_afterwards(self):
        with recall.index_lock():
            pass
        with recall.index_lock() as have_lock:
            self.assertTrue(have_lock)

    def test_the_lock_is_released_even_when_the_body_raises(self):
        with self.assertRaises(ValueError):
            with recall.index_lock():
                raise ValueError("indexing blew up")
        with recall.index_lock() as have_lock:
            self.assertTrue(have_lock)

    def test_a_second_run_does_not_get_the_lock(self):
        with lock_held_elsewhere(self.lock_path):
            with redirect_stderr(io.StringIO()):
                with recall.index_lock() as have_lock:
                    self.assertFalse(have_lock)

    def test_a_second_run_gives_up_rather_than_hanging(self):
        with lock_held_elsewhere(self.lock_path):
            started = time.monotonic()
            with redirect_stderr(io.StringIO()):
                with recall.index_lock():
                    pass
            waited = time.monotonic() - started
        self.assertGreaterEqual(waited, recall.LOCK_WAIT_SECONDS)
        self.assertLess(waited, recall.LOCK_WAIT_SECONDS + 5)

    def test_giving_up_says_so(self):
        stderr = io.StringIO()
        with lock_held_elsewhere(self.lock_path):
            with redirect_stderr(stderr):
                with recall.index_lock():
                    pass
        self.assertIn("Another process is indexing", stderr.getvalue())

    def test_the_lock_file_is_created_if_absent(self):
        self.assertFalse(os.path.exists(self.lock_path))
        with recall.index_lock() as have_lock:
            self.assertTrue(have_lock)
        self.assertTrue(os.path.exists(self.lock_path))


if __name__ == "__main__":
    unittest.main()
