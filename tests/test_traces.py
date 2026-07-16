import pytest

from rl_token_minimize.tools import Workspace
from rl_token_minimize.traces import generate_trace


BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"


def replay(tmp_path, files, trace):
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    ws = Workspace(tmp_path)
    for call in trace:
        result = ws.call(call["name"], call["arguments"])
        assert not result.startswith("error:"), result
    return ws.snapshot()


def test_trace_is_read_then_replace_then_finish():
    trace = generate_trace({"solution.py": BUGGY}, {"solution.py": FIXED})
    assert [c["name"] for c in trace] == ["read_file", "replace_text", "finish"]
    assert trace[0]["arguments"] == {"path": "solution.py"}


def test_trace_replay_produces_reference(tmp_path):
    files = {"solution.py": BUGGY}
    trace = generate_trace(files, {"solution.py": FIXED})
    assert replay(tmp_path, files, trace) == {"solution.py": FIXED}


def test_multi_hunk_trace_replay(tmp_path):
    buggy = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
    fixed = "a = 10\nb = 2\nc = 3\nd = 40\ne = 5\n"
    files = {"f.py": buggy}
    trace = generate_trace(files, {"f.py": fixed})
    assert [c["name"] for c in trace].count("replace_text") == 2
    assert replay(tmp_path, files, trace) == {"f.py": fixed}


def test_ambiguous_hunk_expands_context(tmp_path):
    buggy = "x = 1\ny = 9\nx = 1\n"
    fixed = "x = 1\ny = 9\nx = 2\n"
    files = {"f.py": buggy}
    trace = generate_trace(files, {"f.py": fixed})
    assert replay(tmp_path, files, trace) == {"f.py": fixed}


def test_unresolvable_task_returns_none():
    buggy = "x = 1\n" * 30
    fixed = "x = 1\n" * 15 + "x = 2\n" + "x = 1\n" * 14
    assert generate_trace({"f.py": buggy}, {"f.py": fixed}) is None
    assert generate_trace({"f.py": ""}, {"f.py": "x = 1\n"}) is None


def test_identical_files_returns_none():
    assert generate_trace({"f.py": BUGGY}, {"f.py": BUGGY}) is None


def test_new_or_deleted_files_filtered():
    assert generate_trace({"f.py": BUGGY}, {"f.py": FIXED, "new.py": "x = 1\n"}) is None
    assert generate_trace({"f.py": BUGGY, "old.py": "x = 1\n"}, {"f.py": FIXED}) is None


def test_untouched_files_not_read():
    trace = generate_trace(
        {"f.py": BUGGY, "other.py": "KEEP = 1\n"},
        {"f.py": FIXED, "other.py": "KEEP = 1\n"},
    )
    read_paths = [c["arguments"]["path"] for c in trace if c["name"] == "read_file"]
    assert read_paths == ["f.py"]
