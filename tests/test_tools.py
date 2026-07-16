import pytest

from rl_token_minimize.tools import ToolError, Workspace


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "util.py").write_text("VALUE = 1\n")
    return Workspace(tmp_path)


def test_list_files_returns_sorted_relative_paths(ws):
    assert ws.list_files() == "pkg/util.py\nsolution.py"


def test_search_text_returns_matching_lines_with_locations(ws):
    assert ws.search_text("return") == "solution.py:2:    return a - b"


def test_search_text_no_match_reports_it(ws):
    assert ws.search_text("nonexistent") == "no matches"


def test_read_file_returns_numbered_lines(ws):
    assert ws.read_file("solution.py") == "1|def add(a, b):\n2|    return a - b"


def test_read_file_line_range(ws):
    assert ws.read_file("solution.py", start_line=2, end_line=2) == "2|    return a - b"


def test_read_file_missing_raises(ws):
    with pytest.raises(ToolError, match="no such file"):
        ws.read_file("missing.py")


def test_read_file_outside_workspace_raises(ws):
    with pytest.raises(ToolError):
        ws.read_file("../etc/passwd")


def test_replace_text_applies_unique_match(ws):
    ws.replace_text("solution.py", "return a - b", "return a + b")
    assert (ws.root / "solution.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_replace_text_zero_matches_raises(ws):
    with pytest.raises(ToolError, match="not found"):
        ws.replace_text("solution.py", "return a * b", "return a + b")


def test_replace_text_ambiguous_match_raises(ws):
    (ws.root / "dup.py").write_text("x = 1\nx = 1\n")
    with pytest.raises(ToolError, match="2 times"):
        ws.replace_text("dup.py", "x = 1", "x = 2")


def test_call_dispatches_and_returns_string(ws):
    assert ws.call("read_file", {"path": "solution.py"}).startswith("1|def add")


def test_call_tool_error_returned_as_error_string(ws):
    result = ws.call("read_file", {"path": "missing.py"})
    assert result.startswith("error:")


def test_call_unknown_tool_is_error(ws):
    assert ws.call("run_tests", {}).startswith("error: unknown tool")


def test_call_bad_arguments_is_error(ws):
    assert ws.call("read_file", {"wrong_arg": 1}).startswith("error:")


def test_read_file_invalid_utf8_is_error_not_raise(ws):
    (ws.root / "bad.bin").write_bytes(b"\xff\xfe")
    result = ws.call("read_file", {"path": "bad.bin"})
    assert result.startswith("error:")


def test_search_text_invalid_utf8_file_is_error_not_raise(ws):
    (ws.root / "bad.bin").write_bytes(b"\xff\xfe")
    result = ws.call("search_text", {"pattern": "return"})
    assert result.startswith("error:")


def test_diff_stats_tracks_changed_files(ws):
    before = ws.snapshot()
    ws.replace_text("solution.py", "return a - b", "return a + b")
    assert ws.changed_files(before) == ["solution.py"]
