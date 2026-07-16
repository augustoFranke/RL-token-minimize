from rl_token_minimize.verifier import run_tests

TEST_CODE = "from solution import add\nassert add(2, 3) == 5, 'add(2, 3) should be 5'\n"


def test_passing_solution(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a + b\n")
    result = run_tests(tmp_path, TEST_CODE)
    assert result.passed is True


def test_failing_solution_captures_output(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n")
    result = run_tests(tmp_path, TEST_CODE)
    assert result.passed is False
    assert "AssertionError" in result.output
    assert "add(2, 3) should be 5" in result.output


def test_syntax_error_fails(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b:\n")
    result = run_tests(tmp_path, TEST_CODE)
    assert result.passed is False
    assert "SyntaxError" in result.output


def test_infinite_loop_times_out(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    while True: pass\n")
    result = run_tests(tmp_path, TEST_CODE, timeout=2)
    assert result.passed is False
    assert "timeout" in result.output


def test_output_is_truncated(tmp_path):
    (tmp_path / "solution.py").write_text(
        "def add(a, b):\n    raise ValueError('x' * 100000)\n"
    )
    result = run_tests(tmp_path, TEST_CODE)
    assert result.passed is False
    assert len(result.output) <= 2000


def test_test_file_not_left_in_workspace(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a + b\n")
    run_tests(tmp_path, TEST_CODE)
    assert [p.name for p in tmp_path.iterdir()] == ["solution.py"]
