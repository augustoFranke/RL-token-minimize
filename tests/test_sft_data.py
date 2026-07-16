from rl_token_minimize.sft_data import build_sft_example
from rl_token_minimize.tasks import Task, build_humanevalfix_task

TASK = Task(
    task_id="t1",
    files={"solution.py": "def add(a, b):\n    return a - b\n"},
    reference_files={"solution.py": "def add(a, b):\n    return a + b\n"},
    prompt="add(2, 3) returns -1 instead of 5. Fix it.",
    test_code="from solution import add\nassert add(2, 3) == 5\n",
    expected_files=["solution.py"],
)


def test_example_has_system_user_then_tool_turns():
    messages = build_sft_example(TASK)["messages"]
    assert [m["role"] for m in messages[:2]] == ["system", "user"]
    assert messages[1]["content"] == TASK.prompt
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["tool_calls"][0]["function"]["name"] == "finish"


def test_tool_results_are_real_outputs():
    messages = build_sft_example(TASK)["messages"]
    read_response = messages[3]
    assert read_response["role"] == "tool"
    assert "return a - b" in read_response["content"]
    replace_response = messages[5]
    assert replace_response["content"] == "replaced in solution.py"


def test_unresolvable_task_returns_none():
    task = Task(
        task_id="t2",
        files={"f.py": ""},
        reference_files={"f.py": "x = 1\n"},
        prompt="p",
        test_code="",
    )
    assert build_sft_example(task) is None


def test_build_humanevalfix_task():
    row = {
        "task_id": "Python/0",
        "prompt": "def add(a, b):\n",
        "buggy_solution": "    return a - b\n",
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "entry_point": "add",
    }
    task = build_humanevalfix_task(row, failure_output="AssertionError")
    assert task.task_id == "Python_0"
    assert task.files == {"solution.py": "def add(a, b):\n    return a - b\n"}
    assert task.reference_files == {"solution.py": "def add(a, b):\n    return a + b\n"}
    assert task.expected_files == ["solution.py"]
    assert "from solution import add" in task.test_code
    assert "check(add)" in task.test_code
    assert "AssertionError" in task.prompt
    assert "add" in task.prompt
    assert "return a + b" not in task.prompt


def test_concise_failure_strips_paths_and_truncates():
    from rl_token_minimize.tasks import concise_failure

    raw = (
        'Traceback (most recent call last):\n'
        '  File "/var/folders/xy/T/tmpabc/hidden_test.py", line 9, in <module>\n'
        '    check(add)\n'
        '  File "/private/tmp/ws/solution.py", line 2, in add\n'
        'AssertionError: add(2, 3) should be 5'
    )
    out = concise_failure(raw)
    assert "/var/folders" not in out
    assert "/private/tmp" not in out
    assert "AssertionError: add(2, 3) should be 5" in out
    long = "\n".join(f"line {i}" for i in range(100))
    assert len(concise_failure(long).splitlines()) <= 15
