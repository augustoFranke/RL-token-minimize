from rl_token_minimize.episode import parse_tool_calls, run_episode
from rl_token_minimize.reward import RewardConfig
from rl_token_minimize.tasks import Task


def tc(name, **arguments):
    import json
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


TASK = Task(
    task_id="t1",
    files={"solution.py": "def add(a, b):\n    return a - b\n"},
    reference_files={"solution.py": "def add(a, b):\n    return a + b\n"},
    prompt="add(2, 3) returns -1 instead of 5. Fix it.",
    test_code="from solution import add\nassert add(2, 3) == 5\n",
    expected_files=["solution.py"],
)


def scripted(responses):
    it = iter(responses)
    return lambda messages: next(it)


def test_parse_single_tool_call():
    calls = parse_tool_calls(tc("read_file", path="a.py"))
    assert calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]


def test_parse_ignores_surrounding_text_and_thinking():
    text = "<think>hmm</think>\nLet me look.\n" + tc("list_files")
    assert parse_tool_calls(text) == [{"name": "list_files", "arguments": {}}]


def test_parse_no_tool_call_returns_empty():
    assert parse_tool_calls("I think the bug is in add.") == []


def test_parse_malformed_json_returns_none():
    assert parse_tool_calls("<tool_call>\n{not json}\n</tool_call>") is None


def test_successful_episode_passes_and_rewards(tmp_path):
    policy = scripted([
        tc("read_file", path="solution.py"),
        tc("replace_text", path="solution.py", old_text="return a - b", new_text="return a + b"),
        tc("finish"),
    ])
    result = run_episode(TASK, policy, workdir=tmp_path)
    assert result.outcome.tests_passed is True
    assert result.outcome.finished is True
    assert result.reward >= RewardConfig().pass_reward


def test_malformed_output_is_protocol_error(tmp_path):
    result = run_episode(TASK, scripted(["<tool_call>{bad}</tool_call>"]), workdir=tmp_path)
    assert result.outcome.protocol_error is True
    assert result.reward == RewardConfig().protocol_penalty


def test_plain_text_without_tool_call_is_protocol_error(tmp_path):
    result = run_episode(TASK, scripted(["The bug is a minus sign."]), workdir=tmp_path)
    assert result.outcome.protocol_error is True


def test_finish_without_edit_gets_no_edit_penalty(tmp_path):
    result = run_episode(TASK, scripted([tc("finish")]), workdir=tmp_path)
    assert result.outcome.edited is False
    assert result.reward == RewardConfig().no_edit_penalty


def test_failed_replace_is_tracked_and_episode_continues(tmp_path):
    policy = scripted([
        tc("replace_text", path="solution.py", old_text="return a * b", new_text="return a + b"),
        tc("replace_text", path="solution.py", old_text="return a - b", new_text="return a + b"),
        tc("finish"),
    ])
    result = run_episode(TASK, policy, workdir=tmp_path)
    assert result.outcome.failed_replaces == 1
    assert result.outcome.tests_passed is True


def test_max_turns_ends_episode_unfinished(tmp_path):
    policy = scripted([tc("list_files")] * 50)
    result = run_episode(TASK, policy, workdir=tmp_path, max_turns=3)
    assert result.outcome.finished is False
    assert result.outcome.tool_calls == 3


def test_tool_results_are_appended_as_tool_messages(tmp_path):
    policy = scripted([tc("read_file", path="solution.py"), tc("finish")])
    result = run_episode(TASK, policy, workdir=tmp_path)
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "return a - b" in result.messages[3]["content"]


def test_extra_tool_calls_are_noted_and_only_first_applied(tmp_path):
    text = tc("read_file", path="solution.py") + tc("list_files")
    policy = scripted([text, tc("finish")])
    result = run_episode(TASK, policy, workdir=tmp_path)
    tool_msg = result.messages[3]["content"]
    assert "note: 1 extra tool call(s) ignored" in tool_msg
    assert "return a - b" in tool_msg
    assert result.outcome.tool_calls == 1


def test_thinking_tokens_counted_separately(tmp_path):
    policy = scripted(["<think>a b c d</think>" + tc("finish")])
    result = run_episode(TASK, policy, workdir=tmp_path, token_counter=lambda s: len(s.split()))
    assert result.outcome.tokens > 0
    assert 0 < result.thinking_tokens < result.outcome.tokens
