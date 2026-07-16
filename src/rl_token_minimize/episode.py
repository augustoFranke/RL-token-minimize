import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .reward import EpisodeOutcome, RewardConfig, compute_reward, patch_similarity
from .tasks import Task
from .tools import Workspace
from .verifier import run_tests

SYSTEM_PROMPT = (
    "You are a coding agent that fixes bugs. Use the available tools to inspect the "
    "workspace and make surgical edits with replace_text. Make one tool call per turn. "
    "When the bug is fixed, call finish. Be economical: read only what you need and "
    "keep your output short."
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict] | None:
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        try:
            call = json.loads(block)
        except json.JSONDecodeError:
            return None
        if not isinstance(call, dict) or "name" not in call:
            return None
        call.setdefault("arguments", {})
        calls.append({"name": call["name"], "arguments": call["arguments"]})
    return calls


@dataclass
class EpisodeResult:
    outcome: EpisodeOutcome
    reward: float
    messages: list[dict]
    verifier_output: str
    thinking_tokens: int


def run_episode(
    task: Task,
    policy: Callable[[list[dict]], str],
    workdir: Path,
    max_turns: int = 16,
    reward_config: RewardConfig | None = None,
    token_counter: Callable[[str], int] = lambda s: len(s) // 4,
) -> EpisodeResult:
    cfg = reward_config or RewardConfig()
    workspace_dir = Path(workdir) / task.task_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    for name, content in task.files.items():
        target = workspace_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    ws = Workspace(workspace_dir)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    tokens = thinking_tokens = tool_calls = failed_replaces = 0
    protocol_error = finished = False

    for _ in range(max_turns):
        text = policy(messages)
        tokens += token_counter(text)
        thinking_tokens += sum(token_counter(t) for t in THINK_RE.findall(text))
        messages.append({"role": "assistant", "content": text})
        calls = parse_tool_calls(text)
        if not calls:
            protocol_error = True
            break
        call = calls[0]
        if call["name"] == "finish":
            finished = True
            break
        tool_calls += 1
        result = ws.call(call["name"], call["arguments"])
        if call["name"] == "replace_text" and result.startswith("error:"):
            failed_replaces += 1
        if len(calls) > 1:
            result += f"\nnote: {len(calls) - 1} extra tool call(s) ignored; make one tool call per turn."
        messages.append({"role": "tool", "content": result})

    final_files = ws.snapshot()
    changed = ws.changed_files(task.files)
    verifier = run_tests(workspace_dir, task.test_code)
    outcome = EpisodeOutcome(
        protocol_error=protocol_error,
        finished=finished,
        edited=bool(changed),
        tests_passed=verifier.passed,
        touched_expected=bool(set(changed) & set(task.expected_files)),
        similarity=patch_similarity(final_files, task.reference_files),
        unrelated_edits=bool(set(changed) - set(task.expected_files)),
        tokens=tokens,
        tool_calls=tool_calls,
        failed_replaces=failed_replaces,
    )
    return EpisodeResult(
        outcome=outcome,
        reward=compute_reward(outcome, cfg),
        messages=messages,
        verifier_output=verifier.output,
        thinking_tokens=thinking_tokens,
    )
