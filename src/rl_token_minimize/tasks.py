import re
from dataclasses import dataclass, field

SOLUTION_FILE = "solution.py"
MAX_FAILURE_LINES = 15


def concise_failure(output: str) -> str:
    output = re.sub(r'"/[^"]*/([^/"]+)"', r'"\1"', output)
    return "\n".join(output.splitlines()[-MAX_FAILURE_LINES:])

PROMPT_TEMPLATE = (
    "A test is failing in this workspace. The bug is in the function `{entry_point}`.\n\n"
    "Failing test output:\n{failure_output}\n\n"
    "Find and fix the bug, then call finish."
)


@dataclass
class Task:
    task_id: str
    files: dict[str, str]
    reference_files: dict[str, str]
    prompt: str
    test_code: str
    expected_files: list[str] = field(default_factory=list)


def build_humanevalfix_task(row: dict, failure_output: str) -> Task:
    entry_point = row["entry_point"]
    test_code = (
        f"from solution import {entry_point}\n\n"
        f"{row['test']}\n\ncheck({entry_point})\n"
    )
    return Task(
        task_id=row["task_id"].replace("/", "_"),
        files={SOLUTION_FILE: row["prompt"] + row["buggy_solution"]},
        reference_files={SOLUTION_FILE: row["prompt"] + row["canonical_solution"]},
        prompt=PROMPT_TEMPLATE.format(entry_point=entry_point, failure_output=failure_output),
        test_code=test_code,
        expected_files=[SOLUTION_FILE],
    )
