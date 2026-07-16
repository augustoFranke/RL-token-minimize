"""Build task + SFT trace datasets from HumanEvalFix (bigcode/humanevalpack).

Keeps only tasks where: the buggy code fails the tests, the reference code
passes, and the reference patch converts to clean unique replace_text calls.
Writes data/tasks_{train,eval}.jsonl and data/sft_train.jsonl.
"""

import dataclasses
import json
import random
import tempfile
from pathlib import Path

from datasets import load_dataset

from rl_token_minimize.sft_data import build_sft_example
from rl_token_minimize.tasks import build_humanevalfix_task, concise_failure
from rl_token_minimize.verifier import run_tests

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_SIZE = 40
SEED = 0


def run_on(files: dict[str, str], test_code: str):
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            (Path(tmp) / name).write_text(content)
        return run_tests(Path(tmp), test_code)


def main():
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    tasks, sft_examples = [], {}
    skipped = {"buggy_passes": 0, "fixed_fails": 0, "no_clean_trace": 0}
    for row in ds:
        task = build_humanevalfix_task(row, failure_output="")
        buggy_result = run_on(task.files, task.test_code)
        if buggy_result.passed:
            skipped["buggy_passes"] += 1
            continue
        if not run_on(task.reference_files, task.test_code).passed:
            skipped["fixed_fails"] += 1
            continue
        task = build_humanevalfix_task(row, failure_output=concise_failure(buggy_result.output))
        example = build_sft_example(task)
        if example is None:
            skipped["no_clean_trace"] += 1
            continue
        tasks.append(task)
        sft_examples[task.task_id] = example

    random.Random(SEED).shuffle(tasks)
    eval_tasks, train_tasks = tasks[:EVAL_SIZE], tasks[EVAL_SIZE:]

    DATA_DIR.mkdir(exist_ok=True)
    for name, split in [("tasks_train", train_tasks), ("tasks_eval", eval_tasks)]:
        with open(DATA_DIR / f"{name}.jsonl", "w") as f:
            for task in split:
                f.write(json.dumps(dataclasses.asdict(task)) + "\n")
    with open(DATA_DIR / "sft_train.jsonl", "w") as f:
        for task in train_tasks:
            f.write(json.dumps(sft_examples[task.task_id]) + "\n")

    print(f"kept {len(tasks)} tasks ({len(train_tasks)} train / {len(eval_tasks)} eval)")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
