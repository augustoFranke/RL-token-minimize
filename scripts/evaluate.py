"""Evaluate a policy on the eval task split: pass rate, token, and tool metrics.

Run on the base model, after SFT, and after RL with the same flags to compare
quality vs token output.
"""

import argparse
import json
import tempfile
from pathlib import Path

from rl_token_minimize.episode import run_episode
from rl_token_minimize.modeling import MODEL_NAME, load_model_and_tokenizer
from rl_token_minimize.policy import HFPolicy
from rl_token_minimize.tasks import Task

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--split", default="eval", choices=["eval", "train"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--out", default=None, help="write per-task results to this JSONL file")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, adapter_path=args.adapter)
    model.eval()
    tasks = [Task(**json.loads(line)) for line in open(DATA_DIR / f"tasks_{args.split}.jsonl")]
    if args.limit:
        tasks = tasks[: args.limit]

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, task in enumerate(tasks):
            policy = HFPolicy(model, tokenizer, thinking=args.thinking,
                              max_new_tokens=args.max_new_tokens)
            result = run_episode(task, policy, workdir=Path(tmp) / str(i),
                                 max_turns=args.max_turns, token_counter=policy.token_counter)
            o = result.outcome
            results.append({
                "task_id": task.task_id, "passed": o.tests_passed, "reward": result.reward,
                "tokens": o.tokens, "thinking_tokens": result.thinking_tokens,
                "tool_calls": o.tool_calls, "failed_replaces": o.failed_replaces,
                "protocol_error": o.protocol_error, "finished": o.finished,
            })
            print(f"[{i + 1}/{len(tasks)}] {task.task_id}: "
                  f"{'PASS' if o.tests_passed else 'fail'} tokens={o.tokens}")

    n = len(results)
    summary = {
        "n": n,
        "adapter": args.adapter,
        "thinking": args.thinking,
        "pass_rate": sum(r["passed"] for r in results) / n,
        "mean_reward": sum(r["reward"] for r in results) / n,
        "mean_tokens": sum(r["tokens"] for r in results) / n,
        "mean_thinking_tokens": sum(r["thinking_tokens"] for r in results) / n,
        "mean_tool_calls": sum(r["tool_calls"] for r in results) / n,
        "protocol_error_rate": sum(r["protocol_error"] for r in results) / n,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
