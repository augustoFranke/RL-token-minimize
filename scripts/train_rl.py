"""GRPO-style RL on the episode harness, starting from the SFT adapter.

Own-loop policy gradient: G sampled rollouts per task, group-normalized
advantages, log-prob recomputation over generated tokens only. On-policy with
one update per batch, so no importance ratio or clipping is needed.

Runs from the handoff:
  1. --no-thinking
  2. --thinking --token-cost all      (raw token penalty)
  3. --thinking --token-cost final    (reasoning excluded from cost; both tracked)
"""

import argparse
import dataclasses
import json
import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from rl_token_minimize.episode import run_episode
from rl_token_minimize.modeling import MODEL_NAME, load_model_and_tokenizer
from rl_token_minimize.policy import HFPolicy
from rl_token_minimize.reward import RewardConfig, compute_reward
from rl_token_minimize.tasks import Task

DATA_DIR = Path(__file__).parent.parent / "data"


def episode_logprob(model, prompt_ids, gen_ids):
    input_ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(model.device)
    logits = model(input_ids).logits[0, prompt_ids.shape[0] - 1 : -1]
    return F.log_softmax(logits, dim=-1).gather(1, gen_ids.unsqueeze(1).to(model.device)).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="checkpoints/sft")
    parser.add_argument("--output", default="checkpoints/rl")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tasks-per-step", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--token-cost", choices=["all", "final"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, adapter_path=args.adapter)
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
    # fp32 activations over multi-thousand-token episodes OOM a 16GB T4
    # without checkpointing; generate() is unaffected (runs under no_grad)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    tasks = [Task(**json.loads(line)) for line in open(DATA_DIR / "tasks_train.jsonl")]
    rng = random.Random(args.seed)
    cfg = RewardConfig()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.jsonl", "a")

    for step in range(args.steps):
        # eval mode for rollouts: keeps KV-cache generation (checkpointing
        # forces cache off in train mode) and turns off LoRA dropout
        model.eval()
        rollouts = []
        stats = {"reward": [], "passed": [], "tokens": [], "thinking_tokens": []}
        with tempfile.TemporaryDirectory() as tmp:
            for t_idx, task in enumerate(rng.sample(tasks, args.tasks_per_step)):
                group = []
                for g in range(args.group_size):
                    policy = HFPolicy(model, tokenizer, thinking=args.thinking,
                                      max_new_tokens=args.max_new_tokens, do_sample=True)
                    result = run_episode(
                        task, policy, workdir=Path(tmp) / f"{t_idx}_{g}",
                        max_turns=args.max_turns, reward_config=cfg,
                        token_counter=policy.token_counter,
                    )
                    reward = result.reward
                    if args.token_cost == "final":
                        adjusted = dataclasses.replace(
                            result.outcome, tokens=result.outcome.tokens - result.thinking_tokens
                        )
                        reward = compute_reward(adjusted, cfg)
                    group.append((policy.segments, reward))
                    stats["reward"].append(reward)
                    stats["passed"].append(result.outcome.tests_passed)
                    stats["tokens"].append(result.outcome.tokens)
                    stats["thinking_tokens"].append(result.thinking_tokens)
                rewards = torch.tensor([r for _, r in group])
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
                rollouts.extend(
                    (segments, adv.item()) for (segments, _), adv in zip(group, advantages)
                )

        model.train()
        optimizer.zero_grad()
        active = [(s, a) for s, a in rollouts if abs(a) > 1e-6 and s]
        for segments, adv in active:
            gen_tokens = sum(len(g) for _, g in segments)
            for prompt_ids, gen_ids in segments:
                logp = episode_logprob(model, prompt_ids, gen_ids)
                (-adv * logp / gen_tokens / len(active)).backward()
        if active:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        record = {
            "step": step,
            "mean_reward": sum(stats["reward"]) / len(stats["reward"]),
            "pass_rate": sum(stats["passed"]) / len(stats["passed"]),
            "mean_tokens": sum(stats["tokens"]) / len(stats["tokens"]),
            "mean_thinking_tokens": sum(stats["thinking_tokens"]) / len(stats["thinking_tokens"]),
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(record)
        if step % 20 == 19:
            model.save_pretrained(out_dir / f"step_{step + 1}")

    model.save_pretrained(out_dir / "final")
    print(f"saved to {out_dir / 'final'}")


if __name__ == "__main__":
    main()
