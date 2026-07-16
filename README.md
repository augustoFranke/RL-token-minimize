# RL-token-minimize

Train a small local coding agent (Qwen3-0.6B + LoRA) to fix bugs via tool calls,
then use RL to reduce token/tool output while preserving repair quality.
Designed to run on a 16GB MacBook Pro (MPS).

## Design (locked in during planning — see handoff doc)

- **Task**: agentic code repair. The model gets a short issue prompt with failing
  test output, explores the workspace with tools, and makes surgical edits.
- **Tools exposed to the model (v1)**: `list_files`, `search_text`, `read_file`,
  `replace_text`, `finish`. No shell, no `run_tests`, no patch/diff submission,
  no `write_file`. `replace_text` requires a unique match.
- **Hidden verifier**: dataset-provided tests run in a subprocess after the
  episode ends. The model never sees or runs them.
- **Data**: HumanEvalFix (`bigcode/humanevalpack`, python) — real buggy/fixed
  pairs with runnable tests. Tasks are filtered: buggy must fail, reference must
  pass, and the reference patch must convert to clean unique `replace_text`
  calls. 161/164 tasks survive (121 train / 40 eval).
- **SFT**: canonical tool-call traces generated *mechanically* from reference
  patches (difflib hunks → `replace_text` calls, context-expanded until unique),
  replayed against a real workspace so tool outputs are genuine. Goal is to
  teach the format for RL, not maximize SFT quality.
- **RL**: own-loop GRPO (group-relative advantages, log-probs over generated
  tokens only), TRL/PEFT for the model side only.
- **Reward**: deterministic, correctness-gated, shaped. Tests passing is
  checked first, so a pass always scores ≥ 0.5 regardless of how the episode
  ended (protocol error, turn exhaustion, etc.):
  - tests pass → 1.0 + token bonus − penalties (extra tool calls, failed
    `replace_text`, unrelated edits, and a small penalty for not ending with a
    clean `finish` call), floored at 0.5 so any pass beats any miss
  - invalid tool protocol (and tests don't pass) → −1.0
  - finished without editing → −0.5
  - edited but tests fail → 0.0, or +0.2 if the patch touches the expected file
    and resembles the reference (difflib similarity)

## Layout

```
src/rl_token_minimize/
  tools.py      # the 5 tools + Workspace (path-jailed, unique-match replace)
  tasks.py      # Task schema + HumanEvalFix adapter + failure-output sanitizer
  traces.py     # mechanical canonical trace generation from reference patches
  verifier.py   # hidden test runner (subprocess, timeout, truncated output)
  reward.py     # RewardConfig + compute_reward + patch_similarity
  episode.py    # agent loop: policy ↔ tools, tool-call parsing, outcome stats
  sft_data.py   # trace → chat messages with real replayed tool outputs
  policy.py     # HFPolicy: chat-template generation, records token segments for RL
  modeling.py   # model/tokenizer/LoRA loading (MPS-aware)
scripts/
  prepare_data.py  # build data/tasks_*.jsonl + data/sft_train.jsonl
  train_sft.py     # LoRA SFT on canonical traces (TRL SFTTrainer)
  train_rl.py      # GRPO loop over episodes
  evaluate.py      # pass rate + token/tool metrics on the eval split
```

## Workflow

```bash
uv run pytest                        # 59 tests over the deterministic core
uv run scripts/prepare_data.py      # regenerate datasets
uv run scripts/evaluate.py --limit 5             # baseline (base model)
uv run scripts/train_sft.py                       # → checkpoints/sft
uv run scripts/evaluate.py --adapter checkpoints/sft   # post-SFT baseline
# the three runs from the plan:
uv run scripts/train_rl.py --no-thinking --output checkpoints/rl_run1
uv run scripts/train_rl.py --thinking --token-cost all --output checkpoints/rl_run2
uv run scripts/train_rl.py --thinking --token-cost final --output checkpoints/rl_run3
uv run scripts/evaluate.py --adapter checkpoints/rl_run1/final
```

`train_rl.py` logs per-step `mean_reward` / `pass_rate` / `mean_tokens` /
`mean_thinking_tokens` to `<output>/log.jsonl`.

## Known limitations / next steps

- **Corpus size**: 121 training tasks is below the 300–500 smoke minimum from
  the plan. The `Task` schema is dataset-agnostic — add adapters (e.g. filtered
  SWE-bench-style or CommitPackFT-derived tasks) in `tasks.py` to scale.
- **SFT loss** is over the full rendered conversation (Qwen3's chat template has
  no `{% generation %}` markers for assistant-only masking). Acceptable for
  format-teaching; revisit if SFT overfits to tool outputs.
- **Assistant message fidelity**: at inference the episode stores assistant
  turns as raw text; SFT stores structured `tool_calls`. Both render to the same
  `<tool_call>` format, but Qwen3's template strips prior-turn `<think>` blocks
  only in the structured path — worth checking once thinking-mode runs start.
- **MPS precision**: TRL/accelerate silently enables bf16 autocast, which
  produces NaN gradients on MPS — `train_sft.py` forces it off and models load
  in fp32 on MPS. If you move to CUDA, re-enable bf16 for speed.
- **Training scripts are smoke-tested for plumbing, not convergence** — do a
  `--limit`/low-`--steps` run first.
