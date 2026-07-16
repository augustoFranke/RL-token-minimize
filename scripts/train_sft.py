"""SFT on canonical tool-call traces with LoRA.

Renders each trace through the Qwen3 chat template (tool schemas included, so
the training-time prompt matches inference) and trains a LoRA adapter.
Goal: teach the tool-call format well enough for RL, not maximize SFT quality.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from rl_token_minimize.modeling import MODEL_NAME, load_model_and_tokenizer, lora_config
from rl_token_minimize.tools import TOOL_SCHEMAS

DATA_DIR = Path(__file__).parent.parent / "data"


class MPSMemoryGuard(TrainerCallback):
    # near the MPS watermark, Metal command buffers fail silently and training
    # continues with NaN tensors — fail fast and keep the allocator drained
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm"):
            if logs and key in logs and not math.isfinite(float(logs[key])):
                raise RuntimeError(f"non-finite {key} at step {state.global_step}: {logs}")

    def on_step_end(self, args, state, control, **kwargs):
        # per-step only: emptying the cache between grad-accum micro-batches
        # frees buffers still referenced by in-flight MPS work and corrupts
        # training deterministically
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap examples for smoke runs")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    rows = [json.loads(line) for line in open(DATA_DIR / "sft_train.jsonl")]
    if args.limit:
        rows = rows[: args.limit]
    texts = [
        tokenizer.apply_chat_template(
            row["messages"], tools=TOOL_SCHEMAS, tokenize=False, enable_thinking=False
        )
        for row in rows
    ]
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            max_length=4096,
            # TRL enables bf16 autocast by default; on MPS that yields NaN grads
            bf16=False,
            fp16=False,
            # fp32 activations for ~1k-token sequences blow past the 16GB MPS
            # watermark, corrupting Metal command buffers instead of OOMing
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # the default fused AdamW intermittently NaNs on MPS (torch 2.13)
            optim="adamw_torch",
        ),
        train_dataset=dataset,
        peft_config=lora_config(),
        processing_class=tokenizer,
        callbacks=[MPSMemoryGuard()],
    )
    trainer.train()
    trainer.save_model(args.output)
    print(f"saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
