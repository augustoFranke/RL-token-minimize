"""SFT on canonical tool-call traces with LoRA.

Renders each trace through the Qwen3 chat template (tool schemas included, so
the training-time prompt matches inference) and trains a LoRA adapter.
Goal: teach the tool-call format well enough for RL, not maximize SFT quality.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from rl_token_minimize.modeling import MODEL_NAME, load_model_and_tokenizer, lora_config
from rl_token_minimize.tools import TOOL_SCHEMAS

DATA_DIR = Path(__file__).parent.parent / "data"


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
        ),
        train_dataset=dataset,
        peft_config=lora_config(),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    print(f"saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
