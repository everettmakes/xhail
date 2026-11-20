#!/usr/bin/env python3
"""QLoRA fine-tune a small instruct model to write XHAIL mode declarations.

Runs on one GPU (a free Colab T4 is enough for 1.5B). Each training row is a
conversational prompt/completion pair:

    system: SYSTEM_PROMPT
    user:   task program with modes removed (background + #examples)
    asst:   gold #modeh / #modeb lines

Loss is on the completion only (TRL's default for prompt/completion data).

Usage (Colab):
    pip install unsloth
    python experiments/bias_llm/train_unsloth.py --data experiments/bias_llm/data \
        --model unsloth/Qwen2.5-1.5B-Instruct --epochs 2 --out experiments/bias_llm/checkpoints/qwen1.5b-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unsloth import FastLanguageModel  # must be imported before transformers/trl

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from common import build_prompt


def sft_rows(path: Path) -> Dataset:
    """Rows from build_sft.py: chat messages whose last turn is the target completion."""
    rows = []
    with open(path) as fh:
        for line in fh:
            msgs = json.loads(line)["messages"]
            rows.append({"prompt": msgs[:-1], "completion": msgs[-1:]})
    return Dataset.from_list(rows)


def to_rows(path: Path) -> Dataset:
    rows = []
    with open(path) as fh:
        for line in fh:
            t = json.loads(line)
            rows.append({
                "prompt": build_prompt(t["program"]),
                "completion": [{"role": "assistant", "content": "\n".join(t["gold_modes"])}],
            })
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=str(Path(__file__).parent / "checkpoints" / "lora"))
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--sft", action="store_true", help="train on data/sft_{train,val}{tag}.jsonl (write + repair)")
    ap.add_argument("--sft-tag", default="", help="suffix of the SFT files, e.g. _v3")
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_len, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    data = Path(args.data)
    if args.sft:
        train_ds = sft_rows(data / f"sft_train{args.sft_tag}.jsonl")
        val_ds = sft_rows(data / f"sft_val{args.sft_tag}.jsonl")
    else:
        train_ds, val_ds = to_rows(data / "train.jsonl"), to_rows(data / "val.jsonl")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=args.out,
            max_length=args.max_len,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optim="adamw_8bit",
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="no",
            seed=args.seed,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
