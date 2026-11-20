#!/usr/bin/env python3
"""Generate mode declarations for a task set with a (fine-tuned) model.

    # fine-tuned adapter
    python experiments/bias_llm/predict.py --model experiments/bias_llm/checkpoints/lora \
        --tasks experiments/bias_llm/data/test.jsonl --out experiments/bias_llm/preds/ft_test.jsonl

    # untuned base model with k in-context examples (prompting baseline)
    python experiments/bias_llm/predict.py --model unsloth/Qwen2.5-1.5B-Instruct --fewshot 3 \
        --tasks benchmarks --out experiments/bias_llm/preds/base3shot_bench.jsonl

Writes JSONL rows {"id", "completion"} for evaluate.py --method file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unsloth import FastLanguageModel

from common import build_prompt, fewshot_messages, load_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="adapter dir or HF model id")
    ap.add_argument("--tasks", required=True, help="JSONL path, or 'benchmarks'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=6144)
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_len, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    shots = fewshot_messages(args.fewshot, seed=0)
    tasks = load_tasks(args.tasks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for i in range(0, len(tasks), args.batch):
            chunk = tasks[i : i + args.batch]
            texts = []
            for t in chunk:
                sys_msg, user_msg = build_prompt(t["program"])
                texts.append(tok.apply_chat_template([sys_msg, *shots, user_msg],
                                                     tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for t, seq in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
                fh.write(json.dumps({"id": t["id"], "completion": tok.decode(seq, skip_special_tokens=True)}) + "\n")
            print(f"  {min(i + args.batch, len(tasks))}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
