#!/usr/bin/env python3
"""QLoRA SFT on gold modes with MLX (Apple Silicon). Same recipe as train_unsloth.py.

Batch 4 x 4 accumulation steps (effective 16) and a hard MLX memory cap keep peak memory
low; an uncapped batch of 16 alongside other GPU work crashed a 16 GB MacBook Air.

    python train_mlx.py --out checkpoints/mlx-qwen1.5b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import build_prompt

HERE = Path(__file__).parent


def write_chat_jsonl(src: Path, dst: Path) -> int:
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            t = json.loads(line)
            msgs = [*build_prompt(t["program"]), {"role": "assistant", "content": "\n".join(t["gold_modes"])}]
            fout.write(json.dumps({"messages": msgs}) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--out", default=str(HERE / "checkpoints" / "mlx-qwen1.5b"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--mem-gb", type=int, default=6, help="MLX memory cap")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-iters", type=int, help="stop early (quick diagnostic runs)")
    args = ap.parse_args()

    out = Path(args.out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    n_train = write_chat_jsonl(HERE / "data" / "train.jsonl", data / "train.jsonl")
    write_chat_jsonl(HERE / "data" / "val.jsonl", data / "valid.jsonl")
    iters = int(args.epochs * n_train / args.batch)  # micro-batches
    updates = iters // args.grad_accum
    if args.max_iters:
        iters = args.max_iters

    config = {
        "model": args.model, "train": True, "data": str(data), "adapter_path": str(out),
        "fine_tune_type": "lora", "num_layers": -1, "mask_prompt": True,
        "batch_size": args.batch, "grad_accumulation_steps": args.grad_accum, "iters": iters, "learning_rate": args.lr,
        "lr_schedule": {"name": "cosine_decay", "arguments": [args.lr, updates], "warmup": 10},
        "lora_parameters": {"rank": args.rank, "scale": 1.0, "dropout": 0.0,
                            "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]},
        "max_seq_length": 2048, "grad_checkpoint": True, "seed": 3407,
        "steps_per_report": 4, "steps_per_eval": 1000, "val_batches": 10, "save_every": 1000,
    }
    (out / "train_config.yaml").write_text(json.dumps(config, indent=2))  # JSON is valid YAML
    capped = ("import sys, mlx.core as mx; from mlx_lm.lora import main; "
              f"mx.set_memory_limit({args.mem_gb} << 30); mx.set_cache_limit(1 << 30); "
              f"sys.argv = ['lora', '--config', {str(out / 'train_config.yaml')!r}]; main()")
    subprocess.run([sys.executable, "-c", capped], check=True)


if __name__ == "__main__":
    main()
