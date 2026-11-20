#!/usr/bin/env python3
"""Generate biases on Colab (Unsloth, CUDA). Scoring happens afterwards with score_ladder.py.

Write mode (from scratch), with retries up to k attempts, stopping at the first bias XHAIL solves:
    resample   retry by sampling the same prompt again
    feedback   retry with a repair prompt: the last bias plus XHAIL's diagnosis of it
The first attempt is greedy and shared, so "single" is attempt 1 of either strategy.

    python colab_eval.py --model checkpoints/qwen-sft --tasks gold --strategy feedback --seed 0 --out preds/sft_gold_feedback_s0.jsonl

Repair mode (one greedy fix of a corrupted bias, from corrupt.py rows):
    python colab_eval.py --model checkpoints/qwen-sft --repair data/repair_test.jsonl --limit 300 --out preds/sft_repair_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

# XHAIL checks fork from a small helper started before the model loads (much faster than
# forking the GPU process). Must be set before common/feedback are imported.
os.environ.setdefault("XHAIL_MP_START", "forkserver")

from common import build_prompt, extract_modes, fewshot_messages, load_tasks, repair_prompt, retrieve_shots  # noqa: E402
from feedback import diagnose, run_phases  # noqa: E402


def make_generator(model_name: str, batch: int):
    from unsloth import FastLanguageModel  # imported late: the forkserver must start first

    model, tok = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=4096, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def generate(conversations: list[list[dict]], temp: float) -> list[str]:
        out = []
        for i in range(0, len(conversations), batch):
            texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations[i:i + batch]]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            kw = {"do_sample": True, "temperature": temp, "top_p": 0.95} if temp > 0 else {"do_sample": False}
            seqs = model.generate(**enc, max_new_tokens=384, pad_token_id=tok.pad_token_id, **kw)
            out += [tok.decode(s, skip_special_tokens=True) for s in seqs[:, enc["input_ids"].shape[1]:]]
        return out

    return generate


def write_mode(args, generate) -> list[dict]:
    tasks = load_tasks(args.tasks)[: args.limit]
    fixed = fewshot_messages(args.fewshot, seed=0)
    pool = load_tasks(args.shot_pool) if args.shots else []
    shots = {t["id"]: retrieve_shots(t["program"], pool, args.shots, seed=args.seed, exclude=t["id"]) if args.shots else fixed
             for t in tasks}

    def prompt(t):
        sys_msg, user = build_prompt(t["program"])
        return [sys_msg, *shots[t["id"]], user]

    state = {t["id"]: {"id": t["id"], "strategy": args.strategy, "seed": args.seed, "attempts": [], "solved_at": None}
             for t in tasks}
    active = list(tasks)
    for rnd in range(args.k):
        if rnd == 0:
            convs, temp = [prompt(t) for t in active], 0.0
        elif args.strategy == "resample":
            convs, temp = [prompt(t) for t in active], 0.7
        else:
            convs, temp = [repair_prompt(t["program"], state[t["id"]]["attempts"][-1]["completion"],
                                         state[t["id"]]["attempts"][-1]["feedback"]) for t in active], 0.7
            convs = [[c[0], *shots[t["id"]], c[1]] for c, t in zip(convs, active)]
        texts = generate(convs, temp)
        still = []
        for t, text in zip(active, texts):
            d = diagnose(t["program"], extract_modes(text))
            s = state[t["id"]]
            s["attempts"].append({"completion": text, "category": d["category"], "feedback": d["feedback"]})
            if d["solved"]:
                s["solved_at"] = rnd + 1
            else:
                still.append(t)
        active = still
        print(f"  round {rnd + 1}: {len(tasks) - len(active)}/{len(tasks)} solved", flush=True)
        if not active:
            break
    for s in state.values():
        solved = [a for a in s["attempts"] if a["category"] == "solved"]
        s["completion"] = solved[0]["completion"] if solved else s["attempts"][-1]["completion"]
    return list(state.values())


def repair_mode(args, generate) -> list[dict]:
    rows = [json.loads(line) for line in open(args.repair)]
    rows = [r for r in rows if r["failed_at"] != "none"]
    random.Random(0).shuffle(rows)
    rows = rows[: args.limit]
    texts = generate([repair_prompt(r["program"], r["broken"], r["feedback"]) for r in rows], 0.0)
    return [{"id": r["id"], "corruption": r["corruption"], "failed_at": r["failed_at"], "completion": t}
            for r, t in zip(rows, texts)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", help="JSONL path, 'benchmarks' or 'gold' (write mode)")
    ap.add_argument("--repair", help="corrupt.py JSONL (repair mode)")
    ap.add_argument("--strategy", choices=["resample", "feedback"], default="feedback")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot", type=int, default=0, help="fixed k-shot examples (one per family)")
    ap.add_argument("--shots", type=int, default=0, help="k retrieved examples of similar shape (overrides --fewshot)")
    ap.add_argument("--shot-pool", default="data/train.jsonl", help="tasks to retrieve examples from")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mp.get_context("forkserver").set_forkserver_preload(["common", "feedback"])
    run_phases("a.\n#example b.\n#modeh b.\n", timeout_s=30)  # start the forkserver while the process is small
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generate = make_generator(args.model, args.batch)
    rows = repair_mode(args, generate) if args.repair else write_mode(args, generate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
