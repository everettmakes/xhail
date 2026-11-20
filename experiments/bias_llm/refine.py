#!/usr/bin/env python3
"""Propose a bias with an LLM, check it with XHAIL, and retry up to k times (MLX, Apple Silicon).

Three strategies, all with the same first attempt (greedy) and the same budget of k XHAIL calls:

    single     one attempt, no retry
    resample   retry by sampling a fresh answer to the same prompt (XHAIL only accepts/rejects)
    feedback   retry in the same conversation, with XHAIL's diagnosis of what went wrong

A task stops at the first bias XHAIL solves on the training examples. That bias (or the
last one tried) is then scored on held-out examples by evaluate.score, exactly as before.

    python refine.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --fewshot 3 \
        --tasks data/test.jsonl --strategy feedback --k 4 --name 7b_feedback_test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from common import build_prompt, extract_modes, fewshot_messages, load_tasks
from evaluate import RESULTS_DIR, score, summarise
from feedback import diagnose

HERE = Path(__file__).parent


def refine(task: dict, ask, strategy: str, k: int) -> dict:
    sys_msg, user_msg = build_prompt(task["program"])
    convo = [user_msg]
    attempts = []
    for i in range(1 if strategy == "single" else k):
        text = ask(convo, temp=0.0 if i == 0 else 0.7)
        modes = extract_modes(text)
        d = diagnose(task["program"], modes)
        attempts.append({"completion": text, "modes": modes, "category": d["category"], "feedback": d["feedback"]})
        if d["solved"]:
            break
        if strategy == "feedback":
            convo += [{"role": "assistant", "content": text},
                      {"role": "user", "content": "XHAIL could not learn from those modes.\n\n" + d["feedback"]
                       + "\n\nOutput the full corrected list of mode declarations only."}]
    return {"attempts": attempts, "modes": attempts[-1]["modes"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="MLX LoRA adapter dir")
    ap.add_argument("--tasks", required=True, help="JSONL path, or 'benchmarks'")
    ap.add_argument("--strategy", choices=["single", "resample", "feedback"], required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mem-gb", type=int, default=8, help="MLX memory cap")
    args = ap.parse_args()

    mx.set_memory_limit(args.mem_gb << 30)
    mx.set_cache_limit(1 << 30)
    mx.random.seed(args.seed)
    model, tok = load(args.model, adapter_path=args.adapter)
    shots = fewshot_messages(args.fewshot, seed=0)

    def ask(convo: list[dict], temp: float) -> str:
        msgs = [build_prompt("")[0], *shots, *convo]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt, max_tokens=256, sampler=make_sampler(temp=temp))

    tasks = load_tasks(args.tasks)[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    with open(RESULTS_DIR / f"{args.name}.transcripts.jsonl", "w") as tf:
        for i, t in enumerate(tasks, 1):
            r = refine(t, ask, args.strategy, args.k)
            row = score(t, r["modes"], timeout=20.0)
            row["attempts"] = len(r["attempts"])
            row["first_category"] = r["attempts"][0]["category"]
            rows.append(row)
            tf.write(json.dumps({"id": t["id"], **r}) + "\n")
            tf.flush()
            print(f"  {i}/{len(tasks)} {t['id']} attempts={row['attempts']} solved={row['solved']} "
                  f"acc={row['acc']}", flush=True)

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    summary = {}
    for key, subset in [("overall", rows), *sorted(by_family.items())]:
        s = summarise(subset)
        s["mean_attempts"] = round(sum(r["attempts"] for r in subset) / len(subset), 2)
        summary[key] = s
    with open(RESULTS_DIR / f"{args.name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS_DIR / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
