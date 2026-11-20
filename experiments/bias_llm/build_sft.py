#!/usr/bin/env python3
"""Build chat SFT data with two jobs: write a bias from scratch, and repair a broken one.

    write:   program + vocabulary                           -> gold modes
    repair:  program + vocabulary + broken modes + feedback -> gold modes

Repair rows come from corrupt.py. Rows whose break turned out harmless (failed_at == none)
are dropped, and each failure level is capped so no single kind of error dominates.

    python build_sft.py --split train --cap 500
    python build_sft.py --split train --cap 600 --with-new --tag _v3   # adds data/{split}_new.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import build_prompt, load_tasks, repair_prompt

DATA = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--cap", type=int, default=500, help="max repair rows per failure level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-new", action="store_true", help="also use data/{split}_new.jsonl and repair_{split}_new.jsonl")
    ap.add_argument("--tag", default="", help="output suffix, e.g. _v3")
    args = ap.parse_args()
    parts = [args.split] + ([f"{args.split}_new"] if args.with_new else [])

    rng = random.Random(args.seed)
    rows = []
    for t in [t for p in parts for t in load_tasks(str(DATA / f"{p}.jsonl"))]:
        rows.append({"job": "write", "id": t["id"],
                     "messages": [*build_prompt(t["program"]), {"role": "assistant", "content": "\n".join(t["gold_modes"])}]})

    repairs = [json.loads(line) for p in parts for line in open(DATA / f"repair_{p}.jsonl")]
    repairs = [r for r in repairs if r["failed_at"] != "none"]
    rng.shuffle(repairs)
    per_level: Counter = Counter()
    for r in repairs:
        if per_level[r["failed_at"]] >= args.cap:
            continue
        per_level[r["failed_at"]] += 1
        rows.append({"job": "repair", "id": r["id"], "failed_at": r["failed_at"],
                     "messages": [*repair_prompt(r["program"], r["broken"], r["feedback"]),
                                  {"role": "assistant", "content": r["gold"]}]})

    rng.shuffle(rows)
    out = DATA / f"sft_{args.split}{args.tag}.jsonl"
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {out}: write={sum(r['job'] == 'write' for r in rows)}, repair by level={dict(per_level)}")


if __name__ == "__main__":
    main()
