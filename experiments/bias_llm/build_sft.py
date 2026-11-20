#!/usr/bin/env python3
"""Build chat SFT data.

Bias job (default), two kinds of row:
    write:   program + vocabulary                           -> gold modes
    repair:  program + vocabulary + broken modes + feedback -> gold modes
Rule job (--job rule), the direct-rule baseline:
    rule:    program + vocabulary                           -> the intended rule(s)

Repair rows come from corrupt.py. Rows whose break turned out harmless (failed_at == none)
are dropped, and each failure level is capped so no single kind of error dominates.

    python build_sft.py --split train --cap 500
    python build_sft.py --split train --cap 600 --extra _new --tag _v3          # data/{split}_new.jsonl too
    python build_sft.py --split train --cap 600 --extra _new,_v4 --tag _v4
    python build_sft.py --split train --extra _new,_v4 --job rule --tag _rule_v4
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import build_prompt, load_tasks, repair_prompt
from rules import rule_prompt, rule_text

DATA = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--job", choices=["bias", "rule"], default="bias")
    ap.add_argument("--cap", type=int, default=500, help="max repair rows per failure level (bias job)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="", help="comma-separated extra data suffixes, e.g. _new,_v4")
    ap.add_argument("--with-new", action="store_true", help="same as --extra _new (kept for the v3 notebook)")
    ap.add_argument("--tag", default="", help="output suffix, e.g. _v4")
    args = ap.parse_args()
    extras = [e for e in args.extra.split(",") if e] + (["_new"] if args.with_new else [])
    parts = [args.split] + [f"{args.split}{e}" for e in extras]

    rng = random.Random(args.seed)
    rows = []
    for t in [t for p in parts for t in load_tasks(str(DATA / f"{p}.jsonl"))]:
        if args.job == "rule":
            rows.append({"job": "rule", "id": t["id"],
                         "messages": [*rule_prompt(t["program"]), {"role": "assistant", "content": rule_text(t)}]})
        else:
            rows.append({"job": "write", "id": t["id"],
                         "messages": [*build_prompt(t["program"]), {"role": "assistant", "content": "\n".join(t["gold_modes"])}]})

    per_level: Counter = Counter()
    if args.job == "bias":
        repairs = [json.loads(line) for p in parts for line in open(DATA / f"repair_{p}.jsonl")]
        repairs = [r for r in repairs if r["failed_at"] != "none"]
        rng.shuffle(repairs)
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
    print(f"wrote {len(rows)} rows to {out}: {dict(Counter(r['job'] for r in rows))}, repair by level={dict(per_level)}")


if __name__ == "__main__":
    main()
