#!/usr/bin/env python3
"""Score biases on the ladder: share of tasks that reach each level (cumulative).

    python score_ladder.py --tasks data/test.jsonl --method file --pred preds/ft_test.jsonl --name ft_test
    python score_ladder.py --tasks gold --method gold
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

from baselines import exhaustive_bias
from common import extract_modes, load_tasks
from evaluate import RESULTS_DIR
from ladder import LEVELS, climb
from prune import prune


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="JSONL path, 'benchmarks' or 'gold'")
    ap.add_argument("--method", choices=["gold", "exhaustive", "file"], required=True)
    ap.add_argument("--pred", help="predictions JSONL with {id, completion}")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prune", action="store_true", help="verify-then-prune each bias before scoring")
    args = ap.parse_args()

    tasks = load_tasks("gold" if args.tasks == "benchmarks" else args.tasks)
    if args.tasks == "benchmarks":
        tasks = [t for t in tasks if t["id"].startswith("bench-")]
    preds = {}
    if args.method == "file":
        with open(args.pred) as fh:
            preds = {r["id"]: r["completion"] for r in map(json.loads, fh)}

    rows = []
    for t in tasks:
        if args.method == "gold":
            text = "\n".join(t["gold_modes"])
        elif args.method == "exhaustive":
            text = "\n".join(exhaustive_bias(t["program"]))
        else:
            text = preds.get(t["id"], "")
        if args.prune and climb(t, text)["level"] >= 2:  # well-formed: safe to prune
            text = "\n".join(prune(t["program"], extract_modes(text)))
        r = climb(t, text)
        rows.append({"id": t["id"], "family": t["family"], "level": r["level"],
                     "failed": r["failed"] or "", "reason": r["reason"]})

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"ladder_{args.name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    reach = {LEVELS[i]: round(sum(r["level"] > i for r in rows) / n, 3) for i in range(5)}
    fails = Counter(r["failed"] or "none" for r in rows)
    summary = {"n": n, "reached": reach, "first_failure": dict(fails)}
    (RESULTS_DIR / f"ladder_{args.name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(args.name, json.dumps(summary))


if __name__ == "__main__":
    main()
