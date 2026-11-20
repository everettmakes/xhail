#!/usr/bin/env python3
"""Score a bias generator by what XHAIL does with its mode declarations.

Methods:
    gold        the task's own gold modes (upper reference)
    exhaustive  typed declare-everything heuristic (baselines.py)
    file        predictions JSONL from predict.py: {"id", "completion"}

Per-task metrics:
    valid       output contained >= 1 #modeh and XHAIL parsed it
    solved      XHAIL returned a hypothesis within the timeout
    acc         held-out example accuracy of the learned rules (0 if unsolved)
    hyp_match   learned hypothesis == hypothesis learned from gold modes
    runtime_s   XHAIL wall time (search cost of the bias)
    n_modes     size of the bias
    mode_p/r/f1 overlap with gold mode set

Usage:
    python experiments/bias_llm/evaluate.py --tasks experiments/bias_llm/data/test.jsonl --method exhaustive
    python experiments/bias_llm/evaluate.py --tasks benchmarks --method file --pred preds/bench.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from baselines import exhaustive_bias
from common import (
    background_only,
    canon_hypothesis,
    extract_modes,
    holdout_accuracy,
    load_tasks,
    mode_prf,
    run_xhail,
    splice,
)

RESULTS_DIR = Path(__file__).parent / "results"


def predict_modes(task: dict, method: str, preds: dict[str, str]) -> list[str]:
    if method == "gold":
        return task["gold_modes"]
    if method == "exhaustive":
        return exhaustive_bias(task["program"])
    return extract_modes(preds.get(task["id"], ""))


def score(task: dict, modes: list[str], timeout: float) -> dict:
    row = {"id": task["id"], "family": task["family"], "n_modes": len(modes)}
    row["mode_p"], row["mode_r"], row["mode_f1"] = mode_prf(modes, task["gold_modes"])
    has_head = any(m.startswith("#modeh") for m in modes)
    res = run_xhail(splice(task["program"], modes), timeout_s=timeout) if has_head else {"ok": False, "error": "no #modeh"}
    row["valid"] = bool(has_head and (res.get("ok") or res.get("timeout")))
    row["timeout"] = bool(res.get("timeout"))
    row["solved"] = bool(res.get("ok") and res.get("success"))
    row["runtime_s"] = res.get("runtime_s", float("nan"))
    row["hypothesis"] = " | ".join(res.get("hypothesis", []))
    row["error"] = res.get("error", "")
    if row["solved"] and task["holdout"]:
        row["acc"] = holdout_accuracy(background_only(task["program"]), res["hypothesis"], task["holdout"])
    else:
        row["acc"] = None if not task["holdout"] else 0.0
    gold_h = task.get("gold_hypothesis")
    if gold_h is None:  # benchmarks: learn once with gold modes
        g = run_xhail(splice(task["program"], task["gold_modes"]), timeout_s=timeout)
        gold_h = g.get("hypothesis", [])
    row["hyp_match"] = row["solved"] and canon_hypothesis(res["hypothesis"]) == canon_hypothesis(gold_h)
    return row


def summarise(rows: list[dict]) -> dict:
    def mean(key, subset=None):
        vals = [r[key] for r in (subset or rows) if r[key] is not None]
        return round(st.mean(vals), 3) if vals else None

    solved = [r for r in rows if r["solved"]]
    return {
        "n": len(rows),
        "valid": mean("valid"),
        "solved": mean("solved"),
        "acc": mean("acc"),
        "hyp_match": mean("hyp_match"),
        "timeouts": sum(r["timeout"] for r in rows),
        "median_runtime_s_solved": round(st.median([r["runtime_s"] for r in solved]), 4) if solved else None,
        "mean_n_modes": mean("n_modes"),
        "mode_f1": mean("mode_f1"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="JSONL path, or 'benchmarks'")
    ap.add_argument("--method", choices=["gold", "exhaustive", "file"], required=True)
    ap.add_argument("--pred", help="predictions JSONL (method=file)")
    ap.add_argument("--name", help="run name for output files")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)[: args.limit]
    preds = {}
    if args.method == "file":
        with open(args.pred) as fh:
            preds = {r["id"]: r["completion"] for r in map(json.loads, fh)}

    rows = []
    for i, t in enumerate(tasks, 1):
        rows.append(score(t, predict_modes(t, args.method, preds), args.timeout))
        if i % 25 == 0:
            print(f"  {i}/{len(tasks)}", flush=True)

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    summary = {"overall": summarise(rows), **{f: summarise(v) for f, v in sorted(by_family.items())}}

    name = args.name or f"{Path(args.tasks).stem}_{args.method}"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS_DIR / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))

    cols = list(summary["overall"])
    print(f"\n{name}\n| split | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for k, s in summary.items():
        print(f"| {k} | " + " | ".join("" if s[c] is None else str(s[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
