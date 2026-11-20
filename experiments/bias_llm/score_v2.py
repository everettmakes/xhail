#!/usr/bin/env python3
"""Score the v2 predictions (preds/v2) on the ladder and write the results tables.

Write runs are scored twice: attempt 1 alone ("single") and the run's final answer after up to
4 attempts. Every score is reported with and without verify-then-prune. Seeded runs are
summarised as the mean over seeds with the min-max range. Ladder levels are ordinal, so
results are shares of tasks reaching each level, never an average level.

    python score_v2.py            # writes results/v2_scores.json and results/v2_report.md
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from baselines import exhaustive_bias
from common import extract_modes, load_tasks
from ladder import LEVELS, climb
from prune import prune

HERE = Path(__file__).parent
PREDS = HERE / "preds" / "v2"
RESULTS = HERE / "results"
TIERS = {"seen": "data/train.jsonl", "test": "data/test.jsonl", "gold": "gold"}

_cache: dict = {}


def level(task: dict, text: str, pruned: bool) -> int:
    key = (task["id"], text, pruned)
    if key not in _cache:
        if pruned and climb(task, text)["level"] >= 2:
            text = "\n".join(prune(task["program"], extract_modes(text)))
        _cache[key] = climb(task, text)["level"]
    return _cache[key]


def reach(levels: list[int]) -> dict[str, float]:
    return {LEVELS[i]: sum(lv > i for lv in levels) / len(levels) for i in range(5)}


def main() -> None:
    tasks = {tier: {t["id"]: t for t in load_tasks(str(HERE / src) if src != "gold" else "gold")} for tier, src in TIERS.items()}
    runs = defaultdict(list)  # (tier, method, pruned) -> list of {task_id: level}, one per seed
    per_task = defaultdict(dict)  # gold grid: method -> {task_id: level (seed 0, pruned)}

    for f in sorted(PREDS.glob("*.jsonl")):
        m = re.fullmatch(r"(sft|base3shot)_(seen|test|gold)_(resample|feedback)_s(\d)", f.stem)
        if not m:
            continue
        model, tier, strategy, seed = m.groups()
        rows = [json.loads(line) for line in open(f)]
        for pruned in (False, True):
            single = {r["id"]: level(tasks[tier][r["id"]], r["attempts"][0]["completion"], pruned) for r in rows}
            final = {r["id"]: level(tasks[tier][r["id"]], r["completion"], pruned) for r in rows}
            if strategy == "resample":  # attempt 1 is shared by both strategies; count it once
                runs[(tier, f"{model} single", pruned)].append(single)
            runs[(tier, f"{model} {strategy} k=4", pruned)].append(final)
            if strategy == "feedback" and model == "base3shot":
                runs[(tier, f"{model} single", pruned)].append(single)
            if tier == "gold" and seed == "0" and pruned:
                per_task.setdefault(f"{model} single", single)
                per_task[f"{model} {strategy} k=4"] = final
        print(f"scored {f.name}", flush=True)

    for tier in TIERS:  # reference rows: gold and exhaustive biases
        for pruned in (False, True):
            ts = tasks[tier].values()
            ts = list(ts)[:160] if tier == "seen" else list(ts)
            runs[(tier, "gold bias", pruned)].append({t["id"]: level(t, "\n".join(t["gold_modes"]), pruned) for t in ts})
            runs[(tier, "exhaustive", pruned)].append({t["id"]: level(t, "\n".join(exhaustive_bias(t["program"])), pruned) for t in ts})
    for t in tasks["gold"].values():
        per_task["gold bias"][t["id"]] = level(t, "\n".join(t["gold_modes"]), True)
        per_task["exhaustive"][t["id"]] = level(t, "\n".join(exhaustive_bias(t["program"])), True)

    # repair runs: one greedy fix of a corrupted bias
    repair = {}
    for tier in TIERS:
        rows = [json.loads(line) for line in open(PREDS / f"sft_repair_{tier}.jsonl")]
        by_level = defaultdict(list)
        for r in rows:
            by_level[r["failed_at"]].append(level(tasks[tier][r["id"]], r["completion"], False))
        repair[tier] = {"n": len(rows), "reach": reach([lv for v in by_level.values() for lv in v]),
                        "correct_by_broken_level": {k: sum(lv >= 4 for lv in v) / len(v) for k, v in by_level.items()},
                        "n_by_broken_level": {k: len(v) for k, v in by_level.items()}}

    summary = {}
    for (tier, method, pruned), seeds in runs.items():
        per_seed = [reach(list(s.values())) for s in seeds]
        summary.setdefault(tier, {}).setdefault(method, {})["pruned" if pruned else "raw"] = {
            "n": len(seeds[0]), "seeds": len(seeds),
            "mean": {k: sum(p[k] for p in per_seed) / len(per_seed) for k in per_seed[0]},
            "min": {k: min(p[k] for p in per_seed) for k in per_seed[0]},
            "max": {k: max(p[k] for p in per_seed) for k in per_seed[0]},
        }
    RESULTS.mkdir(exist_ok=True)
    out = {"write": summary, "repair": repair, "gold_grid": per_task}
    (RESULTS / "v2_scores.json").write_text(json.dumps(out, indent=2))
    print("wrote results/v2_scores.json")


if __name__ == "__main__":
    main()
