#!/usr/bin/env python3
"""Score the v3 run (preds/v3, plus the v2 baselines in preds/v2) on the ladder and write the report.

The v3 run is a 2x2: model (v2 = old templates, v3 = old + new templates) x shots (0, or 2
retrieved examples of similar shape), on three tiers:
    test   160 unseen programs from the old families
    new     80 unseen programs from the new families (EC heads with constants, constants in
           the body, propositional)
    gold    16 hand-written problems
Each write run has up to 4 attempts (resample), 3 seeds. "single" = attempt 1, "k=4" = final
answer. Levels are ordinal, so we report the share of tasks reaching each level.

    python score_v3.py      # writes results/v3_scores.json and results/v3_report.md
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import load_tasks
from ladder import LEVELS, climb

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TIERS = {"test": "data/test.jsonl", "new": "data/test_new.jsonl", "gold": "gold",
         "seen_old": "data/train.jsonl", "seen_new": "data/train_new.jsonl"}
TIER_NAMES = {"test": "Unseen synthetic, old families (160)", "new": "Unseen synthetic, new families (80)",
              "gold": "Gold set (16 hand-written problems)", "seen_old": "Seen, old families (80 training tasks)",
              "seen_new": "Seen, new families (80 training tasks)"}
REPAIR = {"gold": ("data/repair_gold.jsonl", "gold"), "new": ("data/repair_test_new.jsonl", "new")}
SHORT = {"format": "fmt", "semantics": "sem", "expressive": "expr", "correct": "corr", "tight": "tight"}
WORKERS = 4  # keep the Mac cool


def _level(args):
    task, text = args
    return climb(task, text)["level"]


def reach(levels: list[int]) -> dict[str, float]:
    return {LEVELS[i]: sum(lv > i for lv in levels) / len(levels) for i in range(5)}


def write_runs(preds: Path) -> list[tuple[str, str, str, int, list[dict]]]:
    """(tier, model, shots, seed, rows) for every write-mode file we know how to read."""
    out = []
    for f in sorted((preds / "v3").glob("*.jsonl")):
        m = re.fullmatch(r"(v2|v3)_shots(\d)_(test|new|gold|seen_old|seen_new)_resample_s(\d)", f.stem)
        if m:
            model, shots, tier, seed = m.groups()
            out.append((tier, model, shots, int(seed), [json.loads(line) for line in open(f)]))
    for f in sorted((preds / "v2").glob("sft_*_resample_s*.jsonl")):  # v2, no shots, from the v2 run
        m = re.fullmatch(r"sft_(test|gold)_resample_s(\d)", f.stem)
        if m:
            tier, seed = m.groups()
            out.append((tier, "v2", "0", int(seed), [json.loads(line) for line in open(f)]))
    return out


def main() -> None:
    preds = HERE / "preds"
    tasks = {tier: {t["id"]: t for t in load_tasks(src if src == "gold" else str(HERE / src))} for tier, src in TIERS.items()}
    runs = write_runs(preds)
    repairs = {tier: [json.loads(line) for line in open(preds / "v3" / f"v3_repair_{tier}.jsonl")]
               for tier in REPAIR if (preds / "v3" / f"v3_repair_{tier}.jsonl").exists()}

    # every (task, completion) pair to score, deduplicated
    jobs: dict[tuple[str, str, str], tuple[dict, str]] = {}
    for tier, _, _, _, rows in runs:
        for r in rows:
            for text in (r["attempts"][0]["completion"], r["completion"]):
                jobs[(tier, r["id"], text)] = (tasks[tier][r["id"]], text)
    for tier, rows in repairs.items():
        for r in rows:
            jobs[(REPAIR[tier][1], r["id"], r["completion"])] = (tasks[REPAIR[tier][1]][r["id"]], r["completion"])
    for tier in ("test", "new", "gold"):
        for t in tasks[tier].values():
            jobs[(tier, t["id"], "\n".join(t["gold_modes"]))] = (t, "\n".join(t["gold_modes"]))
    print(f"scoring {len(jobs)} biases with {WORKERS} workers", flush=True)
    keys = list(jobs)
    with ProcessPoolExecutor(WORKERS) as ex:
        levels = dict(zip(keys, ex.map(_level, [jobs[k] for k in keys], chunksize=8)))

    seeds = defaultdict(list)  # (tier, method) -> [ {id: level} per seed ]
    grid = defaultdict(dict)  # gold, seed 0: method -> {id: level}
    fam = defaultdict(lambda: defaultdict(list))  # new tier, seed 0, k=4: method -> family -> levels
    for tier, model, shots, seed, rows in sorted(runs, key=lambda x: x[3]):
        name = f"{model} shots{shots}"
        single = {r["id"]: levels[(tier, r["id"], r["attempts"][0]["completion"])] for r in rows}
        final = {r["id"]: levels[(tier, r["id"], r["completion"])] for r in rows}
        seeds[(tier, f"{name} single")].append(single)
        seeds[(tier, f"{name} k=4")].append(final)
        if seed == 0 and tier == "gold":
            grid[f"{name} single"] = single
            grid[f"{name} k=4"] = final
        if seed == 0 and tier == "new":
            for i, lv in final.items():
                fam[f"{name} k=4"][tasks["new"][i]["family"]].append(lv)
    for tier in ("test", "new", "gold"):
        g = {i: levels[(tier, i, "\n".join(t["gold_modes"]))] for i, t in tasks[tier].items()}
        seeds[(tier, "gold bias")].append(g)
        if tier == "gold":
            grid["gold bias"] = g

    summary: dict = {}
    for (tier, method), per in seeds.items():
        rs = [reach(list(s.values())) for s in per]
        summary.setdefault(tier, {})[method] = {
            "n": len(per[0]), "seeds": len(per),
            "mean": {k: sum(r[k] for r in rs) / len(rs) for k in LEVELS},
            "min": {k: min(r[k] for r in rs) for k in LEVELS},
            "max": {k: max(r[k] for r in rs) for k in LEVELS}}
    rep = {}
    for tier, rows in repairs.items():
        by = defaultdict(list)
        for r in rows:
            by[r["failed_at"]].append(levels[(REPAIR[tier][1], r["id"], r["completion"])])
        allv = [lv for v in by.values() for lv in v]
        rep[tier] = {"n": len(allv), "correct": sum(lv >= 4 for lv in allv) / len(allv),
                     "by_level": {k: [sum(lv >= 4 for lv in v) / len(v), len(v)] for k, v in by.items()}}
    families = {m: {f: reach(v)["correct"] for f, v in d.items()} for m, d in fam.items()}

    RESULTS.mkdir(exist_ok=True)
    out = {"write": summary, "repair": rep, "gold_grid": grid, "new_by_family": families}
    (RESULTS / "v3_scores.json").write_text(json.dumps(out, indent=2))
    report(out)
    print("wrote results/v3_scores.json and results/v3_report.md")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def cell(s: dict, lv: str) -> str:
    if s["seeds"] == 1 or s["min"][lv] == s["max"][lv]:
        return pct(s["mean"][lv])
    return f"{pct(s['mean'][lv])} ({pct(s['min'][lv])}–{pct(s['max'][lv])})"


def report(d: dict) -> None:
    order = ["gold bias"] + [f"{m} shots{s} {k}" for m in ("v2", "v3") for s in ("0", "2") for k in ("single", "k=4")]
    out = ["# Bias generator v3: ladder results", "",
           "% of tasks reaching each level (cumulative). Seeded runs: mean over 3 seeds (min–max). "
           "`k=4` = up to 4 resampled attempts, stopping at the first bias XHAIL can learn from. "
           "v2 = trained on the old families; v3 = old + new families. shots2 = two retrieved examples of similar shape.", ""]
    for tier, name in TIER_NAMES.items():
        rows = d["write"].get(tier)
        if not rows:
            continue
        out += [f"## {name}", "", "| method | n | " + " | ".join(SHORT[l] for l in LEVELS) + " |",
                "|---|---:|" + "---:|" * len(LEVELS)]
        for m in order:
            if m in rows:
                s = rows[m]
                out.append(f"| {m} | {s['n']} | " + " | ".join(cell(s, l) for l in LEVELS) + " |")
        out.append("")
    if d["new_by_family"]:
        methods = [m for m in order if m in d["new_by_family"]]
        fams = sorted({f for v in d["new_by_family"].values() for f in v})
        out += ["## New families: % correct by family (k=4, seed 0)", "",
                "| family | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
        for f in fams:
            out.append(f"| {f} | " + " | ".join(pct(d["new_by_family"][m].get(f, 0)) for m in methods) + " |")
        out.append("")
    grid = d["gold_grid"]
    methods = [m for m in order if m in grid]
    out += ["## Gold set, per problem (highest level reached, seed 0)", "",
            "0 = format fails, 1 = semantics, 2 = expressive, 3 = correct, 4 = tight fails, 5 = all pass. "
            "Correct means 4 or 5.", "", "| problem | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for i in sorted(grid["gold bias"]):
        out.append(f"| {i} | " + " | ".join(str(grid[m].get(i, "")) for m in methods) + " |")
    out.append("")
    if d["repair"]:
        out += ["## Repair (v3): one greedy fix of a corrupted bias", "",
                "| tier | n | % correct | % correct by the level the broken bias failed at |", "|---|---:|---:|---|"]
        for tier, r in d["repair"].items():
            by = ", ".join(f"{k} {pct(v[0])} (n={v[1]})" for k, v in sorted(r["by_level"].items()))
            out.append(f"| {tier} | {r['n']} | {pct(r['correct'])} | {by} |")
    (RESULTS / "v3_report.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
