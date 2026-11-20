#!/usr/bin/env python3
"""Score the v4 run (preds/v4): bias + XHAIL vs writing the rule directly.

Both jobs are scored on the same question, "correct": does the final rule agree with the
intended rule on the labelled atoms (or the whole domain for gold)? For the bias job that is
ladder level 4 (XHAIL learns the rule from the bias); for the rule job it is the model's own rule.
"single" = attempt 1, "k=4" = final answer after up to 4 resampled attempts. 3 seeds.

    python score_v4.py      # writes results/v4_scores.json and results/v4_report.md
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import load_tasks
from ladder import climb
from rules import score_rule

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TIERS = {"test": "data/test.jsonl", "new": "data/test_new.jsonl", "ec": "data/test_v4.jsonl", "gold": "gold",
         "seen": "data/train_v4.jsonl"}
TIER_NAMES = {"test": "unseen, old families (160)", "new": "unseen, v3 families (80)",
              "ec": "unseen, v4 event-calculus families (120)", "gold": "gold (16)", "seen": "seen, v4 EC (80)"}
WORKERS = 4  # keep the Mac cool


def _score(args) -> tuple[int, bool]:
    """(raw level, correct?) for one completion."""
    job, task, text = args
    if job == "rule":
        lv = score_rule(task, text)
        return lv, lv >= 3
    lv = climb(task, text)["level"]
    return lv, lv >= 4


def main() -> None:
    tasks = {k: {t["id"]: t for t in load_tasks(v if v == "gold" else str(HERE / v))} for k, v in TIERS.items()}
    runs = []
    for f in sorted((HERE / "preds" / "v4").glob("*.jsonl")):
        m = re.fullmatch(r"(bias|rule)_(test|new|ec|gold|seen)_resample_s(\d)", f.stem)
        if m:
            runs.append((*m.groups(), [json.loads(line) for line in open(f)]))
    jobs = {}
    for job, tier, _, rows in runs:
        for r in rows:
            for text in (r["attempts"][0]["completion"], r["completion"]):
                jobs[(job, tier, r["id"], text)] = (job, tasks[tier][r["id"]], text)
    print(f"scoring {len(jobs)} completions with {WORKERS} workers", flush=True)
    keys = list(jobs)
    with ProcessPoolExecutor(WORKERS) as ex:
        scored = dict(zip(keys, ex.map(_score, [jobs[k] for k in keys], chunksize=8)))

    per = defaultdict(list)  # (tier, job, single|k=4) -> [share correct per seed]
    fam = defaultdict(lambda: defaultdict(list))  # ec tier, seed 0, k=4: job -> family -> [correct]
    grid = defaultdict(dict)  # gold, seed 0: "job single|k=4" -> id -> level
    for job, tier, seed, rows in runs:
        for which, get in (("single", lambda r: r["attempts"][0]["completion"]), ("k=4", lambda r: r["completion"])):
            res = {r["id"]: scored[(job, tier, r["id"], get(r))] for r in rows}
            per[(tier, job, which)].append(sum(c for _, c in res.values()) / len(res))
            if seed == "0" and tier == "gold":
                grid[f"{job} {which}"] = {i: lv for i, (lv, _) in res.items()}
            if seed == "0" and tier == "ec" and which == "k=4":
                for i, (_, c) in res.items():
                    fam[job][tasks["ec"][i]["family"]].append(c)
    summary = {f"{tier}|{job}|{which}": {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "seeds": len(v)}
               for (tier, job, which), v in per.items()}
    families = {job: {f: sum(v) / len(v) for f, v in d.items()} for job, d in fam.items()}
    RESULTS.mkdir(exist_ok=True)
    out = {"correct": summary, "ec_by_family": families, "gold_grid": grid}
    (RESULTS / "v4_scores.json").write_text(json.dumps(out, indent=2))
    report(out)
    print("wrote results/v4_scores.json and results/v4_report.md")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def cell(s: dict | None) -> str:
    if not s:
        return "–"
    return pct(s["mean"]) if s["seeds"] == 1 or s["min"] == s["max"] else f"{pct(s['mean'])} ({pct(s['min'])}–{pct(s['max'])})"


def report(d: dict) -> None:
    c = d["correct"]
    out = ["# Bias generator v4: bias + XHAIL vs writing the rule", "",
           "% of tasks whose final rule is **correct** (agrees with the intended rule on the labelled atoms; "
           "gold: on the whole domain). Mean over 3 seeds (min–max). Same base model, same training programs.", "",
           "| method | " + " | ".join(TIER_NAMES[t] for t in TIERS) + " |", "|---|" + "---:|" * len(TIERS)]
    for job in ("bias", "rule"):
        for which in ("single", "k=4"):
            out.append(f"| {job} {which} | " + " | ".join(cell(c.get(f"{t}|{job}|{which}")) for t in TIERS) + " |")
    out.append("")
    if d["ec_by_family"]:
        fams = sorted({f for v in d["ec_by_family"].values() for f in v})
        out += ["## v4 event-calculus families: % correct (k=4, seed 0)", "", "| family | bias | rule |", "|---|---:|---:|"]
        for f in fams:
            out.append(f"| {f} | " + " | ".join(pct(d["ec_by_family"].get(j, {}).get(f, 0)) for j in ("bias", "rule")) + " |")
        out.append("")
    grid = d["gold_grid"]
    if grid:
        methods = [m for m in ("bias single", "bias k=4", "rule single", "rule k=4") if m in grid]
        ids = sorted({i for v in grid.values() for i in v})
        out += ["## Gold, per problem (seed 0)", "",
                "bias: ladder level 0–5 (correct = 4 or 5). rule: 0 = no usable rule, 1 = parses, "
                "2 = fits the training examples, 3 = correct.", "",
                "| problem | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
        for i in ids:
            out.append(f"| {i} | " + " | ".join(str(grid[m].get(i, "")) for m in methods) + " |")
    (RESULTS / "v4_report.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
