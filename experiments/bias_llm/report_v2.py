#!/usr/bin/env python3
"""Turn results/v2_scores.json into results/v2_report.md (tables only; prose lives in the write-up)."""

from __future__ import annotations

import json
from pathlib import Path

from ladder import LEVELS

HERE = Path(__file__).parent
ORDER = ["gold bias", "exhaustive", "base3shot single", "base3shot feedback k=4",
         "sft single", "sft resample k=4", "sft feedback k=4"]
TIER_NAMES = {"seen": "Seen programs (160 training tasks)", "test": "Unseen synthetic (160 test tasks)",
              "gold": "Gold set (16 hand-written problems)"}
SHORT = {"format": "fmt", "semantics": "sem", "expressive": "expr", "correct": "corr", "tight": "tight"}


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def cell(s: dict, lv: str) -> str:
    if s["seeds"] == 1 or s["min"][lv] == s["max"][lv]:
        return pct(s["mean"][lv])
    return f"{pct(s['mean'][lv])} ({pct(s['min'][lv])}–{pct(s['max'][lv])})"


def main() -> None:
    d = json.loads((HERE / "results" / "v2_scores.json").read_text())
    out = ["# Bias generator v2: ladder results", "",
           "% of tasks reaching each level (cumulative). Seeded runs: mean over 3 seeds (min–max). "
           "`k=4` = up to 4 attempts, stopping at the first bias XHAIL can learn from.", ""]
    for tier, name in TIER_NAMES.items():
        rows = d["write"].get(tier, {})
        for variant, label in [("raw", "as generated"), ("pruned", "after verify-then-prune")]:
            out += [f"## {name}: {label}", "", "| method | n | " + " | ".join(SHORT[l] for l in LEVELS) + " |",
                    "|---|---:|" + "---:|" * len(LEVELS)]
            for m in ORDER:
                if m in rows:
                    s = rows[m][variant]
                    out.append(f"| {m} | {s['n']} | " + " | ".join(cell(s, l) for l in LEVELS) + " |")
            out.append("")

    grid = d["gold_grid"]
    methods = [m for m in ORDER if m in grid]
    ids = sorted(grid["gold bias"])
    out += ["## Gold set, per problem (highest level reached, after pruning, seed 0)", "",
            "Levels: 0 = format fails, 1 = semantics fails, 2 = expressive fails, 3 = correct fails, 4 = tight fails, 5 = all pass.", "",
            "| problem | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for i in ids:
        out.append(f"| {i} | " + " | ".join(str(grid[m].get(i, "")) for m in methods) + " |")
    out.append("")

    out += ["## Repair: one greedy fix of a corrupted bias", "",
            "| tier | n | reach correct | correct, by the level the broken bias failed at |", "|---|---:|---:|---|"]
    for tier, r in d["repair"].items():
        by = ", ".join(f"{k} {pct(v)}% (n={r['n_by_broken_level'][k]})" for k, v in sorted(r["correct_by_broken_level"].items()))
        out.append(f"| {tier} | {r['n']} | {pct(r['reach']['correct'])} | {by} |")
    (HERE / "results" / "v2_report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
