#!/usr/bin/env python3
"""Break gold biases in the ways a model typically gets them wrong, and label each break with the ladder.

Each corruption is aimed at one ladder level, but the label is whatever level the broken bias
actually fails at (some breaks turn out harmless). Every row is also a repair example:
program + broken bias + XHAIL feedback (test-time information only) -> gold bias.

    python corrupt.py --tasks data/train.jsonl --out data/repair_train.jsonl
    python corrupt.py --tasks gold --out data/repair_gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re

from baselines import exhaustive_bias
from feedback import MODE_PARTS_RE, TYPE_RE, diagnose
from ladder import climb, ground_atoms, parse_rule
from common import load_tasks


def _types(modes):
    return sorted({t for m in modes for t in TYPE_RE.findall(m.split(None, 1)[-1])})


def _retype(mode, old, new):
    """Rename a type inside a mode's arguments, leaving the #modeh/#modeb keyword alone."""
    kw, rest = mode.split(None, 1)
    return kw + " " + re.sub(rf"([+\-#]){old}\b", rf"\g<1>{new}", rest)


def _body(modes):
    return [i for i, m in enumerate(modes) if m.startswith("#modeb")]


# Each returns a new mode list, or None if it does not apply to this bias.

def drop_period(modes, rng, task):
    i = rng.randrange(len(modes))
    return modes[:i] + [modes[i].rstrip(".")] + modes[i + 1:]


def no_hash(modes, rng, task):
    i = rng.randrange(len(modes))
    return modes[:i] + [modes[i].lstrip("#")] + modes[i + 1:]


def prose(modes, rng, task):
    return ["Here are the mode declarations:"] + modes


def no_head(modes, rng, task):
    return [m for m in modes if not m.startswith("#modeh")]


def unknown_type(modes, rng, task):
    ts = _types(modes)
    if not ts:
        return None
    t = rng.choice(ts)
    new = rng.choice([t + "s", "thing", "object", t[:-1] if len(t) > 3 else t + "_t"])
    return [_retype(m, t, new) for m in modes]


def typo_pred(modes, rng, task):
    body = _body(modes)
    if not body:
        return None
    i = rng.choice(body)
    name = MODE_PARTS_RE.fullmatch(modes[i]).group(3)
    if len(name) < 4:
        return None
    k = rng.randrange(1, len(name) - 1)
    return modes[:i] + [modes[i].replace(name + "(", name[:k] + name[k + 1:] + "(", 1)] + modes[i + 1:]


def drop_arg(modes, rng, task):
    cands = [i for i in _body(modes) if "," in modes[i] and "(" not in modes[i].split("(", 1)[1]]
    if not cands:
        return None
    i = rng.choice(cands)
    return modes[:i] + [re.sub(r",[^,()]+\)\.$", ").", modes[i])] + modes[i + 1:]


def body_as_head(modes, rng, task):
    body = [m for m in modes if m.startswith("#modeb") and " not " not in m]
    if not body:
        return None
    h = rng.choice(body).replace("#modeb", "#modeh").replace("-", "+")
    return [h] + [m for m in modes if not m.startswith("#modeh")]


def drop_needed(modes, rng, task):
    rules = task["target"] if isinstance(task["target"], list) else [task["target"]]
    used = set()
    for r in rules:
        for lit in parse_rule(r)[1]:
            m = re.match(r"(?:not\s+)?(\w+)", lit)
            used.add(m.group(1))
    cands = [i for i in _body(modes) if MODE_PARTS_RE.fullmatch(modes[i]).group(3) in used]
    if not cands:
        return None
    i = rng.choice(cands)
    return modes[:i] + modes[i + 1:]


def drop_negation(modes, rng, task):
    if not any(" not " in m for m in modes):
        return None
    return [m for m in modes if " not " not in m]


def output_to_input(modes, rng, task):
    cands = [i for i, m in enumerate(modes) if "-" in m]
    if not cands:
        return None
    i = rng.choice(cands)
    return modes[:i] + [modes[i].replace("-", "+")] + modes[i + 1:]


def const_var_swap(modes, rng, task):
    cands = [i for i, m in enumerate(modes) if "#" in m[6:]]
    if cands:
        return [m[:6] + m[6:].replace("#", "+") for m in modes]
    cands = [i for i, m in enumerate(modes) if "+" in m]
    if not cands:
        return None
    i = rng.choice(cands)
    return modes[:i] + [modes[i].replace("+", "#", 1)] + modes[i + 1:]


def wrong_real_type(modes, rng, task):
    bg = ground_atoms(task["program"])
    real = sorted({a.split("(")[0] for a in bg if re.fullmatch(r"\w+\([^,()]+\)", a)})
    ts = _types(modes)
    others = [r for r in real if r not in ts]
    if not ts or not others:
        return None
    t, new = rng.choice(ts), rng.choice(others)
    i = rng.choice([i for i, m in enumerate(modes) if t in _types([m])])
    return modes[:i] + [_retype(modes[i], t, new)] + modes[i + 1:]


def shuffle_body(modes, rng, task):
    head = [m for m in modes if m.startswith("#modeh")]
    body = [m for m in modes if not m.startswith("#modeh")]
    if len(body) < 3:
        return None
    rng.shuffle(body)
    return head + body


def add_distractors(modes, rng, task):
    extra = [m for m in exhaustive_bias(task["program"]) if m.startswith("#modeb") and m not in modes]
    if not extra:
        return None
    return modes + rng.sample(extra, min(len(extra), rng.randint(3, 8)))


def add_one(modes, rng, task):
    extra = [m for m in exhaustive_bias(task["program"]) if m.startswith("#modeb") and m not in modes]
    return modes + [rng.choice(extra)] if extra else None


CORRUPTIONS = {
    "format": [drop_period, no_hash, prose, no_head],
    "semantics": [unknown_type, typo_pred, drop_arg, body_as_head],
    "expressive": [drop_needed, drop_negation, output_to_input, const_var_swap, wrong_real_type],
    "correct": [shuffle_body, add_distractors],
    "tight": [add_one],
}


def corrupt_task(task: dict, rng: random.Random, per_level: int) -> list[dict]:
    rows, seen = [], {"\n".join(task["gold_modes"])}
    for aimed, fns in CORRUPTIONS.items():
        for fn in rng.sample(fns, min(per_level, len(fns))):
            broken = fn(list(task["gold_modes"]), rng, task)
            text = "\n".join(broken) if broken else ""
            if not broken or text in seen:
                continue
            seen.add(text)
            lad = climb(task, text)
            fb = diagnose(task["program"], lad.get("modes", []))
            rows.append({
                "id": task["id"], "family": task["family"], "corruption": fn.__name__, "aimed_at": aimed,
                "failed_at": lad["failed"] or "none", "reason": lad["reason"],
                "broken": text, "feedback": fb["feedback"] or f"XHAIL learned: {' '.join(fb.get('hypothesis', []))}",
                "gold": "\n".join(task["gold_modes"]), "program": task["program"],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-level", type=int, default=2, help="corruptions tried per level per task")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = load_tasks(args.tasks)[: args.limit]
    n = 0
    with open(args.out, "w") as fh:
        for i, t in enumerate(tasks, 1):
            for row in corrupt_task(t, rng, args.per_level):
                fh.write(json.dumps(row) + "\n")
                n += 1
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} tasks, {n} rows", flush=True)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
