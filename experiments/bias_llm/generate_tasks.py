#!/usr/bin/env python3
"""Generate synthetic XHAIL tasks with gold mode declarations.

Each task has a hidden target rule. Labels come from solving BG + target with
clingo, so they are correct by construction. A task is kept only if XHAIL,
given the gold modes, learns a hypothesis that scores 100% on held-out
examples — so every gold bias is known to be sufficient.

Four task families (mirroring the hand-written benchmarks):
    unary   h(X) :- [not] p(X), [not] q(X).                (animals, penguins)
    part    h(X) :- has(X,Y), p(Y), [q(Y)].                 (trains)
    chain   h(X,Y) :- r1(X,Z), r2(Z,Y).                     (grandfather)
    ec      happens(e(#c),T) :- holdsAt(f(#d),T).           (event_calculus, epidemic)

Output: JSONL, one task per line:
    {"id", "family", "program" (no modes), "gold_modes", "target",
     "holdout": [{"atom", "positive"}], "gold_hypothesis"}

Usage:
    python experiments/bias_llm/generate_tasks.py --n 2000 --seed 0 \
        --out experiments/bias_llm/data
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import _solve, holdout_accuracy, run_xhail, splice

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ENTITY_TYPES = {
    "person": ["alice", "bob", "carol", "dan", "eve", "fay", "gus", "hal", "ivy", "jo",
               "kim", "lee", "max", "ned", "ola", "pam"],
    "animal": ["rex", "tom", "kit", "bo", "zed", "pip", "ace", "moe", "ben", "dot",
               "fox", "gem", "hop", "lux", "nip", "oz"],
    "city": ["york", "bath", "leeds", "derby", "hull", "ely", "wells", "truro", "ripon",
             "exeter", "ayr", "perth", "wick", "cork", "sligo", "tralee"],
    "item": ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10", "i11", "i12",
             "i13", "i14", "i15", "i16"],
    "patient": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11",
                "p12", "p13", "p14", "p15", "p16"],
}
PROPERTIES = ["red", "large", "old", "heavy", "fast", "rare", "cheap", "wild", "tall",
              "loud", "soft", "bright", "wet", "hot", "young", "striped", "spotted",
              "sick", "rich", "coastal", "urban", "fragile", "metal", "wooden", "sweet",
              "sour", "active", "smoker", "vaccinated", "insured", "tagged", "sealed"]
TARGETS = ["good", "valid", "safe", "happy", "target", "special", "eligible", "risky",
           "approved", "selected", "flagged", "premium", "healthy", "ready", "stable"]
PART_SCHEMES = [  # (whole type, part type, relation, wholes, part prefix)
    ("train", "car", "has_car", "t", "c"),
    ("molecule", "atom", "has_atom", "m", "a"),
    ("order", "line", "has_line", "o", "l"),
    ("team", "player", "has_player", "tm", "pl"),
    ("house", "room", "has_room", "h", "r"),
]
RELATIONS = ["parent", "friend", "knows", "manages", "follows", "likes", "trusts",
             "teaches", "sibling", "linked", "near", "calls", "owes", "helps"]
CHAIN_TARGETS = ["grandparent", "reachable", "related", "connected", "boss_of",
                 "influences", "second_degree", "indirect"]
EC_SCHEMES = [  # (type, event verb, effect fluent, trigger fluents, toggles)
    ("person", "infect", "ill", ["ill", "tired", "busy", "hungry"]),
    ("machine", "restart", "running", ["overheated", "idle", "faulty", "locked"]),
    ("light", "switch", "on", ["dark", "motion", "armed", "blocked"]),
    ("valve", "open", "flowing", ["pressured", "clear", "cold", "damaged"]),
    ("robot", "charge", "charged", ["low", "docked", "idle", "moving"]),
]
EC_ENTITIES = {
    "person": ["alice", "bob", "carol", "dan"], "machine": ["m1", "m2", "m3", "m4"],
    "light": ["l1", "l2", "l3", "l4"], "valve": ["v1", "v2", "v3", "v4"],
    "robot": ["r1", "r2", "r3", "r4"],
}
EC_AXIOMS = """holdsAt(F,T) :- time(T), time(S), S<T, happens(E,S), initiates(E,F,S), not clipped(S,F,T).
clipped(S,F,T) :- time(S), time(T), time(R), S<R, R<T, happens(E,R), terminates(E,F,R).
holdsAt(F,T) :- time(T), initially(F), not clipped(0,F,T)."""


# v3 families: names deliberately unlike the gold set (no fruit/colour, sprinklers, sugar, blocks).
NEG_RATE = 0.35
EC_VERBS = ["boot", "seal", "flush", "rinse", "lock", "unlock", "wake", "ping", "tune", "sweep",
            "fold", "stamp", "load", "drain", "patch", "reset", "polish", "mount", "brief", "audit"]
EC_FLUENTS = ["primed", "warm", "noisy", "stuck", "sealed", "dusty", "loaded", "ready", "alert",
              "drained", "open", "faulty", "quiet", "bright", "full", "hidden", "busy", "tense"]
ATTR_SCHEMES = [  # (entity type, attribute relation, value type, values)
    ("box", "size", "measure", ["small", "medium", "large"]),
    ("shirt", "fabric", "material", ["wool", "silk", "linen", "cotton"]),
    ("road", "zone", "region", ["north", "south", "east", "west"]),
    ("tile", "shape", "form", ["square", "round", "oval"]),
    ("worker", "grade", "band", ["junior", "senior", "lead"]),
]
PROP_ATOMS = ["alpha", "beta", "gamma", "delta", "sigma", "omega", "kappa", "zeta"]


def facts(pred: str, args_list) -> str:
    return " ".join(f"{pred}({','.join(a) if isinstance(a, tuple) else a})." for a in args_list)


# ---------------------------------------------------------------------------
# Families. Each returns (bg, target_rule, gold_modes, candidates, ex_pred)
#   candidates: ground atoms (str) that are examples — label = in model
# ---------------------------------------------------------------------------


def gen_unary(rng: random.Random):
    tname = rng.choice(list(ENTITY_TYPES))
    ents = rng.sample(ENTITY_TYPES[tname], rng.randint(8, 14))
    props = rng.sample(PROPERTIES, rng.randint(3, 6))
    head = rng.choice(TARGETS)
    use_neg = rng.random() < NEG_RATE
    n_body = rng.choice([1, 2])
    body_props = rng.sample(props, n_body)
    lits = []
    for i, p in enumerate(body_props):
        neg = use_neg and (i == n_body - 1)
        lits.append(f"{'not ' if neg else ''}{p}(X)")
    bg = [facts(tname, ents)]
    for p in props:
        ext = [e for e in ents if rng.random() < 0.5]
        if ext:
            bg.append(facts(p, ext))
    target = f"{head}(X) :- {tname}(X), {', '.join(lits)}."
    gold = [f"#modeh {head}(+{tname})."]
    gold += [f"#modeb {p}(+{tname})." for p in props]
    if use_neg:
        gold += [f"#modeb not {p}(+{tname})." for p in props]
    cands = [f"{head}({e})" for e in ents]
    return "\n".join(bg), target, gold, cands


def gen_part(rng: random.Random):
    whole, part, rel, wp, pp = rng.choice(PART_SCHEMES)
    n = rng.randint(5, 9)
    wholes = [f"{wp}{i}" for i in range(1, n + 1)]
    has, parts = [], []
    for w in wholes:
        for j in range(1, rng.randint(2, 3) + 1):
            p = f"{pp}{w[len(wp):]}{j}"
            has.append((w, p))
            parts.append(p)
    props = rng.sample(PROPERTIES, rng.randint(3, 5))
    head = rng.choice(TARGETS)
    body_props = rng.sample(props, rng.choice([1, 2]))
    bg = [facts(whole, wholes), facts(rel, has), f"{part}(Y) :- {rel}(_,Y)."]
    for p in props:
        ext = [x for x in parts if rng.random() < 0.4]
        if ext:
            bg.append(facts(p, ext))
    body = ", ".join(f"{p}(Y)" for p in body_props)
    target = f"{head}(X) :- {whole}(X), {rel}(X,Y), {body}."
    gold = [f"#modeh {head}(+{whole}).", f"#modeb {rel}(+{whole},-{part})."]
    gold += [f"#modeb {p}(+{part})." for p in props]
    cands = [f"{head}({w})" for w in wholes]
    return "\n".join(bg), target, gold, cands


def gen_chain(rng: random.Random):
    tname = rng.choice(["person", "city", "item"])
    ents = rng.sample(ENTITY_TYPES[tname], rng.randint(6, 8))
    rels = rng.sample(RELATIONS, rng.randint(2, 3))
    r1, r2 = rels[0], rels[rng.randint(0, 1)] if len(rels) == 2 else rels[1]
    head = rng.choice(CHAIN_TARGETS)
    bg = [facts(tname, ents)]
    for r in rels:
        edges = [(a, b) for a in ents for b in ents if a != b and rng.random() < 0.18]
        if edges:
            bg.append(facts(r, edges))
    target = f"{head}(X,Y) :- {r1}(X,Z), {r2}(Z,Y), X != Y."
    gold = [f"#modeh {head}(+{tname},+{tname})."]
    for r in rels:
        gold += [f"#modeb {r}(+{tname},-{tname}).", f"#modeb {r}(+{tname},+{tname})."]
    cands = [f"{head}({a},{b})" for a in ents for b in ents if a != b]
    return "\n".join(bg), target, gold, cands


def gen_ec(rng: random.Random):
    tname, verb, effect, triggers = rng.choice(EC_SCHEMES)
    ents = rng.sample(EC_ENTITIES[tname], rng.randint(2, 3))
    horizon = rng.randint(6, 9)
    fls = rng.sample(triggers, rng.randint(2, 3))
    if effect in fls:
        fls.remove(effect)
    cause, subj = rng.choice(ents), None
    subj = rng.choice([e for e in ents if e != cause])
    trig = fls[0]
    bg = [EC_AXIOMS, facts("time", [str(t) for t in range(horizon + 1)]), facts(tname, ents)]
    bg.append(f"initiates({verb}(X), {effect}(X), T) :- {tname}(X), time(T).")
    for f in fls:
        bg.append(f"initiates(start_{f}(X), {f}(X), T) :- {tname}(X), time(T).")
        bg.append(f"terminates(stop_{f}(X), {f}(X), T) :- {tname}(X), time(T).")
    # Scripted narrative: trigger fluents switch on/off at random times.
    evs = []
    for f in fls:
        for e in ents:
            if rng.random() < 0.6:
                s = rng.randint(0, horizon - 3)
                evs.append(f"happens(start_{f}({e}),{s}).")
                if rng.random() < 0.5:
                    evs.append(f"happens(stop_{f}({e}),{rng.randint(s + 1, horizon - 1)}).")
    # Guarantee the trigger fires, but not at time 0 (so the empty-body rule is wrong).
    s = rng.randint(1, horizon - 4)
    evs.append(f"happens(start_{trig}({cause}),{s}).")
    bg.append(" ".join(sorted(set(evs))))
    target = f"happens({verb}({subj}),T) :- time(T), holdsAt({trig}({cause}),T)."
    gold = [f"#modeh happens({verb}(#{tname}),+time)."]
    gold += [f"#modeb holdsAt({f}(#{tname}),+time)." for f in fls]
    cands = [f"holdsAt({effect}({subj}),{t})" for t in range(1, horizon + 1)]
    return "\n".join(bg), target, gold, cands


def gen_ec_named(rng: random.Random):
    """EC with naturally named events: the event to learn is the one whose effect is observed.

    Distractor events have their own effects and some of them happen in the narrative, so the
    head cannot be found by name. A third of tasks need a negated trigger (not holdsAt).
    """
    tname, _, _, _ = rng.choice(EC_SCHEMES)
    ents = rng.sample(EC_ENTITIES[tname], rng.randint(2, 3))
    horizon = rng.randint(6, 9)
    verbs = rng.sample(EC_VERBS, rng.randint(3, 4))
    fluents = rng.sample(EC_FLUENTS, len(verbs) + 2)
    effects = dict(zip(verbs, fluents))  # each event initiates its own fluent
    verb, effect = verbs[0], effects[verbs[0]]
    trig, blocker = fluents[-2], fluents[-1]
    cause = rng.choice(ents)
    subj = rng.choice([e for e in ents if e != cause])
    bg = [EC_AXIOMS, facts("time", [str(t) for t in range(horizon + 1)]), facts(tname, ents)]
    for v, f in effects.items():
        bg.append(f"initiates({v}(X), {f}(X), T) :- {tname}(X), time(T).")
    for f in (trig, blocker):
        bg.append(f"initiates(raise_{f}(X), {f}(X), T) :- {tname}(X), time(T).")
        bg.append(f"terminates(drop_{f}(X), {f}(X), T) :- {tname}(X), time(T).")
    evs = []
    for v in verbs[1:]:  # distractor events really happen, so their fluents hold too
        for e in ents:
            if rng.random() < 0.5:
                evs.append(f"happens({v}({e}),{rng.randint(0, horizon - 2)}).")
    s0 = rng.randint(1, horizon - 4)
    evs.append(f"happens(raise_{trig}({cause}),{s0}).")
    use_neg = rng.random() < 1 / 3
    if use_neg:
        b0 = rng.randint(s0, horizon - 2)
        evs.append(f"happens(raise_{blocker}({cause}),{b0}).")
    bg.append(" ".join(sorted(set(evs))))
    body = f"holdsAt({trig}({cause}),T)" + (f", not holdsAt({blocker}({cause}),T)" if use_neg else "")
    target = f"happens({verb}({subj}),T) :- time(T), {body}."
    gold = [f"#modeh happens({verb}(#{tname}),+time)."]
    gold += [f"#modeb holdsAt({f}(#{tname}),+time)." for f in (trig, blocker)]
    if use_neg:
        gold += [f"#modeb not holdsAt({f}(#{tname}),+time)." for f in (trig, blocker)]
    cands = [f"holdsAt({effect}({subj}),{t})" for t in range(1, horizon + 1)]
    return "\n".join(bg), target, gold, cands


def gen_const_attr(rng: random.Random):
    """A rule that tests an attribute against one constant value: needs #type in the body."""
    tname, attr, vtype, values = rng.choice(ATTR_SCHEMES)
    ents = [f"{tname[0]}{i}" for i in range(1, rng.randint(8, 13))]
    props = rng.sample(PROPERTIES, rng.randint(1, 3))
    head = rng.choice(TARGETS)
    val = rng.choice(values)
    bg = [facts(tname, ents), facts(vtype, values), facts(attr, [(e, rng.choice(values)) for e in ents])]
    for p in props:
        ext = [e for e in ents if rng.random() < 0.5]
        if ext:
            bg.append(facts(p, ext))
    target = f"{head}(X) :- {tname}(X), {attr}(X,{val})."
    gold = [f"#modeh {head}(+{tname}).", f"#modeb {attr}(+{tname},#{vtype})."]
    gold += [f"#modeb {p}(+{tname})." for p in props]
    cands = [f"{head}({e})" for e in ents]
    return "\n".join(bg), target, gold, cands


def gen_prop(rng: random.Random):
    """Zero-arity atoms. One positive example, so the target is the bare fact (as XHAIL learns it)."""
    atoms = rng.sample(PROP_ATOMS, rng.randint(3, 5))
    head, inputs = atoms[0], atoms[1:]
    present = [a for a in inputs if rng.random() < 0.7] or inputs[:1]
    bg = " ".join(f"{a}." for a in present)
    gold = [f"#modeh {head}."] + [f"#modeb {a}." for a in present]
    return bg, f"{head}.", gold, [head]


FAMILIES = {"unary": gen_unary, "part": gen_part, "chain": gen_chain, "ec": gen_ec,
            "ec_named": gen_ec_named, "const_attr": gen_const_attr, "prop": gen_prop}


# ---------------------------------------------------------------------------
# Assemble + verify
# ---------------------------------------------------------------------------


def make_task(family: str, rng: random.Random, holdout_frac: float = 0.3):
    bg, target, gold, cands = FAMILIES[family](rng)
    model = _solve(bg + "\n" + target)
    if model is None:
        return None, "unsat"
    labelled = [{"atom": c, "positive": c in model} for c in cands]
    pos = [x for x in labelled if x["positive"]]
    neg = [x for x in labelled if not x["positive"]]
    if family == "prop":
        pos, neg = labelled, []
    elif len(pos) < 3 or len(neg) < 3:
        return None, "imbalanced"
    if family == "chain" and len(neg) > 4 * len(pos):  # keep chain tasks small
        neg = rng.sample(neg, 4 * len(pos))
    rng.shuffle(pos)
    rng.shuffle(neg)
    hp, hn = max(1, int(len(pos) * holdout_frac)), max(1, int(len(neg) * holdout_frac))
    if family == "prop":
        hp = hn = 0
    holdout = pos[:hp] + neg[:hn]
    shown = pos[hp:] + neg[hn:]
    rng.shuffle(shown)
    ex_lines = [f"#example {'' if x['positive'] else 'not '}{x['atom']}." for x in shown]
    program = bg + "\n\n" + "\n".join(ex_lines) + "\n"

    res = run_xhail(splice(program, gold), timeout_s=8)
    if not (res.get("ok") and res.get("success")):
        return None, "gold_fails"
    acc = holdout_accuracy(bg, res["hypothesis"], holdout) if holdout else 1.0
    if acc != 1.0:
        return None, "gold_underdetermined"
    return {
        "family": family, "program": program, "gold_modes": gold, "target": target,
        "holdout": holdout, "gold_hypothesis": res["hypothesis"],
        "gold_runtime_s": res["runtime_s"],
    }, "ok"


def _attempt(args: tuple[str, int, int]):
    family, seed, j = args
    rng = random.Random(f"{seed}-{family}-{j}")
    return family, j, make_task(family, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="total tasks to keep (split evenly by family)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--splits", default="0.84,0.08,0.08", help="train,val,test fractions")
    ap.add_argument("--families", default="unary,part,chain,ec", help="comma-separated subset of FAMILIES")
    ap.add_argument("--neg-rate", type=float, default=0.35, help="share of unary tasks that need negation")
    args = ap.parse_args()
    global NEG_RATE, FAMILIES
    NEG_RATE = args.neg_rate
    FAMILIES = {f: FAMILIES[f] for f in args.families.split(",")}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quota = {f: args.n // len(FAMILIES) for f in FAMILIES}
    kept: dict[str, list[tuple[int, dict]]] = {f: [] for f in FAMILIES}
    next_j = {f: 0 for f in FAMILIES}
    reasons: Counter = Counter()
    seen: set[str] = set()
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        while any(len(kept[f]) < quota[f] for f in FAMILIES):
            batch = []
            for f in FAMILIES:
                need = quota[f] - len(kept[f])
                for _ in range(min(need * 2, 64) if need > 0 else 0):
                    batch.append((f, args.seed, next_j[f]))
                    next_j[f] += 1
            for fam, j, (task, why) in pool.map(_attempt, batch):
                reasons[why] += 1
                if task is None or task["program"] in seen or len(kept[fam]) >= quota[fam]:
                    continue
                seen.add(task["program"])
                kept[fam].append((j, task))
            print(f"  kept {sum(map(len, kept.values()))}/{sum(quota.values())} "
                  f"({time.time() - t0:.0f}s)  {dict(reasons)}", flush=True)

    tasks = []
    for f in FAMILIES:
        for i, (_, t) in enumerate(sorted(kept[f], key=lambda x: x[0])):
            t["id"] = f"{f}-{i:05d}"
            tasks.append(t)
    random.Random(args.seed).shuffle(tasks)
    fr = [float(x) for x in args.splits.split(",")]
    n_tr, n_va = int(fr[0] * len(tasks)), int(fr[1] * len(tasks))
    splits = {"train": tasks[:n_tr], "val": tasks[n_tr:n_tr + n_va], "test": tasks[n_tr + n_va:]}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} tasks  {dict(Counter(r['family'] for r in rows))}")
    print("generation attempts:", dict(reasons))


if __name__ == "__main__":
    main()
