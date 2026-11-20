"""Non-learned bias generators to compare the fine-tuned model against.

exhaustive_bias(program)
    A typed "declare everything" heuristic — the bias a user writes with no
    domain insight. Every background predicate becomes a #modeb (with negated
    and input/output variants); the head comes from the examples. Types are
    inferred as the largest unary predicate covering each argument position.
    It is usually sufficient but large, so it measures what a *tight* bias
    buys in search cost.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import product

import clingo

from common import background_only, canon_mode

EC_INTERNAL = {"holdsAt", "clipped", "initiates", "terminates", "happens", "initially", "time"}


def _model(program: str) -> list[clingo.Symbol]:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as h:
        for m in h:
            return list(m.symbols(atoms=True))
    return []


def _examples(program: str) -> list[clingo.Symbol]:
    out = []
    for m in re.finditer(r"#example\s+(not\s+)?([^.]*)\.", program):
        out.append(clingo.parse_term(m.group(2).strip()))
    return out


def exhaustive_bias(program: str) -> list[str]:
    atoms = _model(background_only(program))
    exs = _examples(program)
    if not exs:
        return []

    unary: dict[str, set[str]] = defaultdict(set)
    ext: dict[tuple[str, int], list[tuple[str, ...]]] = defaultdict(list)
    for a in atoms:
        args = tuple(str(x) for x in a.arguments)
        ext[(a.name, len(args))].append(args)
        if len(args) == 1:
            unary[a.name].add(args[0])

    def type_of(consts: set[str]) -> str | None:
        cands = [p for p, e in unary.items() if consts <= e and p not in EC_INTERNAL]
        return max(cands, key=lambda p: (len(unary[p]), p)) if cands else None

    head = exs[0]
    modes: list[str] = []

    # --- Event Calculus tasks: examples are holdsAt(fluent, time) ---------
    if head.name == "holdsAt" and len(head.arguments) == 2:
        target_fluents = {e.arguments[0].name for e in exs if e.name == "holdsAt"}
        events, fluents = set(), set()
        happened = {str(a.arguments[0].name) for a in atoms if a.name == "happens"}
        for a in atoms:
            if a.name in ("initiates", "terminates"):
                ev, fl = a.arguments[0], a.arguments[1]
                if len(ev.arguments) != 1 or len(fl.arguments) != 1:
                    continue
                t_fl = type_of({str(fl.arguments[0])})
                if t_fl:
                    fluents.add((fl.name, t_fl))
                t_ev = type_of({str(ev.arguments[0])})
                if fl.name in target_fluents and ev.name not in happened and t_ev:
                    events.add((ev.name, t_ev))
        for ev, t in sorted(events):
            modes.append(f"#modeh happens({ev}(#{t}),+time).")
        for fl, t in sorted(fluents):
            modes.append(f"#modeb holdsAt({fl}(#{t}),+time).")
            modes.append(f"#modeb not holdsAt({fl}(#{t}),+time).")
        return [canon_mode(m) for m in modes]

    # --- Relational tasks -------------------------------------------------
    arity = len(head.arguments)
    head_types = []
    for i in range(arity):
        t = type_of({str(e.arguments[i]) for e in exs if e.name == head.name})
        if t is None:
            return []
        head_types.append(t)
    type_preds = set(head_types)

    body_specs = []
    for (name, n), rows in sorted(ext.items()):
        if name in EC_INTERNAL or name == head.name:
            continue
        if n == 0:
            body_specs.append((name, []))
            continue
        ts = [type_of({r[i] for r in rows}) for i in range(n)]
        if any(t is None for t in ts):
            continue
        if n == 1 and ts[0] == name:  # a type predicate itself
            type_preds.add(name)
            continue
        body_specs.append((name, ts))

    sig = ",".join(f"+{t}" for t in head_types)
    modes.append(f"#modeh {head.name}({sig})." if arity else f"#modeh {head.name}.")
    for name, ts in body_specs:
        if name in type_preds:
            continue
        if not ts:
            modes.append(f"#modeb {name}.")
            continue
        if len(ts) == 1:
            modes.append(f"#modeb {name}(+{ts[0]}).")
            modes.append(f"#modeb not {name}(+{ts[0]}).")
            continue
        # "-" before "+": XHAIL is sensitive to mode order (see README)
        for marks in product("-+", repeat=len(ts) - 1):
            # an input (+) argument must be bindable from the head's types
            if any(m == "+" and t not in head_types for m, t in zip(marks, ts[1:])):
                continue
            args = [f"+{ts[0]}"] + [f"{m}{t}" for m, t in zip(marks, ts[1:])]
            modes.append(f"#modeb {name}({','.join(args)}).")
    return [canon_mode(m) for m in modes]
