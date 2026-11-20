"""Score a candidate bias on a ladder of checks. The highest level passed is the score.

    1 format      every line is a well-formed mode declaration and there is a #modeh
    2 semantics   predicates, arities and types exist in the program; the head is the example predicate
    3 expressive  the intended rule can be written in this bias (types, +/-/# markers, variable linking)
    4 correct     XHAIL learns rules that agree with the intended rule: on every labelled atom
                  (training + held-out) when the task has held-out labels, else on the whole domain
    5 tight       no more modes than the gold bias

Levels 3-5 need the intended rule (task["target"]), so they are for scoring and training
data only. Feedback shown to a model at test time comes from feedback.diagnose, which
never sees the target.
"""

from __future__ import annotations

import re

import clingo

from common import background_only, run_xhail, splice, _guard, domain_facts
from feedback import EXAMPLE_ATOM_RE, MODE_PARTS_RE, lint

LEVELS = ["format", "semantics", "expressive", "correct", "tight"]
VAR = re.compile(r"[A-Z]\w*")
BUILTIN = re.compile(r"!=|<=|>=|<|>|=")


def split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    return out + [cur.strip()] if cur.strip() else out


def parse_atom(s: str) -> tuple[str, list[str]]:
    s = s.strip()
    name, _, rest = s.partition("(")
    return name.strip(), split_top(rest[:-1]) if rest else []


def parse_rule(rule: str) -> tuple[str, list[str]]:
    rule = rule.strip().rstrip(".")
    head, _, body = rule.partition(":-")
    return head.strip(), split_top(body)


def ground_atoms(program: str) -> set[str]:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], background_only(program))
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as h:
        for m in h:
            return {str(s) for s in m.symbols(atoms=True)}
    return set()


# ---------------------------------------------------------------------------
# Level 3: can the intended rule be written in this bias?
# ---------------------------------------------------------------------------


def _match(term: str, schema: str, bg: set[str], env: dict, inputs: list, outputs: list) -> bool:
    """Match one target term against one mode schema term, collecting variable roles."""
    if schema[0] in "+-#":
        kind, typ = schema[0], schema[1:]
        if kind == "#":
            return not VAR.fullmatch(term) and f"{typ}({term})" in bg
        if not VAR.fullmatch(term):
            return False
        (inputs if kind == "+" else outputs).append((term, typ))
        return True
    tn, targs = parse_atom(term)
    sn, sargs = parse_atom(schema)
    return tn == sn and len(targs) == len(sargs) and all(
        _match(t, s, bg, env, inputs, outputs) for t, s in zip(targs, sargs))


def _match_literal(lit: str, modes: list[str], kind: str, bg: set[str]):
    neg = lit.startswith("not ")
    atom = lit[4:].strip() if neg else lit
    for m in modes:
        p = MODE_PARTS_RE.fullmatch(m)
        if not p or p.group(1) != kind or bool(p.group(2)) != neg:
            continue
        schema = m.split(None, 2 if neg else 1)[-1].rstrip(".").strip()
        inputs, outputs = [], []
        if _match(atom, schema, bg, {}, inputs, outputs):
            yield inputs, outputs


def expressive(target_rules: list[str], modes: list[str], bg: set[str]) -> str | None:
    """None if every target rule fits the bias, else the reason it does not."""
    for rule in target_rules:
        head, body = parse_rule(rule)
        heads = list(_match_literal(head, modes, "h", bg))
        if not heads:
            return f"no #modeh can produce the head `{head}`"
        types = {v: t for v, t in heads[0][0] + heads[0][1]}
        todo = []
        for lit in body:
            name, args = parse_atom(lit.removeprefix("not ").strip())
            if BUILTIN.search(lit):
                continue
            if len(args) == 1 and VAR.fullmatch(args[0]) and types.get(args[0]) == name:
                continue  # type guard such as patient(X)
            todo.append(lit)
        progress = True
        while todo and progress:
            progress = False
            for lit in list(todo):
                for inputs, outputs in _match_literal(lit, modes, "b", bg):
                    if all(types.get(v) == t for v, t in inputs) and all(types.get(v, t) == t for v, t in outputs):
                        types.update(dict(outputs))
                        todo.remove(lit)
                        progress = True
                        break
        if todo:
            return f"no #modeb (with matching types and +/- links) for `{todo[0]}` in `{rule}`"
    return None


# ---------------------------------------------------------------------------
# Level 4: does what XHAIL learns behave like the intended rule?
# ---------------------------------------------------------------------------


def extension(program: str, rules: list[str], head_pred: str | None) -> set[str] | None:
    """Atoms true in background + rules: those of head_pred, or all atoms when head_pred is None."""
    bg = background_only(program)
    prog = bg + "\n" + domain_facts(bg) + "\n" + "\n".join(_guard(r) for r in rules)
    try:
        atoms = ground_atoms(prog)
    except Exception:
        return None
    return atoms if head_pred is None else {a for a in atoms if a.startswith(head_pred + "(") or a == head_pred}


def agrees(task: dict, rules: list[str]) -> bool:
    """Do `rules` behave like the task's intended rule? With held-out labels, compare on every
    labelled atom (training examples + held-out, whatever their predicate: in event-calculus tasks
    they are holdsAt atoms, not the learned happens atoms). Without, compare the head predicate's
    whole extension."""
    target = task["target"] if isinstance(task["target"], list) else [task["target"]]
    if task.get("holdout"):
        labelled = {re.sub(r"\s+", "", a) for a in  # clingo prints atoms without spaces
                    [h["atom"] for h in task["holdout"]] + EXAMPLE_ATOM_RE.findall(task["program"])}
        want, got = extension(task["program"], target, None), extension(task["program"], rules, None)
        return want is not None and got is not None and want & labelled == got & labelled
    head_pred = parse_atom(parse_rule(target[0])[0])[0]
    want, got = extension(task["program"], target, head_pred), extension(task["program"], rules, head_pred)
    return want is not None and got == want


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def climb(task: dict, completion: str, timeout: float = 20.0) -> dict:
    """Return {"level": 0-5, "failed": level name or None, "reason": str, "hypothesis": [...]}."""
    def fail(i: int, reason: str, **kw) -> dict:
        return {"level": i, "failed": LEVELS[i], "reason": reason, **kw}

    program = task["program"]
    lines = [ln.strip() for ln in completion.strip().splitlines() if ln.strip()]
    bad = [ln for ln in lines if not MODE_PARTS_RE.fullmatch(re.sub(r"\s+", " ", ln))]
    modes = [re.sub(r"\s*([(),])\s*", r"\1", re.sub(r"\s+", " ", ln)) for ln in lines if ln not in bad]
    if not lines:
        return fail(0, "empty output")
    if bad:
        return fail(0, f"not a mode declaration: `{bad[0]}`")
    if not any(m.startswith("#modeh") for m in modes):
        return fail(0, "no #modeh")

    issues = lint(program, modes)
    if issues:
        return fail(1, issues[0], modes=modes)

    target = task.get("target")
    target_rules = target if isinstance(target, list) else [target] if target else []
    bg = ground_atoms(program)
    if target_rules:
        why = expressive(target_rules, modes, bg)
        if why:
            return fail(2, why, modes=modes)

    res = run_xhail(splice(program, modes), timeout_s=timeout)
    hyp = res.get("hypothesis", [])
    if not (res.get("ok") and res.get("success")):
        return fail(3, "XHAIL timed out" if res.get("timeout") else f"XHAIL learned nothing ({res.get('error', 'no hypothesis')})",
                    modes=modes)
    if target_rules and not agrees(task, hyp):
        return fail(3, f"learned {hyp}, which behaves differently from the intended rule", modes=modes, hypothesis=hyp)

    if len(modes) > len(task["gold_modes"]):
        return fail(4, f"{len(modes)} modes where gold needs {len(task['gold_modes'])}", modes=modes, hypothesis=hyp)
    return {"level": 5, "failed": None, "reason": "", "modes": modes, "hypothesis": hyp}
