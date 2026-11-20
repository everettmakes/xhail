"""Shared helpers: split XHAIL programs, run XHAIL safely, score learned rules."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path

import clingo

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from xhail import learn_from_string  # noqa: E402

# How XHAIL runs are isolated. "fork" is fastest from a small process; from a large GPU process
# use "forkserver" (set XHAIL_MP_START) so each run forks from a small helper instead.
MP_START = os.environ.get("XHAIL_MP_START", "fork")

MODE_RE = re.compile(r"#mode[hb]\b[^.]*\.")
EXAMPLE_RE = re.compile(r"#example\b[^.]*\.")
COMMENT_RE = re.compile(r"%[^\n]*")

SYSTEM_PROMPT = (
    "You write language bias for the XHAIL inductive logic programming system. "
    "Given background knowledge and examples, output only the mode declarations "
    "(#modeh / #modeb), one per line. Use +type for input variables, -type for "
    "output variables and #type for ground constants. Prefix a #modeb schema with "
    "'not' to allow a negated body literal."
)


# ---------------------------------------------------------------------------
# Program manipulation
# ---------------------------------------------------------------------------


def strip_comments(program: str) -> str:
    return COMMENT_RE.sub("", program)


def extract_modes(text: str) -> list[str]:
    """Return canonical mode declarations found in text (order kept, deduped)."""
    seen, out = set(), []
    for m in MODE_RE.findall(strip_comments(text)):
        c = canon_mode(m)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def canon_mode(mode: str) -> str:
    mode = re.sub(r"\s+", " ", mode.strip())
    mode = re.sub(r"\s*([(),])\s*", r"\1", mode)
    return mode


def remove_modes(program: str) -> str:
    """Program with comments and mode declarations removed, blank lines squashed."""
    body = MODE_RE.sub("", strip_comments(program))
    lines = [ln.rstrip() for ln in body.splitlines()]
    out, blank = [], False
    for ln in lines:
        if not ln.strip():
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(ln)
        blank = False
    return "\n".join(out).strip() + "\n"


def background_only(program: str) -> str:
    """Plain ASP background: no modes, no examples, no comments."""
    return EXAMPLE_RE.sub("", remove_modes(program))


def splice(program_without_modes: str, modes: list[str]) -> str:
    return program_without_modes.rstrip() + "\n\n" + "\n".join(modes) + "\n"


def vocabulary(program: str, max_members: int = 6) -> str:
    """What the program offers a bias: predicates with arity, example predicates, candidate types.

    Computed from the background's first model and the #example lines only, so it is
    available at test time and never reveals the answer.
    """
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], background_only(program))
    ctl.ground([("base", [])])
    atoms: list[clingo.Symbol] = []
    with ctl.solve(yield_=True) as h:
        for m in h:
            atoms = list(m.symbols(atoms=True))
            break
    sigs = sorted({(a.name, len(a.arguments)) for a in atoms})
    types: dict[str, list[str]] = {}
    for a in atoms:
        if len(a.arguments) == 1:
            types.setdefault(a.name, []).append(str(a.arguments[0]))
    ex = set()
    for e in EXAMPLE_RE.findall(program):
        name, args = re.match(r"#example\s+(?:not\s+)?(\w+)(\(.*\))?", e).groups()
        depth, arity = 0, 1 if args else 0
        for ch in (args or "")[1:-1]:
            depth += ch == "("
            depth -= ch == ")"
            arity += ch == "," and depth == 0
        ex.add(f"{name}/{arity}")
    ex = sorted(ex)

    def members(t: str) -> str:
        ms = sorted(types[t])
        return ", ".join(ms[:max_members]) + (", ..." if len(ms) > max_members else "")

    lines = [
        "Background predicates: " + ", ".join(f"{n}/{a}" for n, a in sigs),
        "Example predicates: " + ", ".join(ex),
        "Unary predicates (usable as types): " + "; ".join(f"{t} = {{{members(t)}}}" for t in sorted(types)),
    ]
    return "\n".join(lines)


def shape(program: str) -> frozenset[str]:
    """Coarse structural features of a task, used to retrieve similar worked examples.

    Built only from the program (background + examples), never from modes or answers.
    """
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], background_only(program))
    ctl.ground([("base", [])])
    atoms: list[clingo.Symbol] = []
    with ctl.solve(yield_=True) as h:
        for m in h:
            atoms = list(m.symbols(atoms=True))
            break
    unary = {}
    for a in atoms:
        if len(a.arguments) == 1:
            unary.setdefault(a.name, set()).add(str(a.arguments[0]))
    feats = set()
    ex = EXAMPLE_RE.findall(program)
    names = {re.match(r"#example\s+(?:not\s+)?(\w+)", e).group(1) for e in ex}
    feats |= {f"ex:{n}" for n in names if n in ("holdsAt",)}
    feats.add(f"ex_arity:{max((e.count(',') + 1 if '(' in e else 0) for e in ex) if ex else 0}")
    feats.add(f"neg_examples:{any(' not ' in e for e in ex)}")
    if "clipped(" in background_only(program):
        feats.add("event_calculus")
    seconds: dict[str, set[str]] = {}
    for a in atoms:
        if len(a.arguments) == 2 and a.name not in ("happens", "holdsAt", "initiates", "terminates"):
            feats.add("binary")
            seconds.setdefault(a.name, set()).add(str(a.arguments[1]))
    for vals in seconds.values():  # e.g. size(b1,small): few distinct, non-numeric, typed values
        if len(vals) <= 4 and not any(v.isdigit() for v in vals) and any(vals <= ms for ms in unary.values()):
            feats.add("const_valued_attr")
    feats.add(f"unary_preds:{min(len(unary), 6) // 2}")
    return frozenset(feats)


def retrieve_shots(program: str, pool: list[dict], k: int, seed: int = 0, exclude: str | None = None) -> list[dict]:
    """k worked examples (user/assistant turns) from pool tasks with the most similar shape."""
    target = shape(program)
    rng = random.Random(seed)
    scored = []
    for t in pool:
        if t["id"] == exclude:
            continue
        other = t.get("_shape") or t.setdefault("_shape", shape(t["program"]))
        scored.append((len(target & other) / len(target | other), rng.random(), t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    msgs = []
    for _, _, t in scored[:k]:
        msgs.append(build_prompt(t["program"])[1])
        msgs.append({"role": "assistant", "content": "\n".join(t["gold_modes"])})
    return msgs


def build_prompt(program_without_modes: str) -> list[dict]:
    content = program_without_modes.rstrip() + "\n\n" + vocabulary(program_without_modes) if program_without_modes.strip() else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def repair_prompt(program_without_modes: str, broken: str, feedback: str) -> list[dict]:
    """Second-round prompt: the program, a bias that did not work, and XHAIL's diagnosis of it."""
    sys_msg, user = build_prompt(program_without_modes)
    user = {"role": "user", "content": user["content"] + "\n\nProposed modes:\n" + broken.strip()
            + "\n\nXHAIL check:\n" + feedback.strip() + "\n\nWrite the corrected mode declarations."}
    return [sys_msg, user]


def fewshot_messages(k: int, seed: int) -> list[dict]:
    """k solved training tasks as prior chat turns, one per family where possible."""
    if k == 0:
        return []
    pool = load_tasks(str(Path(__file__).parent / "data" / "train.jsonl"))
    rng = random.Random(seed)
    rng.shuffle(pool)
    picked, fams = [], set()
    for t in pool:
        if t["family"] not in fams or len(fams) == 4:
            picked.append(t)
            fams.add(t["family"])
        if len(picked) == k:
            break
    msgs = []
    for t in picked:
        msgs.append(build_prompt(t["program"])[1])
        msgs.append({"role": "assistant", "content": "\n".join(t["gold_modes"])})
    return msgs


# ---------------------------------------------------------------------------
# Running XHAIL with a hard timeout (LLM output can make search blow up)
# ---------------------------------------------------------------------------


def _worker(program: str, depth: int, q: mp.Queue) -> None:
    try:
        t0 = time.perf_counter()
        r = learn_from_string(program, depth=depth)
        q.put({"ok": True, "success": r.success, "hypothesis": list(r.hypothesis),
               "runtime_s": time.perf_counter() - t0})
    except Exception as e:  # parse errors, clingo errors, ...
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})


def run_xhail(program: str, timeout_s: float = 20.0, depth: int = 10) -> dict:
    ctx = mp.get_context(MP_START)
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(program, depth, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.kill()
        p.join()
        return {"ok": False, "error": "timeout", "runtime_s": timeout_s, "timeout": True}
    try:
        return q.get(timeout=1)
    except Exception:
        return {"ok": False, "error": f"crashed (exit {p.exitcode})"}


# ---------------------------------------------------------------------------
# Scoring a learned hypothesis on held-out examples
# ---------------------------------------------------------------------------

VAR_RE = re.compile(r"\bV\d+\b")


def _guard(rule: str) -> str:
    """Make a learned rule safe for clingo by grounding each variable over dom/1."""
    rule = rule.strip()
    if rule.endswith(":- ."):
        rule = rule[:-4].strip() + "."
    vars_ = sorted(set(VAR_RE.findall(rule)))
    if not vars_:
        return rule
    guards = ", ".join(f"dom({v})" for v in vars_)
    if ":-" in rule:
        head, body = rule[:-1].split(":-", 1)
        return f"{head.strip()} :- {body.strip()}, {guards}."
    return f"{rule[:-1]} :- {guards}."


def _solve(program: str) -> set[str] | None:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    model: set[str] | None = None
    with ctl.solve(yield_=True) as h:
        for m in h:
            model = {str(s) for s in m.symbols(atoms=True)}
            break
    return model


def _terms(sym: clingo.Symbol, acc: set[str]) -> None:
    if sym.type == clingo.SymbolType.Function:
        for a in sym.arguments:
            acc.add(str(a))
            _terms(a, acc)


def domain_facts(background: str) -> str:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], background)
    ctl.ground([("base", [])])
    acc: set[str] = set()
    with ctl.solve(yield_=True) as h:
        for m in h:
            for s in m.symbols(atoms=True):
                _terms(s, acc)
            break
    return " ".join(f"dom({t})." for t in sorted(acc))


def holdout_accuracy(background: str, rules: list[str], holdout: list[dict]) -> float | None:
    """Fraction of held-out examples classified correctly by BG + rules."""
    if not holdout:
        return None
    prog = background + "\n" + domain_facts(background) + "\n" + "\n".join(_guard(r) for r in rules)
    try:
        model = _solve(prog)
    except Exception:
        return 0.0
    if model is None:
        return 0.0
    correct = sum((ex["atom"] in model) == ex["positive"] for ex in holdout)
    return correct / len(holdout)


# ---------------------------------------------------------------------------
# Mode-set overlap with gold
# ---------------------------------------------------------------------------


def mode_prf(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    p, g = set(pred), set(gold)
    if not p or not g:
        return 0.0, 0.0, 0.0
    tp = len(p & g)
    prec, rec = tp / len(p), tp / len(g)
    f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Task loading (synthetic JSONL, or the hand-written benchmarks as an OOD set)
# ---------------------------------------------------------------------------

BENCHMARKS_DIR = REPO_ROOT / "experiments" / "benchmarks"


GOLD_DIR = Path(__file__).parent / "gold"
TARGET_RE = re.compile(r"^%\s*target:\s*(.+)$", re.M)


def load_gold() -> list[dict]:
    """Hand-written gold set: the 10 benchmarks plus gold/*.lp, each with its intended rules."""
    import json

    bench_targets = json.loads((GOLD_DIR / "benchmark_targets.json").read_text())
    tasks = []
    for t in load_tasks("benchmarks"):
        t["target"] = bench_targets[t["id"].removeprefix("bench-")]
        tasks.append(t)
    for f in sorted(GOLD_DIR.glob("*.lp")):
        text = f.read_text()
        tasks.append({
            "id": f"gold-{f.stem}", "family": "gold", "program": remove_modes(text),
            "gold_modes": extract_modes(text), "target": TARGET_RE.findall(text),
            "holdout": [], "gold_hypothesis": None,
        })
    return tasks


def load_tasks(source: str) -> list[dict]:
    """Load tasks from a JSONL file, 'benchmarks' (experiments/benchmarks/*.lp) or 'gold' (benchmarks + gold/)."""
    import json

    if source == "gold":
        return load_gold()
    if source == "benchmarks":
        tasks = []
        for f in sorted(BENCHMARKS_DIR.glob("*.lp")):
            text = f.read_text()
            tasks.append({
                "id": f"bench-{f.stem}", "family": "benchmark",
                "program": remove_modes(text), "gold_modes": extract_modes(text),
                "holdout": [], "gold_hypothesis": None,
            })
        return tasks
    with open(source) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def canon_hypothesis(rules: list[str]) -> tuple[str, ...]:
    """Order- and variable-name-insensitive form of a hypothesis (approximate)."""
    out = []
    for r in rules:
        r = r.strip().rstrip(".").replace(" ", "")
        head, _, body = r.partition(":-")
        lits, depth, cur = [], 0, ""
        for ch in body:
            if ch == "," and depth == 0:
                lits.append(cur)
                cur = ""
                continue
            depth += ch == "("
            depth -= ch == ")"
            cur += ch
        if cur:
            lits.append(cur)
        lits.sort(key=lambda s: VAR_RE.sub("V", s))
        text = head + (":-" + ",".join(lits) if lits else "")
        mapping: dict[str, str] = {}
        text = VAR_RE.sub(lambda m: mapping.setdefault(m.group(), f"V{len(mapping) + 1}"), text)
        out.append(text)
    return tuple(sorted(out))
