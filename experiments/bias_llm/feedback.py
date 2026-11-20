"""Run XHAIL on a candidate bias and turn the outcome into feedback for the model.

Feedback is built only from the program (background + training examples) and from
what XHAIL's phases did with the bias. Held-out labels are never used, so a model
that reads this feedback cannot see the answers it is scored on.
"""

from __future__ import annotations

import multiprocessing as mp
import re
import time

import clingo

from common import MP_START, background_only, learn_from_string, splice

EXAMPLE_ATOM_RE = re.compile(r"#example\s+(?:not\s+)?([a-z]\w*(?:\([^.]*\))?)\s*[.\[]")
MODE_PARTS_RE = re.compile(r"#mode([hb])\s+(not\s+)?([a-z]\w*)(\((.*)\))?\s*\.")
TYPE_RE = re.compile(r"[+\-#]([a-z]\w*)")


def _top_level_arity(args: str) -> int:
    if not args.strip():
        return 0
    depth, n = 0, 1
    for ch in args:
        depth += ch == "("
        depth -= ch == ")"
        n += ch == "," and depth == 0
    return n


def _signature(atom: str) -> tuple[str, int]:
    name, _, rest = atom.partition("(")
    return name.strip(), _top_level_arity(rest[:-1]) if rest else 0


def background_signatures(program: str) -> set[tuple[str, int]]:
    """name/arity of every predicate true in the background's (first) model."""
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], background_only(program))
    ctl.ground([("base", [])])
    sigs: set[tuple[str, int]] = set()
    with ctl.solve(yield_=True) as h:
        for m in h:
            sigs = {(s.name, len(s.arguments)) for s in m.symbols(atoms=True)}
            break
    return sigs


def _rule_bodies(program: str) -> str:
    """Text of every background rule body (abducible heads such as happens/2 occur only there)."""
    return " ".join(c.split(":-", 1)[1] for c in background_only(program).split(".") if ":-" in c)


def lint(program: str, modes: list[str]) -> list[str]:
    """Static problems: wrong predicates, arities and types. Cheap, no search."""
    bg = background_signatures(program)
    ex = {_signature(a) for a in EXAMPLE_ATOM_RE.findall(program)}
    types = sorted(n for n, a in bg if a == 1)
    fmt = lambda s: f"{s[0]}/{s[1]}"  # noqa: E731
    issues = []
    for m in modes:
        p = MODE_PARTS_RE.fullmatch(m)
        if not p:
            issues.append(f"`{m}` is not a valid mode declaration.")
            continue
        kind, _, name, _, args = p.groups()
        sig = (name, _top_level_arity(args or ""))
        if kind == "h" and sig not in ex and not re.search(rf"\b{name}\(", _rule_bodies(program)):
            issues.append(f"`{m}`: {fmt(sig)} is neither the predicate in the examples "
                          f"({', '.join(sorted(map(fmt, ex)))}) nor used in the body of a background rule.")
        if kind == "b" and sig not in bg:
            issues.append(f"`{m}`: {fmt(sig)} is not defined in the background "
                          f"(available: {', '.join(sorted(map(fmt, bg)))}).")
        for t in TYPE_RE.findall(args or ""):
            if (t, 1) not in bg:
                issues.append(f"`{m}`: type `{t}` is not a unary background predicate "
                              f"(candidates: {', '.join(types)}).")
    return issues


# ---------------------------------------------------------------------------
# Running XHAIL and recording how far each phase got
# ---------------------------------------------------------------------------


def _worker(program: str, q: mp.Queue) -> None:
    phases: dict = {}
    t0 = time.perf_counter()
    try:
        r = learn_from_string(program, on_phase=lambda ph, info: phases.__setitem__(ph, info))
        q.put({"ok": True, "success": r.success, "hypothesis": list(r.hypothesis),
               "phases": phases, "runtime_s": time.perf_counter() - t0})
    except Exception as e:
        q.put({"ok": False, "phases": phases, "error": f"{type(e).__name__}: {e}"[:300]})


def run_phases(program: str, timeout_s: float = 20.0) -> dict:
    ctx = mp.get_context(MP_START)
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(program, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.kill()
        p.join()
        return {"ok": False, "timeout": True, "phases": {}, "error": "timeout"}
    try:
        return q.get(timeout=1)
    except Exception:
        return {"ok": False, "phases": {}, "error": f"crashed (exit {p.exitcode})"}


def diagnose(program: str, modes: list[str], timeout_s: float = 20.0) -> dict:
    """Run XHAIL on program + modes. Returns solved flag, a failure category and feedback text."""
    issues = lint(program, modes)
    if not modes:
        return {"solved": False, "category": "no_modes", "feedback":
                "Your answer contained no mode declarations. Write one per line, e.g. "
                "`#modeh p(+type).` and `#modeb q(+type,-type).`"}
    if not any(m.startswith("#modeh") for m in modes):
        return {"solved": False, "category": "no_modeh", "feedback": "There is no #modeh declaration. "
                "Add one for the predicate used in the examples.\n" + "\n".join(issues)}

    res = run_phases(splice(program, modes), timeout_s)
    ph = res.get("phases", {})
    if res.get("ok") and res.get("success"):
        return {"solved": True, "category": "solved", "hypothesis": res["hypothesis"], "feedback": ""}

    err = res.get("error", "")
    if res.get("timeout") or "DeductionTimeout" in err:
        cat, msg = "timeout", ("XHAIL timed out: the bias is too broad and the search blew up. "
                               "Remove body modes that are unlikely to matter.")
    elif "ParseError" in err or "LexError" in err:
        cat, msg = "parse", f"XHAIL could not parse your modes: {err}"
    elif "abduction" not in ph:
        cat, msg = "abduction", (
            "Abduction failed: no set of head atoms allowed by your #modeh explains the positive "
            "examples while avoiding the negative ones. Check the #modeh predicate, its argument "
            "types, and whether each argument should be +type (input) or #type (constant).")
    elif not any(m.startswith("#modeb") for m in modes):
        cat, msg = "kernel", "There are no #modeb declarations, so no rule body can be built. Add body modes."
    elif ph.get("deduction", {}).get("kernel_size", 1) == 0:
        cat, msg = "kernel", (
            f"Abduction found {ph['abduction'].get('delta_size')} head atom(s) to explain, but deduction "
            "built an empty kernel: none of your #modeb literals hold for those atoms. "
            "The body modes do not connect to the head's arguments (check types and +/- markers).")
    else:
        cat, msg = "induction", (
            f"Abduction found {ph['abduction'].get('delta_size')} head atom(s), deduction built "
            f"{ph.get('deduction', {}).get('kernel_size')} kernel clause(s), but induction found no rule "
            "that covers the positive examples without covering a negative one. The body modes are "
            "probably missing a relevant predicate, a negated form (`#modeb not p(+t).`), or a "
            "-type output variable to link through an intermediate object.")
    if issues:
        msg += "\n\nOther problems:\n- " + "\n- ".join(issues)
    return {"solved": False, "category": cat, "feedback": msg}
