"""Verify-then-prune: shrink a working bias to the modes its learned rule actually needs.

Uses only XHAIL on the training examples (never the target), so it is fair at test time.
A #modeb is dropped when XHAIL still solves and learns the same hypothesis without it.
"""

from __future__ import annotations

from common import canon_hypothesis, run_xhail, splice


def prune(program: str, modes: list[str], timeout: float = 20.0) -> list[str]:
    res = run_xhail(splice(program, modes), timeout_s=timeout)
    if not (res.get("ok") and res.get("success")):
        return modes
    target = canon_hypothesis(res["hypothesis"])
    kept = list(modes)
    for m in reversed(modes):
        if not m.startswith("#modeb"):
            continue
        trial = [x for x in kept if x != m]
        r = run_xhail(splice(program, trial), timeout_s=timeout)
        if r.get("ok") and r.get("success") and canon_hypothesis(r["hypothesis"]) == target:
            kept = trial
    return kept
