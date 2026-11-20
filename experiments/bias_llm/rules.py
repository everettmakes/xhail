"""The direct-rule baseline: the model writes the hypothesis rule itself instead of a bias.

Same programs and the same vocabulary block as the bias job, a different system prompt and
target. Two checks mirror the bias job:

    check_rule   uses only the program (background + training examples): does the rule fit
                 every training example? This is the verifier used between attempts, the
                 counterpart of "XHAIL found a hypothesis".
    score_rule   uses the hidden target: format -> consistent -> correct, where "correct" is
                 the same test as ladder level 4 (agrees with the intended rule on the labelled
                 atoms, or the whole domain when there are no held-out labels).
"""

from __future__ import annotations

import re

from common import background_only, build_prompt, domain_facts, _solve, strip_comments
from ladder import agrees

RULE_SYSTEM_PROMPT = (
    "You write hypotheses for the XHAIL inductive logic programming system. Given background "
    "knowledge and examples, output only the rule or rules that explain the examples, one clause "
    "per line, in clingo syntax (head :- body.). Use 'not' for negated body literals."
)
RULE_LEVELS = ["format", "consistent", "correct"]
EXAMPLE_RE = re.compile(r"#example\s+(not\s+)?([a-z]\w*(?:\([^.]*\))?)\s*[.\[]")
VAR_RE = re.compile(r"(?<![\w\"'])([A-Z_][A-Za-z0-9_]*)")


def rule_prompt(program: str) -> list[dict]:
    sys_msg, user = build_prompt(program)
    return [{"role": "system", "content": RULE_SYSTEM_PROMPT}, user]


def rule_text(task: dict) -> str:
    target = task["target"]
    return "\n".join(target) if isinstance(target, list) else target


def extract_rules(text: str) -> list[str]:
    """Clause-like lines: not directives, ending in a full stop."""
    out = []
    for line in strip_comments(text).splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line.endswith(".") and re.match(r"[a-z]", line):
            out.append(re.sub(r"\s+", " ", line))
    return out


def safe(rule: str) -> str:
    """Ground every variable over dom/1 so a model-written rule is safe for clingo."""
    rule = rule.strip()
    vars_ = sorted({v for v in VAR_RE.findall(rule) if v != "_"})
    if not vars_:
        return rule
    guards = ", ".join(f"dom({v})" for v in vars_)
    if ":-" in rule:
        head, body = rule[:-1].split(":-", 1)
        return f"{head.strip()} :- {body.strip()}, {guards}."
    return f"{rule[:-1]} :- {guards}."


def _model(program: str, rules: list[str]) -> set[str] | None | str:
    bg = background_only(program)
    try:
        return _solve(bg + "\n" + domain_facts(bg) + "\n" + "\n".join(safe(r) for r in rules))
    except Exception as e:  # parse / grounding errors
        return f"error: {e}"[:200]


def check_rule(program: str, text: str) -> dict:
    """Training-data check only (no hidden labels): returns {solved, category, feedback}."""
    rules = extract_rules(text)
    if not rules:
        return {"solved": False, "category": "no_rules", "feedback": "no clauses found"}
    model = _model(program, rules)
    if isinstance(model, str):
        return {"solved": False, "category": "parse", "feedback": model}
    if model is None:
        return {"solved": False, "category": "unsat", "feedback": "background + rules has no answer set"}
    wrong = [("not " if neg else "") + atom for neg, atom in EXAMPLE_RE.findall(program)
             if (re.sub(r"\s+", "", atom) in model) == bool(neg)]  # clingo prints atoms without spaces
    if wrong:
        return {"solved": False, "category": "inconsistent", "feedback": f"misses {len(wrong)} example(s), e.g. {wrong[0]}"}
    return {"solved": True, "category": "solved", "feedback": ""}


def score_rule(task: dict, text: str) -> int:
    """0 = no usable rule, 1 = parses, 2 = fits the training examples, 3 = correct."""
    rules = extract_rules(text)
    if not rules or isinstance(_model(task["program"], rules), str):
        return 0
    if not check_rule(task["program"], text)["solved"]:
        return 1
    return 3 if agrees(task, [safe(r) for r in rules]) else 2
