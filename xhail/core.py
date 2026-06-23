"""
xhail.core — public learning API
=================================

The primary entry point for running XHAIL programmatically.  The
:func:`learn` function drives the three-phase pipeline (abduction →
deduction → induction) and returns a :class:`LearningResult` containing
the learned hypothesis.

Example::

    from xhail import learn, learn_from_string

    # From a file
    result = learn("penguins.lp", depth=10)
    if result.success:
        print(result)               # flies(V1) :- not penguin(V1).
        print(result.to_dict())     # {"success": True, "hypothesis": [...], ...}

    # From a string — great for notebooks and APIs
    result = learn_from_string('''
        bird(a). penguin(b).
        #modeh flies(+bird).
        #modeb not penguin(+bird).
        #example flies(a).
        #example not flies(b).
    ''')
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from .parser.parser import Parser
from .reasoning.abduction import Abduction
from .reasoning.deduction import Deduction
from .reasoning.induction import Induction
from .reasoning.model import Model

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """The output of a XHAIL learning run.

    Attributes:
        hypothesis: The learned rules as formatted strings.
        success: True if at least one rule was learned.
        source: The file or string label used as input.
        n_examples: Number of examples supplied to the learner.
        n_background: Number of background clauses supplied.

    Serialisation::

        result.to_dict()   # plain dict — JSON-serialisable
        result.to_json()   # JSON string
        result.to_lp()     # hypothesis as an ASP program string

    Jupyter display::

        result             # renders as a styled HTML table in a notebook cell
    """

    hypothesis: list[str] = field(default_factory=list)
    all_hypotheses: list[list[str]] = field(default_factory=list)
    success: bool = False
    source: Optional[str] = None
    n_examples: int = 0
    n_background: int = 0

    # Phase timing — populated by learn() when on_phase is used or internally
    _phase_times: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    # ---------- string representations ----------

    def __str__(self) -> str:
        if not self.success:
            return "No hypothesis found."
        return "\n".join(self.hypothesis)

    def __repr__(self) -> str:
        return (
            f"LearningResult(success={self.success}, "
            f"n_rules={len(self.hypothesis)}, source={self.source!r})"
        )

    # ---------- serialisation ----------

    @property
    def n_rules(self) -> int:
        """Number of rules in the learned hypothesis."""
        return len(self.hypothesis)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all result fields.

        Example::

            {
                "success": True,
                "n_rules": 1,
                "hypothesis": ["flies(V1) :- not penguin(V1)."],
                "source": "penguins.lp",
                "n_examples": 4,
                "n_background": 3,
                "phase_times_ms": {"abduction": 3.1, "deduction": 1.8, "induction": 2.4}
            }
        """
        return {
            "success": self.success,
            "n_rules": self.n_rules,
            "hypothesis": list(self.hypothesis),
            "source": self.source,
            "n_examples": self.n_examples,
            "n_background": self.n_background,
            "phase_times_ms": {k: round(v * 1000, 2) for k, v in self._phase_times.items()},
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string.

        Args:
            indent: Number of spaces for pretty-printing (default 2).
                    Pass ``indent=None`` for compact output.

        Example::

            print(result.to_json())
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_lp(self) -> str:
        """Return the learned hypothesis as an ASP program string.

        The output is a valid ``.lp`` snippet that can be appended to
        background knowledge and used directly with clingo.

        Example::

            with open("hypothesis.lp", "w") as f:
                f.write(result.to_lp())
        """
        if not self.hypothesis:
            return "% No hypothesis found.\n"
        return "\n".join(self.hypothesis) + "\n"

    # ---------- Jupyter / IPython display ----------

    def _repr_html_(self) -> str:  # pragma: no cover
        """Rich HTML display for Jupyter notebooks.

        Renders a compact summary table followed by the learned rules in a
        code block.  Jupyter calls this automatically when a ``LearningResult``
        is the last expression in a cell.
        """
        status_colour = "#2a9d5c" if self.success else "#c0392b"
        status_text = "✓ Hypothesis found" if self.success else "✗ No hypothesis"

        rows = [
            ("Status", f"<span style='color:{status_colour};font-weight:bold'>{status_text}</span>"),
            ("Rules learned", str(self.n_rules)),
            ("Examples", str(self.n_examples)),
            ("Background clauses", str(self.n_background)),
            ("Source", f"<code>{self.source or '&lt;string&gt;'}</code>"),
        ]
        if self._phase_times:
            for phase, t in self._phase_times.items():
                rows.append((f"{phase.capitalize()} time", f"{t*1000:.1f} ms"))

        table_rows = "".join(
            f"<tr><td style='padding:4px 12px 4px 0;color:#888;white-space:nowrap'>{k}</td>"
            f"<td style='padding:4px 0'>{v}</td></tr>"
            for k, v in rows
        )
        table = f"<table style='border-collapse:collapse;font-size:0.9em'>{table_rows}</table>"

        if self.hypothesis:
            rules_html = "\n".join(self.hypothesis)
            code_block = (
                f"<pre style='background:#f4f4f4;border-radius:4px;padding:10px 14px;"
                f"margin-top:8px;font-size:0.85em;overflow-x:auto'>{rules_html}</pre>"
            )
        else:
            code_block = ""

        return (
            f"<div style='font-family:sans-serif;line-height:1.5'>"
            f"<b>LearningResult</b><br>{table}{code_block}</div>"
        )


# ---------------------------------------------------------------------------
# Phase callback type
# ---------------------------------------------------------------------------

#: Callback signature for the ``on_phase`` parameter of :func:`learn`.
#:
#: Called after each phase completes with:
#:   - ``phase``:   one of ``"abduction"``, ``"deduction"``, ``"induction"``
#:   - ``info``:    dict with phase-specific metrics (elapsed_ms, etc.)
PhaseCallback = Callable[[str, dict], None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def learn(
    source: Union[str, Path],
    depth: int = 10,
    *,
    debug: bool = False,
    debug_output_dir: Optional[Union[str, Path]] = None,
    on_phase: Optional[PhaseCallback] = None,
    all_solutions: bool = False,
    timeout: Optional[int] = None,
) -> LearningResult:
    """Run the full XHAIL learning pipeline and return the learned hypothesis.

    XHAIL operates in three phases:

    1. **Abduction** — find a minimal set of atoms (Δ) that, together with
       the background knowledge, satisfies the examples.
    2. **Deduction** — build a *kernel set* of maximally-specific clauses
       that cover the abduced atoms.
    3. **Induction** — compress the kernel into a minimal hypothesis H that
       explains all examples.

    Args:
        source: Path to a ``.lp`` file (``str`` or :class:`pathlib.Path`),
            or a raw ASP program string.  A string that ends in ``.lp``
            and exists on disk is treated as a file path; everything else
            is parsed as a literal program string.
        depth: Maximum BFS depth for the deduction phase (default: 10).
        debug: Write intermediate ASP programs to *debug_output_dir*.
        debug_output_dir: Directory for debug output (default: ``./xhail_debug/``).
        on_phase: Optional callback invoked after each phase completes.
            Signature: ``(phase: str, info: dict) -> None``.  ``phase``
            is one of ``"abduction"``, ``"deduction"``, ``"induction"``.
            ``info`` always contains ``"elapsed_ms"``; abduction adds
            ``"delta_size"``; deduction adds ``"kernel_size"``; induction
            adds ``"n_rules"``.

            Example::

                def log_phase(phase, info):
                    print(f"{phase}: {info}")

                result = learn("penguins.lp", on_phase=log_phase)
                # abduction: {'elapsed_ms': 3.1, 'delta_size': 2}
                # deduction: {'elapsed_ms': 1.8, 'kernel_size': 5}
                # induction: {'elapsed_ms': 2.4, 'n_rules': 1}

    Returns:
        A :class:`LearningResult` containing the learned rules and metadata.

        all_solutions: If True, enumerate all optimal hypotheses (not just the
            first).  Results are stored in ``result.all_hypotheses`` as a list
            of hypothesis lists; ``result.hypothesis`` holds the first one.
        timeout: If set, stop clingo after this many seconds in the induction
            phase.  Partial results (if any) are returned.

    Raises:
        ParseError: If *source* cannot be parsed as a valid XHAIL program.
        FileNotFoundError: If *source* looks like a file path but does not exist.
        RuntimeError: If the abduction phase yields no model (UNSAT program).
        DeductionTimeoutError: If the deduction phase exceeds the time limit.
    """
    # Resolve source to either a file path or a raw string
    def _looks_like_path(s: str) -> bool:
        if len(s) > 255:
            return False
        try:
            return s.endswith(".lp") or Path(s).exists()
        except OSError:
            return False

    if isinstance(source, Path) or (isinstance(source, str) and _looks_like_path(source)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Input file not found: {source_path}")
        raw_string = None
        source_label = str(source_path)
    else:
        raw_string = source
        source_label = "<string>"

    # Resolve debug output directory
    output_dir: Optional[Path] = None
    if debug:
        output_dir = Path(debug_output_dir) if debug_output_dir else Path("xhail_debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Debug mode: intermediate programs will be written to %s", output_dir)

    # --- Parse ---
    logger.debug("Parsing %s ...", source_label)
    parser = Parser()
    if raw_string is not None:
        parser.loadString(raw_string)
    else:
        parser.loadFile(str(source_path))
    parser.parseProgram()
    EX, MH, MB, BG, DISP = parser.separate()
    logger.debug(
        "Parsed: %d example(s), %d modeh, %d modeb, %d background clause(s)",
        len(EX), len(MH), len(MB), len(BG),
    )

    # --- Build model and run phases ---
    model = Model(EX, MH, MB, BG, depth, debug_output_dir=output_dir)
    phase_times: dict[str, float] = {}

    # Phase 1 — Abduction
    logger.info("Phase 1: Abduction ...")
    t0 = time.perf_counter()
    Abduction(model).runPhase()
    phase_times["abduction"] = time.perf_counter() - t0
    if on_phase is not None:
        delta_size = len(list(model.getDelta())) if hasattr(model, "getDelta") else 0
        on_phase("abduction", {
            "elapsed_ms": round(phase_times["abduction"] * 1000, 2),
            "delta_size": delta_size,
        })

    # Phase 2 — Deduction
    logger.info("Phase 2: Deduction (depth=%d) ...", depth)
    t0 = time.perf_counter()
    Deduction(model).runPhase()
    phase_times["deduction"] = time.perf_counter() - t0
    if on_phase is not None:
        kernel_size = len(list(model.getKernel())) if hasattr(model, "getKernel") else 0
        on_phase("deduction", {
            "elapsed_ms": round(phase_times["deduction"] * 1000, 2),
            "kernel_size": kernel_size,
        })

    # Phase 3 — Induction
    logger.info("Phase 3: Induction ...")
    t0 = time.perf_counter()
    Induction(model).runPhase(all_solutions=all_solutions, timeout=timeout)
    phase_times["induction"] = time.perf_counter() - t0
    hypothesis_clauses = model.getHypothesis()
    hypothesis_strs = [str(c) for c in hypothesis_clauses]
    if on_phase is not None:
        on_phase("induction", {
            "elapsed_ms": round(phase_times["induction"] * 1000, 2),
            "n_rules": len(hypothesis_strs),
        })

    all_hyp_strs = [[str(c) for c in h] for h in model.getAllHypotheses()]

    # Apply #display filters: if any directives are present, keep only rules
    # whose head predicate matches at least one directive.
    if DISP:
        def _display_filter(rules: list[str]) -> list[str]:
            return [r for r in rules if any(d.matches(r) for d in DISP)]
        hypothesis_strs = _display_filter(hypothesis_strs)
        all_hyp_strs = [_display_filter(h) for h in all_hyp_strs]
        logger.debug(
            "#display filter applied (%d directive(s)): %d rule(s) remaining.",
            len(DISP), len(hypothesis_strs),
        )

    result = LearningResult(
        hypothesis=hypothesis_strs,
        all_hypotheses=all_hyp_strs,
        success=bool(hypothesis_strs),
        source=source_label,
        n_examples=len(EX),
        n_background=len(BG),
    )
    result._phase_times = phase_times
    return result


def learn_from_string(
    program: str,
    depth: int = 10,
    *,
    on_phase: Optional[PhaseCallback] = None,
) -> LearningResult:
    """Run the XHAIL pipeline on a raw ASP program string.

    A convenience wrapper around :func:`learn` for use in notebooks,
    REPLs, and web APIs where writing a temporary file is inconvenient.

    Args:
        program: A complete XHAIL program as a string, including background
            knowledge, mode declarations (``#modeh`` / ``#modeb``), and
            examples (``#example``).
        depth: Maximum BFS depth for the deduction phase (default: 10).
        on_phase: Optional phase-completion callback — see :func:`learn`.

    Returns:
        A :class:`LearningResult` containing the learned rules.

    Example::

        from xhail import learn_from_string

        result = learn_from_string('''
            bird(a). bird(b). bird(c). penguin(d).
            bird(X) :- penguin(X).
            #modeh flies(+bird).
            #modeb not penguin(+bird).
            #example flies(a). #example flies(b). #example flies(c).
            #example not flies(d).
        ''')
        print(result)
        # flies(V1) :- not penguin(V1).

        print(result.to_json())
        # {"success": true, "n_rules": 1, ...}
    """
    return learn(program, depth=depth, on_phase=on_phase)
