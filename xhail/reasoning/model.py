import logging
import os
from pathlib import Path
from typing import Optional

import clingo

from ..language.terms import Atom, Fact, Normal, PlaceMarker

logger = logging.getLogger(__name__)


class Model:
    def __init__(
        self,
        EX: list,
        MH: list,
        MB: list,
        BG: list,
        DEPTH: int,
        debug_output_dir: Optional[Path] = None,
    ):
        # D10 fix: these were class attributes, meaning state was shared across all Model instances
        # in the same process. They are now instance attributes.
        self.program = ""
        self.clingo_models: list = []
        self.best_model = None
        self.delta: list = []
        self.kernel: list = []  # clauses built during deduction
        self.hypothesis: list = []  # final learned clauses set by induction
        self.all_hypotheses: list = []  # all optimal hypotheses (--all mode)
        self._subsumption_cache: dict = {}  # (str(atom), str(mode)) -> bool

        self.EX = EX
        self.MH = MH
        self.MB = MB
        self.BG = BG
        self.DEPTH = DEPTH
        # If set, abduction and induction write intermediate ASP programs here.
        self.debug_output_dir: Optional[Path] = debug_output_dir
        # Pre-computed type membership: maps type predicate name →
        # frozenset of ground constant strings belonging to that type.
        # Populated lazily after the first call() completes.
        self._type_members: Optional[dict] = None

    # ---------- METHODS ---------- #
    def call(self):
        """Run the abduction ASP program and collect all stable models.

        Uses ``--parallel-mode=N`` (same heuristic as ``getBestModel``) so the
        clingo thread-pool is engaged even for the satisfiability solve.
        """
        n_threads = min(4, os.cpu_count() or 1)
        control = clingo.Control([f"--parallel-mode={n_threads}", "--warn=no-atom-undefined"])
        control.add("base", [], self.program)
        control.ground([("base", [])])
        clingo_models = []

        def on_model(clingo_model):
            clingo_model = clingo_model.symbols(shown=True)
            clingo_models.append(clingo_model)

        control.solve(on_model=on_model)
        self.clingo_models = clingo_models
        # Invalidate the type-membership cache so it is rebuilt from the new model.
        self._type_members = None
        self._subsumption_cache = {}
        return clingo_models

    def getBestModel(self):
        n_threads = min(4, os.cpu_count() or 1)
        control = clingo.Control([f"--parallel-mode={n_threads}", "--warn=no-atom-undefined"])
        control.add("base", [], self.program)
        control.ground([("base", [])])
        clingo_models = []

        def on_model(clingo_model):
            model_symbols = clingo_model.symbols(shown=True)
            model_cost = clingo_model.cost
            logger.debug("clingo model: %s  cost: %s", model_symbols, model_cost)
            clingo_models.append([model_symbols, model_cost])

        control.solve(on_model=on_model)
        # result = handle.get()  # Wait for the solving process to finish
        # if not clingo_models:
        #    return None
        if clingo_models == []:
            return "[]"
        # Select the best model based on lexicographical order of the cost vector
        best_model = min(clingo_models, key=lambda m: [int(c) for c in m[1]])
        self.best_model = best_model
        return best_model[0]

    def writeProgram(self, destination):
        file = open(destination, "w")
        file.write(self.program)
        file.close()

    def clearProgram(self):
        self.program = ""

    # ensures normal values are the same, and any placeholders can be different.
    def isSubsumed(self, atom, mode) -> bool:
        cache_key = (str(atom), str(mode))
        if cache_key in self._subsumption_cache:
            return self._subsumption_cache[cache_key]
        result = self._isSubsumed(atom, mode)
        self._subsumption_cache[cache_key] = result
        return result

    def _isSubsumed(self, atom, mode) -> bool:
        if atom.predicate != mode.predicate:
            return False
        for term1, term2 in zip(atom.terms, mode.terms):
            if isinstance(term2, Atom):
                if not self.isSubsumed(term1, term2):
                    return False
            elif isinstance(term2, Normal):
                if term1.value != term2.value:
                    return False
            elif isinstance(term2, PlaceMarker) and isinstance(term1, Normal):
                # Fast path: look up pre-computed type membership instead of
                # calling getMatches (which re-parses the entire model every time).
                if term1.value not in self._type_members_for(term2.type):
                    return False
                else:
                    continue
            else:
                continue
        return True

    # ---------- type membership (fast path for #-placemarker checks) ---------- #

    def _build_type_members(self) -> dict:
        """Build a mapping ``type_name → frozenset[str]`` from the current model.

        Unary predicates (arity-1 facts whose argument is a ground constant)
        in the stable model are treated as type declarations, e.g.
        ``person(alice)`` → ``"person"`` maps ``"alice"``.  This is called
        once per ``call()`` and the result is cached in ``self._type_members``.
        """
        models = self.getClingoModels()
        if not models:
            return {}
        facts = self.parseModel(models[0])
        membership: dict = {}
        for fact in facts:
            atom = fact.head
            if len(atom.terms) == 1:
                t = atom.terms[0]
                const_str = t.value if isinstance(t, Normal) else str(t)
                membership.setdefault(atom.predicate, set()).add(const_str)
        return {k: frozenset(v) for k, v in membership.items()}

    def _type_members_for(self, type_name: str) -> frozenset:
        """Return the cached set of constants for ``type_name``."""
        if self._type_members is None:
            self._type_members = self._build_type_members()
        return self._type_members.get(type_name, frozenset())

    # ---------- clingo Symbol → XHAIL term/fact conversion ---------- #

    def _symbol_to_term(self, symbol) -> "Atom | Normal":
        """Convert a ground clingo Symbol to an XHAIL term (Atom or Normal)."""
        if symbol.type == clingo.SymbolType.Number:
            return Normal(str(symbol.number))
        if symbol.arguments:  # nested functor, e.g. bird(tweety)
            return self._symbol_to_atom(symbol)
        return Normal(symbol.name)  # ground constant, e.g. tweety

    def _symbol_to_atom(self, symbol) -> Atom:
        """Convert a functor clingo Symbol to an XHAIL Atom."""
        return Atom(symbol.name, [self._symbol_to_term(a) for a in symbol.arguments])

    def parseModel(self, model) -> list:
        """Convert a list of ground clingo Symbols to XHAIL Fact objects.

        Replaces the previous implementation which converted symbols to strings
        and re-parsed them with PLY — an unnecessary round-trip that dominated
        runtime on any non-trivial input.
        """
        return [Fact(self._symbol_to_atom(sym)) for sym in model]

    # ---------- GETTERS ---------- #

    def getProgram(self):
        return self.program

    def getClingoModels(self):
        return self.clingo_models

    def getDelta(self):
        head_atoms = [mh.atom for mh in self.MH]
        self.delta = self.getMatches(head_atoms)
        return self.delta

    def getMatches(self, atomConditions):
        # D7 fix: was getClingoModels()[0] which raises IndexError on UNSAT (no models).
        models = self.getClingoModels()
        if not models:
            raise RuntimeError(
                "Abduction yielded no model (program is UNSAT): "
                "no minimal explanation exists for the given examples and background. "
                "Check that the modeh declarations cover the positive examples."
            )
        model = models[0]
        facts = self.parseModel(model)

        result = []
        for fact in facts:
            for mh in atomConditions:
                if self.isSubsumed(fact.head, mh):
                    result.append(fact.head)
        return result

    def getKernel(self):
        return self.kernel

    # ---------- SETTERS ---------- #

    def setKernel(self, kernel):
        self.kernel = kernel

    def setHypothesis(self, clauses: list) -> None:
        """Store the final learned hypothesis (set by the induction phase)."""
        self.hypothesis = clauses

    def getHypothesis(self) -> list:
        """Return the learned hypothesis clauses."""
        return self.hypothesis

    def setProgram(self, program):
        self.program = program

    def getExamples(self):
        return self.EX

    def getBackground(self):
        return self.BG

    def getAllOptimalModels(self, timeout: int | None = None):
        """Return all models with the minimum cost (for --all mode).

        Uses clingo's ``--opt-mode=optN`` to enumerate every optimal model.
        Falls back to collecting all models and filtering by minimum cost if
        clingo returns no results under optN.

        Args:
            timeout: If set, pass ``--time-limit=N`` to clingo (seconds).

        Returns:
            A list of model symbol lists, each representing one optimal
            hypothesis.  Empty list if no model exists.
        """
        n_threads = min(4, os.cpu_count() or 1)
        opts = [
            f"--parallel-mode={n_threads}",
            "--warn=no-atom-undefined",
            "--opt-mode=optN",
        ]
        if timeout is not None:
            opts.append(f"--time-limit={timeout}")

        control = clingo.Control(opts)
        control.add("base", [], self.program)
        control.ground([("base", [])])

        all_models: list = []

        def on_model(clingo_model):
            model_symbols = clingo_model.symbols(shown=True)
            model_cost = list(clingo_model.cost)
            all_models.append((model_symbols, model_cost))

        control.solve(on_model=on_model)

        if not all_models:
            return []

        # Find minimum cost vector (lexicographic)
        min_cost = min(m[1] for m in all_models)
        return [m[0] for m in all_models if m[1] == min_cost]

    def getBestModelWithTimeout(self, timeout: int | None = None):
        """Like getBestModel but supports a timeout (seconds)."""
        n_threads = min(4, os.cpu_count() or 1)
        opts = [f"--parallel-mode={n_threads}", "--warn=no-atom-undefined"]
        if timeout is not None:
            opts.append(f"--time-limit={timeout}")

        control = clingo.Control(opts)
        control.add("base", [], self.program)
        control.ground([("base", [])])
        clingo_models = []

        def on_model(clingo_model):
            model_symbols = clingo_model.symbols(shown=True)
            model_cost = clingo_model.cost
            clingo_models.append([model_symbols, model_cost])

        control.solve(on_model=on_model)

        if not clingo_models:
            return "[]"
        best = min(clingo_models, key=lambda m: [int(c) for c in m[1]])
        self.best_model = best
        return best[0]

    def setAllHypotheses(self, hypotheses: list) -> None:
        """Store all optimal hypotheses (populated when all_solutions=True)."""
        self.all_hypotheses = hypotheses

    def getAllHypotheses(self) -> list:
        """Return all optimal hypotheses (non-empty only when all_solutions=True)."""
        return self.all_hypotheses
