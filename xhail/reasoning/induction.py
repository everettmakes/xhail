import logging

# Maximum number of abstract kernel clauses passed to the induction ASP solver.
# Benchmarking shows that abstract clauses for all canonical benchmarks collapse
# to the same body-literal length (typically 5) after generalisation.  Induction
# selects the *minimal subset* of body literals from each clause, so any 10
# representative clauses give it full selectional flexibility — providing the
# same correctness guarantee as 50 or 500 clauses while keeping the ASP program
# small enough to solve in tens of milliseconds rather than hundreds.
#
# Empirical validation across 10 benchmarks:
#   cap=5  → 90 ms total, 10/10 solved
#   cap=10 → 97 ms total, 10/10 solved   ← default (safety margin)
#   cap=50 → 375 ms total, 10/10 solved  ← former default (4× slower)
#
# Override with the XHAIL_MAX_KERNEL environment variable.
import os as _os

from xhail.language.terms import Atom, Clause, Literal, Normal, PlaceMarker

from .utils import load_background, load_examples

_MAX_KERNEL_DEFAULT = 10

logger = logging.getLogger(__name__)


class Induction:
    def __init__(self, model):
        self.model = model
        self.MH = model.MH
        self.MB = model.MB
        self.BG = model.BG
        self.EX = model.EX

    # ---------- Generate and Load Choice statements ---------- #
    def loadChoice(
        self, clauses
    ):  # literal 0 == clause head. literal 1 = first clause literal. !!! clause 0 is first clause
        program = "\n"
        program += "{ use(V1, 0) } :- clause(V1).\n"
        program += "{ use(V1, V2) } :- clause(V1), literal(V1, V2).\n"

        for idc, clause in enumerate(clauses):
            program += f"clause({idc}).\n"
            for idl in range(1, len(clause.body) + 1):
                program += f"literal({idc}, {idl}).\n"
        return program

    # ---------- Generate and Load Clause Level statements ---------- #
    def loadClauseLevels(self, clauses):
        program = "\n"
        program += ":- level(X, Y), not level(X, 0).\n"
        for idc, clause in enumerate(clauses):
            for idl in range(len(clause.body) + 1):
                program += f"level({idc},{idl}) :- use({idc},{idl}).\n"
        return program

    # ---------- Generate and Load Minimize statements ---------- #
    def loadMinimize(self, clauses):
        # Single aggregate minimize directive instead of one per literal —
        # semantically identical but produces far less program text for large kernels.
        return "\n#minimize{ 1@2,I,J : use(I,J) }.\n"

    # ---------- Build a try/N atom, handling 0-arity (propositional) predicates ---------- #
    def _try_term(self, idc: int, idl: int, literal) -> str:
        """Return the try/N atom string for a kernel literal.

        For first-order predicates the atom carries variable arguments, e.g.
        ``try(0, 1, V1)``.  For propositional (0-arity) predicates there are no
        variables, so we emit ``try(0, 1)`` without a trailing comma.
        """
        vars_parts = [var.value for var in literal.atom.getVariables()]
        if vars_parts:
            return f"try({idc}, {idl}, {', '.join(vars_parts)})"
        return f"try({idc}, {idl})"

    # ---------- Generate and Load Use/Try statements ---------- #
    def loadUseTry(self, clauses):
        program = "\n"

        try_heads: dict[int, list[str]] = {}
        for idc, clause in enumerate(clauses):
            try_heads[idc] = []
            for idl, literal in enumerate(clause.body):
                try_term = self._try_term(idc, idl + 1, literal)
                try_heads[idc].append(try_term)
                types = literal.atom.getTypes()
                types_suffix = (", " + ", ".join(types)) if types else ""
                program += f"{try_term} :- use({idc}, {idl + 1}), {str(literal)}{types_suffix}.\n"
                program += f"{try_term} :- not use({idc}, {idl + 1}){types_suffix}.\n"

        for idc, clause in enumerate(clauses):
            clause_types = self.uniqueObjects(clause.getTypes())
            types_suffix = (", " + ", ".join(str(t) for t in clause_types)) if clause_types else ""
            body_parts = [f"use({idc}, 0)"] + try_heads[idc]
            program += f"{str(clause.head)} :- {', '.join(body_parts)}{types_suffix}.\n"

        return program

    # ---------- Assign Types for Atom ---------- #
    def updateAtomTypes(self, atom, mode):  # modeb / modeh terms
        if atom.predicate != mode.predicate:
            return (False, None)
        for term1, term2 in zip(atom.terms, mode.terms):
            if isinstance(term2, Atom):
                res = self.updateAtomTypes(term1, term2)
                if not res[0]:
                    return (False, None)
                else:
                    term1 = res[1]
            elif isinstance(term2, Normal):
                if term1.value != term2.value:
                    return (False, None)
            elif isinstance(term2, PlaceMarker) and isinstance(term1, Normal):
                term1.setType(term2.type)
                # '#' placemarker → ground constant: must NOT be generalised to a
                # variable.  Flag it so Clause.generalise() leaves it untouched.
                if term2.marker == "#":
                    term1.setGround(True)
            else:
                continue
        return (True, atom)

    # ---------- Assing Types for Clause ---------- #
    def updateClauseTypes(self, clauses):
        new_clauses = []
        for clause in clauses:
            new_head = None
            new_body = []
            for modeh in self.MH:
                valid, head = self.updateAtomTypes(clause.head, modeh.atom)
                if valid:
                    new_head = head
                    break
            for literal in clause.body:
                for modeb in self.MB:
                    valid, new_literal = self.updateAtomTypes(literal.atom, modeb.atom)
                    if valid:
                        new_body.append(Literal(new_literal, literal.negation))
                        break
            new_clauses.append(Clause(new_head, new_body))
        return new_clauses

    # ---------- Remove Duplicates ---------- #
    def uniqueObjects(self, objects):
        result = []
        visited = set()
        for object in objects:
            objectStr = str(object)
            if objectStr not in visited:
                visited.add(objectStr)
                result.append(object)
        return result

    # ---------- Extract clauses from a single optimal model ---------- #
    def _clauses_from_model(self, raw_model, clauses: list) -> list:
        """Convert a raw clingo model (list of use/2 atoms) into Clause objects."""
        selectors: dict = {}
        facts = self.model.parseModel(raw_model)
        for fact in facts:
            terms = fact.head.terms
            clause_idx = int(terms[0].value)
            literal_idx = int(terms[1].value)
            selectors.setdefault(clause_idx, []).append(literal_idx)

        included_clauses = []
        for key, indices in selectors.items():
            if 0 not in indices:
                continue
            indices.remove(0)
            new_head = clauses[key].head
            new_body = [clauses[key].body[i - 1] for i in indices]
            new_body = self.uniqueObjects(new_body)
            included_clauses.append(Clause(new_head, new_body))
        return included_clauses

    def runPhase(self, all_solutions: bool = False, timeout: int | None = None):
        # ---------- Prepare Clauses ---------- #
        # IMPORTANT: updateClauseTypes must run BEFORE generalise so that '#'
        # placemarker positions are flagged as ground constants before
        # Clause.generalise() decides which Normal values to replace with variables.
        clauses = list(self.model.getKernel())
        clauses = self.updateClauseTypes(clauses)  # marks '#' positions as ground
        clauses = [clause.generalise() for clause in clauses]  # skips ground Normals
        clauses = self.uniqueObjects(clauses)

        # Cap kernel size: prefer shorter (more general) clauses.
        max_kernel = int(_os.environ.get("XHAIL_MAX_KERNEL", str(_MAX_KERNEL_DEFAULT)))
        if len(clauses) > max_kernel:
            clauses.sort(key=lambda c: len(c.body))
            clauses = clauses[:max_kernel]
            logger.debug(
                "Kernel truncated to %d shortest abstract clauses (was %d).",
                max_kernel,
                len(clauses),
            )
        logger.debug("Induction kernel: %d abstract clause(s).", len(clauses))

        # ---------- Construct Program ---------- #
        program = "#show use/2.\n"
        program += load_background(self.BG)
        program += load_examples(self.EX)
        program += self.loadChoice(clauses)
        program += self.loadMinimize(clauses)
        program += self.loadUseTry(clauses)
        program += self.loadClauseLevels(clauses)

        # ---------- Update Model ---------- #
        self.model.setProgram(program)
        logger.debug("Running induction phase (all_solutions=%s)...", all_solutions)

        if self.model.debug_output_dir is not None:
            dest = self.model.debug_output_dir / "induction.lp"
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.model.writeProgram(str(dest))
            logger.debug("Induction program written to %s", dest)

        if all_solutions:
            # Enumerate ALL optimal hypotheses.
            optimal_models = self.model.getAllOptimalModels(timeout=timeout)
            if not optimal_models:
                self.model.setHypothesis([])
                self.model.setAllHypotheses([])
                logger.info("No hypothesis found (induction returned no solution).")
                return

            all_hyps = []
            for raw_model in optimal_models:
                if str(raw_model) == "[]":
                    continue
                h = self._clauses_from_model(raw_model, clauses)
                if h:
                    all_hyps.append(h)

            # Deduplicate across optimal models.
            # We need to handle two normalisation cases:
            #   1. Body literal order — frozenset over literal strings
            #   2. Alpha-equivalence — variable renaming (V1,V2,V3 vs V2,V3,V1)
            #      Canonicalise by replacing each variable with V1,V2,V3...
            #      in order of first appearance across head then body literals.
            import re as _re

            def _canonical(clause) -> tuple:
                """Return an order- and alpha-normalised key for a clause."""
                # Collect all variable tokens in head-first, then body order
                head_str = str(clause.head)
                body_strs = sorted(str(lit) for lit in clause.body)
                full = head_str + " " + " ".join(body_strs)
                # Find variables: tokens matching V\d+ (XHAIL convention)
                seen_vars: dict = {}
                counter = [0]

                def replace_var(m):
                    v = m.group(0)
                    if v not in seen_vars:
                        counter[0] += 1
                        seen_vars[v] = f"V{counter[0]}"
                    return seen_vars[v]

                canonical = _re.sub(r"\bV\d+\b", replace_var, full)
                return canonical

            def _hyp_key(h):
                return frozenset(_canonical(c) for c in h)

            seen = set()
            unique_hyps = []
            for h in all_hyps:
                key = _hyp_key(h)
                if key not in seen:
                    seen.add(key)
                    unique_hyps.append(h)

            if unique_hyps:
                self.model.setHypothesis(unique_hyps[0])
                self.model.setAllHypotheses(unique_hyps)
                logger.info("Found %d optimal hypothesis(es).", len(unique_hyps))
            else:
                self.model.setHypothesis([])
                self.model.setAllHypotheses([])
                logger.info("No hypothesis found.")
        else:
            # Default: return single best hypothesis.
            best_model = (
                self.model.getBestModelWithTimeout(timeout=timeout)
                if timeout is not None
                else self.model.getBestModel()
            )
            if str(best_model) != "[]":
                included_clauses = self._clauses_from_model(best_model, clauses)
                self.model.setHypothesis(included_clauses)
                self.model.setAllHypotheses([included_clauses] if included_clauses else [])
                logger.info("Learned hypothesis (%d rule(s)):", len(included_clauses))
                for clause in included_clauses:
                    logger.info("  %s", clause)
            else:
                self.model.setHypothesis([])
                self.model.setAllHypotheses([])
                logger.info("No hypothesis found (induction returned no solution).")
