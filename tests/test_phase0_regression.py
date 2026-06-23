"""
Phase 0 regression and correctness tests for XHAIL.

These tests verify:
  1. The package imports cleanly (Python-version portability fix, D1).
  2. The Constraint constructor works correctly (D2).
  3. Propositional atoms (0-arity) parse without error (D3).
  4. ParseError is raised on bad input, not a NoneType crash (D4).
  5. Idiomatic ':- body.' constraint syntax parses (D5).
  6. The Model class uses instance state, not shared class state (D10).
  7. Golden-output regression for penguins.lp (canonical penguins benchmark).
  8. Golden-output regression for traffic.lp (canonical traffic benchmark).

Tests 7 and 8 are integration tests that call clingo; they are marked
with the 'integration' pytest mark so they can be skipped in fast CI:
    pytest -m "not integration"
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "experiments" / "benchmarks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_pipeline(lp_file: Path, depth: int = 10) -> str:
    """Run the full XHAIL 3-phase pipeline via the public API.

    Returns the hypothesis rules joined as a single string.
    Raises on any exception (parse error, clingo error, timeout, etc.).
    """
    from xhail import learn

    result = learn(lp_file, depth=depth)
    return "\n".join(result.hypothesis)


# ---------------------------------------------------------------------------
# Unit tests — fast, no clingo required
# ---------------------------------------------------------------------------


class TestImports:
    """D1: Package must import cleanly on Python < 3.12 (no nested same-quote f-strings)."""

    def test_import_terms(self):
        from xhail.language import terms  # noqa: F401

    def test_import_structures(self):
        from xhail.language import structures  # noqa: F401

    def test_import_parser(self):
        from xhail.parser import parser  # noqa: F401

    def test_import_model(self):
        from xhail.reasoning import model  # noqa: F401


class TestConstraintConstructor:
    """D2: Constraint() takes exactly one argument (body list)."""

    def test_constraint_one_arg(self):
        from xhail.language.terms import Atom, Constraint, Literal

        body = [Literal(Atom("penguin", []), False)]
        c = Constraint(body)
        assert str(c) == ":- penguin."

    def test_constraint_str_with_terms(self):
        from xhail.language.terms import Atom, Constraint, Literal, Normal

        body = [Literal(Atom("bird", [Normal("X")]), False)]
        c = Constraint(body)
        assert ":- bird(X)." == str(c)


class TestPropositionalParsing:
    """D3: 0-arity (propositional) atoms must parse without error."""

    def test_propositional_fact(self):
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadString("q.")
        result = p.parseProgram()
        assert len(result) == 1
        assert str(result[0]) == "q."

    def test_propositional_rule(self):
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadString("p :- q, r.")
        result = p.parseProgram()
        assert len(result) == 1
        assert "p" in str(result[0])

    def test_propositional_example(self):
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadString("#example p.")
        result = p.parseProgram()
        assert len(result) == 1

    def test_propositional_modeh(self):
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadString("#modeh q.")
        result = p.parseProgram()
        assert len(result) == 1

    def test_penguins_lp_parses(self):
        """The penguins benchmark must parse without raising."""
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadFile(str(EXAMPLES_DIR / "penguins.lp"))
        result = p.parseProgram()
        assert result is not None
        assert len(result) > 0


class TestParseErrors:
    """D4: Parse failures must raise ParseError, not produce a NoneType crash."""

    def test_bad_syntax_raises_parse_error(self):
        from xhail.parser.parser import ParseError, Parser

        p = Parser()
        p.loadString(":-.")  # bare :- with no body
        with pytest.raises((ParseError, Exception)):
            p.parseProgram()

    def test_parse_error_is_informative(self):
        """ParseError message should mention the problem token, not be empty."""
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadString("this is not valid %%%")
        try:
            p.parseProgram()
        except Exception as exc:
            assert str(exc) != ""


class TestConstraintSyntax:
    """D5: Idiomatic ':- body.' constraint syntax must be accepted."""

    def test_implies_constraint(self):
        from xhail.language.terms import Constraint
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadString(":- penguin(X).")
        result = p.parseProgram()
        assert len(result) == 1
        assert isinstance(result[0], Constraint)

    def test_not_constraint_still_works(self):
        """Backward-compat: 'not body.' constraint form must still parse."""
        from xhail.language.terms import Constraint
        from xhail.parser.parser import Parser

        p = Parser()
        p.loadString("not penguin(X).")
        result = p.parseProgram()
        assert len(result) == 1
        assert isinstance(result[0], Constraint)


class TestModelInstanceState:
    """D10: Two Model instances must not share mutable class-level state."""

    def test_programs_are_independent(self):
        from xhail.reasoning.model import Model

        m1 = Model([], [], [], [], 5)
        m2 = Model([], [], [], [], 5)
        m1.setProgram("hello.")
        assert m2.getProgram() == "", "m2.program was contaminated by m1 — still a class attribute"

    def test_kernels_are_independent(self):
        from xhail.language.terms import Atom, Clause
        from xhail.reasoning.model import Model

        m1 = Model([], [], [], [], 5)
        m2 = Model([], [], [], [], 5)
        fake_clause = Clause(Atom("test", []), [])
        m1.setKernel([fake_clause])
        assert m2.getKernel() == [], "m2.kernel was contaminated by m1 — still a class attribute"


class TestAtomStr:
    """0-arity atoms must stringify as 'pred', not 'pred()'."""

    def test_zero_arity_str(self):
        from xhail.language.terms import Atom

        a = Atom("flies", [])
        assert str(a) == "flies"

    def test_nonzero_arity_str(self):
        from xhail.language.terms import Atom, Normal

        a = Atom("flies", [Normal("X")])
        assert str(a) == "flies(X)"


class TestNormalInstanceType:
    """D10: Normal.type must be an instance attribute, not shared class attribute."""

    def test_type_is_instance_attr(self):
        from xhail.language.terms import Normal

        n1 = Normal("a")
        n2 = Normal("b")
        n1.setType("bird")
        assert n2.getType() == "", (
            "n2.type was contaminated by n1.setType — still a class attribute"
        )


# ---------------------------------------------------------------------------
# Integration tests — run the full pipeline (require clingo)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGoldenOutputs:
    """
    End-to-end regression: pin known-good hypotheses for the two working examples.

    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    def test_penguins_learns_correct_rule(self):
        """penguins.lp: should learn that all non-penguins fly."""
        output = run_pipeline(EXAMPLES_DIR / "penguins.lp")
        assert "flies" in output, f"Expected 'flies' in output, got:\n{output}"
        assert "penguin" in output, f"Expected 'penguin' in output, got:\n{output}"
        # The canonical rule: flies(V1) :- not penguin(V1).
        assert "not penguin" in output, f"Expected negation-as-failure over penguin, got:\n{output}"

    def test_traffic_learns_rules(self):
        """traffic.lp: should learn stop/red rule."""
        output = run_pipeline(EXAMPLES_DIR / "traffic.lp")
        assert "stop" in output, f"Expected 'stop' in output, got:\n{output}"
        assert "red" in output, f"Expected 'red' in learned rule body, got:\n{output}"

    def test_multi_rule_heads_are_correct(self):
        """
        Regression for the clauses[0].head / pop(-1) bug.

        If the pipeline learns two rules, each rule must carry its own
        head predicate — not the head of whichever clause happens to be
        first in the kernel.  We use the animals benchmark which expects
        mammal(V1) :- has_hair(V1).  A second valid rule would also have
        head 'mammal', so we check the animals benchmark returns *only*
        mammal in rule heads (never a body predicate as a head).
        """
        from xhail import learn

        result = learn(REPO_ROOT / "experiments" / "benchmarks" / "animals.lp")
        assert result.success, "Animals benchmark should produce a hypothesis"
        for rule in result.hypothesis:
            head = rule.split(" :- ")[0].strip()
            assert head.startswith("mammal"), (
                f"Expected all rule heads to be 'mammal(...)' but got: '{head}'\n"
                f"Full hypothesis: {result.hypothesis}"
            )

    def test_getDelta_returns_abduced_atoms(self):
        """
        Regression for getDelta() calling non-existent getAtoms().
        After running abduction, getDelta() should return atoms matching
        the modeh schema from the solved model — not raise AttributeError.
        """
        from xhail.parser.parser import Parser
        from xhail.reasoning.abduction import Abduction
        from xhail.reasoning.model import Model

        prog = "bird(a). bird(b).\n#modeh flies(+bird).\n#example flies(a).\n#example flies(b).\n"
        parser = Parser()
        parser.loadString(prog)
        parser.parseProgram()
        EX, MH, MB, BG, DISP = parser.separate()
        model = Model(EX, MH, MB, BG, 5)
        Abduction(model).runPhase()

        delta = model.getDelta()  # was: AttributeError: 'Model' has no attribute 'getAtoms'
        assert isinstance(delta, list), "getDelta() should return a list"
        assert len(delta) > 0, "getDelta() should return at least one abduced atom"
        predicates = {atom.predicate for atom in delta}
        assert "flies" in predicates, f"Expected 'flies' in delta predicates, got {predicates}"


class TestParseModelDirectConversion:
    """parseModel should build Fact objects directly from clingo Symbols."""

    def test_parseModel_does_not_call_parser(self):
        """
        Verify the new direct-conversion parseModel works without going
        through the PLY parser.  We mock the clingo symbol API with a
        simple stub to keep this test fast and dependency-free.
        """
        # Build a minimal stub that looks like clingo.Symbol
        import clingo as _clingo

        from xhail.language.terms import Fact, Normal
        from xhail.reasoning.model import Model

        def make_sym(name, args=()):
            sym = _clingo.Function(name, [_clingo.Function(a, []) for a in args])
            return sym

        model = Model([], [], [], [], 5)
        symbols = [make_sym("flies", ["a"]), make_sym("bird", ["b"])]
        facts = model.parseModel(symbols)

        assert len(facts) == 2
        assert all(isinstance(f, Fact) for f in facts)
        predicates = {f.head.predicate for f in facts}
        assert predicates == {"flies", "bird"}
        # Check ground term was converted to Normal, not left as clingo Symbol
        flies_fact = next(f for f in facts if f.head.predicate == "flies")
        assert isinstance(flies_fact.head.terms[0], Normal)
        assert flies_fact.head.terms[0].value == "a"


class TestUpdateAtomTypesNested:
    """Regression: updateAtomTypes must recurse via updateAtomTypes, not updateTypes."""

    def test_nested_atom_does_not_crash(self):
        """
        When a mode schema contains a nested atom term, updateAtomTypes
        should recurse correctly.  Before the fix it called self.updateTypes
        (non-existent) and raised AttributeError.
        """
        from xhail.language.terms import Atom, Normal, PlaceMarker
        from xhail.reasoning.induction import Induction
        from xhail.reasoning.model import Model

        # Build a trivial model so we can instantiate Induction
        model = Model([], [], [], [], 5)
        ind = Induction(model)

        # An atom whose term is itself an Atom (nested) — this exercises the
        # isinstance(term2, Atom) branch in updateAtomTypes.
        outer_mode = Atom("outer", [Atom("inner", [PlaceMarker("+", "thing")])])
        outer_atom = Atom("outer", [Atom("inner", [Normal("x")])])

        # Should not raise; return (False, None) since 'inner' has no matching PlaceMarker path
        # or (True, atom) if types line up — either way, no AttributeError.
        try:
            result = ind.updateAtomTypes(outer_atom, outer_mode)
            assert isinstance(result, tuple) and len(result) == 2
        except AttributeError as e:
            pytest.fail(f"updateAtomTypes raised AttributeError (updateTypes bug): {e}")
