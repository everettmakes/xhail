"""
Tests for features added in v0.1.x:
  - Weighted / prioritised examples  (#example atom =weight @priority.)
  - Mode annotations                 (#modeh schema :min-max =weight @priority.)
  - #display directive               (#display predicate/arity.)
  - --all / all_solutions            (enumerate all optimal hypotheses)
  - Alpha-normalised deduplication   (V1,V2 vs V2,V1 are the same hypothesis)
  - Timeout (threading.Timer)        (--kill / timeout param)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xhail import learn, learn_from_string
from xhail.language.structures import Display
from xhail.parser.parser import ParseError, Parser

BENCHMARKS = Path(__file__).parent.parent / "experiments" / "benchmarks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(program: str):
    p = Parser()
    p.loadString(program)
    p.parseProgram()
    return p.separate()  # EX, MH, MB, BG, DISP


# ---------------------------------------------------------------------------
# Weighted / prioritised examples
# ---------------------------------------------------------------------------

class TestWeightedExamples:
    def test_default_weight_and_priority(self):
        EX, *_ = parse("#example flies(a).")
        assert EX[0].weight == 1
        assert EX[0].priority == 1

    def test_weight_only(self):
        EX, *_ = parse("#example flies(a) =5.")
        assert EX[0].weight == 5
        assert EX[0].priority == 1

    def test_priority_only(self):
        EX, *_ = parse("#example flies(a) @3.")
        assert EX[0].weight == 1
        assert EX[0].priority == 3

    def test_weight_and_priority(self):
        EX, *_ = parse("#example flies(a) =5 @2.")
        assert EX[0].weight == 5
        assert EX[0].priority == 2

    def test_negative_example_with_weight(self):
        EX, *_ = parse("#example not flies(d) =3 @1.")
        assert EX[0].negation is True
        assert EX[0].weight == 3

    def test_is_soft_with_weight(self):
        EX, *_ = parse("#example flies(a) =5.")
        assert EX[0].is_soft is True

    def test_is_soft_default_is_false(self):
        EX, *_ = parse("#example flies(a).")
        assert EX[0].is_soft is False

    def test_multiple_examples_mixed(self):
        prog = "#example flies(a) =5 @2.\n#example not flies(d).\n#example flies(b) @1."
        EX, *_ = parse(prog)
        assert len(EX) == 3
        assert EX[0].weight == 5
        assert EX[0].priority == 2
        assert EX[1].weight == 1
        assert EX[2].priority == 1

    def test_non_integer_weight_raises(self):
        with pytest.raises(ParseError):
            parse("#example flies(a) =bad.")


# ---------------------------------------------------------------------------
# Mode annotations
# ---------------------------------------------------------------------------

class TestModeAnnotations:
    def test_modeh_default_annotation(self):
        _, MH, *_ = parse("#modeh flies(+bird).")
        assert MH[0].min == 0
        assert MH[0].max == 1000000
        assert MH[0].weight == 1

    def test_modeh_min_only(self):
        _, MH, *_ = parse("#modeh flies(+bird) :1.")
        assert MH[0].min == 1
        assert MH[0].max == 1

    def test_modeh_range(self):
        _, MH, *_ = parse("#modeh flies(+bird) :1-3.")
        assert MH[0].min == 1
        assert MH[0].max == 3

    def test_modeh_weight_only(self):
        _, MH, *_ = parse("#modeh flies(+bird) =2.")
        assert MH[0].weight == 2

    def test_modeh_priority_only(self):
        _, MH, *_ = parse("#modeh flies(+bird) @3.")
        assert MH[0].priority == 3

    def test_modeh_full_annotation(self):
        _, MH, *_ = parse("#modeh flies(+bird) :1-5 =2 @3.")
        assert MH[0].min == 1
        assert MH[0].max == 5
        assert MH[0].weight == 2
        assert MH[0].priority == 3

    def test_modeb_neg_with_annotation(self):
        _, _, MB, *_ = parse("#modeb not penguin(+bird) =1 @2.")
        assert MB[0].negation is True
        assert MB[0].weight == 1
        assert MB[0].priority == 2

    def test_multiple_mode_declarations(self):
        prog = "#modeh flies(+bird) :1.\n#modeb penguin(+bird) =2."
        _, MH, MB, *_ = parse(prog)
        assert MH[0].min == 1
        assert MB[0].weight == 2


# ---------------------------------------------------------------------------
# #display directive — parsing
# ---------------------------------------------------------------------------

class TestDisplayParsing:
    def test_display_with_arity(self):
        *_, DISP = parse("#display flies/1.")
        assert len(DISP) == 1
        assert DISP[0].predicate == "flies"
        assert DISP[0].arity == 1

    def test_display_without_arity(self):
        *_, DISP = parse("#display flies.")
        assert DISP[0].predicate == "flies"
        assert DISP[0].arity is None

    def test_multiple_display_directives(self):
        *_, DISP = parse("#display flies/1.\n#display mammal/1.")
        assert len(DISP) == 2
        assert {d.predicate for d in DISP} == {"flies", "mammal"}

    def test_no_display_returns_empty(self):
        *_, DISP = parse("bird(a). #modeh flies(+bird). #example flies(a).")
        assert DISP == []


class TestDisplayMatches:
    def test_matches_by_predicate(self):
        assert Display("flies", 1).matches("flies(V1) :- not penguin(V1).")

    def test_no_match_wrong_predicate(self):
        assert not Display("mammal", 1).matches("flies(V1) :- not penguin(V1).")

    def test_matches_no_arity_any(self):
        d = Display("flies")
        assert d.matches("flies(V1) :- not penguin(V1).")
        assert d.matches("flies(V1, V2) :- b(V1, V2).")

    def test_matches_propositional(self):
        assert Display("output").matches("output.")


@pytest.mark.integration
class TestDisplayFiltering:
    def test_display_filters_hypothesis(self):
        prog = """
        bird(a). bird(b). penguin(c). bird(X) :- penguin(X).
        #modeh flies(+bird).
        #modeb not penguin(+bird).
        #example flies(a). #example not flies(c).
        #display flies/1.
        """
        result = learn_from_string(prog)
        assert result.success
        assert all("flies" in r for r in result.hypothesis)

    def test_no_display_returns_all(self):
        prog = """
        bird(a). penguin(b). bird(X) :- penguin(X).
        #modeh flies(+bird).
        #modeb not penguin(+bird).
        #example flies(a). #example not flies(b).
        """
        result = learn_from_string(prog)
        assert result.success
        assert len(result.hypothesis) >= 1

    def test_display_wrong_predicate_filters_all(self):
        prog = """
        bird(a). penguin(b). bird(X) :- penguin(X).
        #modeh flies(+bird).
        #modeb not penguin(+bird).
        #example flies(a). #example not flies(b).
        #display mammal/1.
        """
        result = learn_from_string(prog)
        assert result.hypothesis == []


# ---------------------------------------------------------------------------
# --all / all_solutions
# ---------------------------------------------------------------------------

TWO_ANSWER_PROG = """
red(a). round(a).
red(b). round(b).
blue(c). square(c).
object(a). object(b). object(c).
#modeh target(+object).
#modeb red(+object).
#modeb round(+object).
#example target(a).
#example target(b).
#example not target(c).
"""


@pytest.mark.integration
class TestAllSolutions:
    def test_single_solution_without_flag(self):
        result = learn_from_string(TWO_ANSWER_PROG)
        assert result.success

    def test_all_solutions_returns_multiple(self):
        result = learn_from_string(TWO_ANSWER_PROG, all_solutions=True)
        assert result.success
        assert len(result.all_hypotheses) >= 2

    def test_hypothesis_is_first_of_all(self):
        result = learn_from_string(TWO_ANSWER_PROG, all_solutions=True)
        assert result.hypothesis == result.all_hypotheses[0]

    def test_all_hypotheses_are_distinct(self):
        result = learn_from_string(TWO_ANSWER_PROG, all_solutions=True)
        keys = [frozenset(r for r in h) for h in result.all_hypotheses]
        assert len(keys) == len(set(keys))

    def test_all_hypotheses_same_rule_count(self):
        result = learn_from_string(TWO_ANSWER_PROG, all_solutions=True)
        if len(result.all_hypotheses) > 1:
            lengths = [len(h) for h in result.all_hypotheses]
            assert len(set(lengths)) == 1

    def test_all_solutions_unique_program_returns_one(self):
        result = learn_from_string("""
            bird(a). bird(b). penguin(c). bird(X) :- penguin(X).
            #modeh flies(+bird).
            #modeb not penguin(+bird).
            #example flies(a). #example flies(b). #example not flies(c).
        """, all_solutions=True)
        assert result.success
        assert len(result.all_hypotheses) == 1

    def test_all_hypotheses_populated_without_flag(self):
        result = learn_from_string("""
            bird(a). penguin(b). bird(X) :- penguin(X).
            #modeh flies(+bird).
            #modeb not penguin(+bird).
            #example flies(a). #example not flies(b).
        """)
        assert len(result.all_hypotheses) == 1


# ---------------------------------------------------------------------------
# Alpha-normalised deduplication
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAlphaNormalisation:
    def test_grandfather_no_duplicate_hypotheses(self):
        """grandfather has 2 genuinely different optimal hypotheses
        (via father vs via parent+parent) — deduplication must not collapse
        them, but must not produce alpha-equivalent duplicates either."""
        result = learn(BENCHMARKS / "grandfather.lp", all_solutions=True)
        assert result.success
        # All returned hypotheses must be distinct by content
        seen = set()
        for h in result.all_hypotheses:
            key = frozenset(frozenset(r.split()) for r in h)
            assert key not in seen, "Duplicate hypothesis found"
            seen.add(key)

    def test_sugar_no_duplicate_hypotheses(self):
        """sugar body-reordering should not produce duplicate hypotheses."""
        result = learn(BENCHMARKS / "sugar.lp", all_solutions=True)
        assert result.success
        seen = set()
        for h in result.all_hypotheses:
            key = frozenset(frozenset(r.split()) for r in h)
            assert key not in seen, "Duplicate hypothesis found"
            seen.add(key)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTimeout:
    def test_timeout_does_not_crash(self):
        result = learn_from_string("""
            bird(a). penguin(b). bird(X) :- penguin(X).
            #modeh flies(+bird).
            #modeb not penguin(+bird).
            #example flies(a). #example not flies(b).
        """, timeout=30)
        assert result.success

    def test_very_short_timeout_returns_gracefully(self):
        result = learn_from_string("""
            bird(a). penguin(b). bird(X) :- penguin(X).
            #modeh flies(+bird).
            #modeb not penguin(+bird).
            #example flies(a). #example not flies(b).
        """, timeout=1)
        assert isinstance(result.success, bool)

    def test_all_solutions_with_timeout(self):
        result = learn_from_string(TWO_ANSWER_PROG, all_solutions=True, timeout=30)
        assert result.success
