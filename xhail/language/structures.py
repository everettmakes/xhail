import copy

from ..language.terms import Atom, Normal, PlaceMarker


# ---------- display ----------- #
class Display:
    """Represents a ``#display predicate/arity.`` directive.

    When one or more ``#display`` directives are present in the input, only
    hypothesis rules whose head predicate matches are included in the output.
    If no ``#display`` directives appear, all learned rules are shown (default).
    """

    KEY_WORD = "#display"

    def __init__(self, predicate: str, arity=None):
        self.predicate = predicate
        self.arity = arity  # None means "any arity"

    def matches(self, rule_str: str) -> bool:
        """Return True if *rule_str* starts with a matching head predicate."""
        head = rule_str.split(":-")[0].strip().rstrip(".")
        head_pred = head.split("(")[0].strip()
        if head_pred != self.predicate:
            return False
        if self.arity is None:
            return True
        lparen = head.find("(")
        if lparen == -1:
            return self.arity == 0
        inner = head[lparen + 1 : head.rfind(")")]
        depth = 0
        arity = 1
        for ch in inner:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                arity += 1
        return arity == self.arity

    def __str__(self) -> str:
        if self.arity is not None:
            return f"#display {self.predicate}/{self.arity}"
        return f"#display {self.predicate}"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def generalise_atom(atom: Atom, n: int = 1, replace_outputs: bool = True):
    """Replace PlaceMarker terms in *atom* with fresh variable Normals."""
    for idt, term in enumerate(atom.terms):
        if isinstance(term, PlaceMarker) and (replace_outputs or term.marker in ("+", "#")):
            new_normal = Normal(f"V{n}")
            new_normal.setType(term.type)
            atom.terms[idt] = new_normal
            n += 1
        elif isinstance(term, Atom):
            atom.terms[idt], n = generalise_atom(term, n, replace_outputs)
    return atom, n


# ----- STRUCTURE CLASS DEFINITIONS ------ #
# ---------- example ----------- #
class Example:
    KEY_WORD = "#example"
    WEIGHT_OPERATOR = "="
    PRIORITY_OPERATOR = "@"
    weight = 1
    priority = 1

    def __init__(self, atom: Atom, negation=False):
        self.atom = atom
        self.negation = negation

    def setWeight(self, weight):
        self.weight = weight

    def setPriority(self, priority):
        self.priority = priority

    @property
    def is_soft(self) -> bool:
        """True when an explicit weight was specified — example may be missed."""
        return self.weight != 1 or self.priority != 1

    def createProgram(self):
        program = []
        negation_string = "not " if self.negation else ""
        if self.is_soft:
            # Soft example: maximise coverage weighted by priority, do not
            # hard-require satisfaction.
            program.append(
                "#maximize{"
                + f"{str(self.weight)}@{str(self.priority)} : {negation_string}{self.atom}"
                + "}."
            )
        else:
            # Hard example: must be satisfied (original behaviour).
            program.append(
                "%#maximize{"
                + f"{str(self.weight)}@{str(self.priority)} : {negation_string}{self.atom}"
                + "}."
            )
            program.append(f":- {self.atom}." if self.negation else f":- not {self.atom}.")
        return "\n".join(program)

    def __str__(self):
        parts = ["#example", "not " if self.negation else "", str(self.atom)]
        if self.weight != 1:
            parts.append(f" ={self.weight}")
        if self.priority != 1:
            parts.append(f" @{self.priority}")
        return "".join(parts)


# ---------- modeh ----------- #
class Modeh:
    KEY_WORD = "#modeh"
    WEIGHT_OPERATOR = "="
    PRIORITY_OPERATOR = "@"
    CONSTRAINT_OPERATOR = ":"
    CONSTRAINT_SEPARATOR = "-"
    weight = 1
    priority = 2
    min = 0
    max = 1000000

    def __init__(self, atom: Atom, n: str):
        self.atom = atom
        self.n = n
        self.types = [term.type for term in atom.terms if isinstance(term, PlaceMarker)]

    def setWeight(self, weight):
        self.weight = weight

    def setPriority(self, priority):
        self.priority = priority

    def setMax(self, max):
        self.max = max

    def setMin(self, min):
        self.min = min

    def generalise(self, atom, n=1):
        return generalise_atom(atom, n, replace_outputs=False)

    def createProgram(self):
        new_atom = copy.deepcopy(self.atom)
        generalised_atom, n = self.generalise(new_atom)

        if not generalised_atom.terms:
            program = []
            program.append(
                f"{self.min} " + "{ abduced_" + str(generalised_atom) + " } " + f"{self.max}."
            )
            program.append(
                "#minimize{"
                + f"{str(self.weight)}@{str(self.priority)} : abduced_{generalised_atom}"
                + "}."
            )
            program.append(f"{generalised_atom} :- abduced_{generalised_atom}.")
            return "\n".join(program)

        types = ", ".join(generalised_atom.getTypes())
        variables = ", ".join([f"V{i}" for i in range(1, n)])

        program = []
        program.append(
            str(self.min)
            + " { abduced_"
            + str(generalised_atom)
            + " : "
            + types
            + " } "
            + str(self.max)
            + "."
        )
        program.append(
            "#minimize{"
            + f"{str(self.weight)}@{str(self.priority)}, {variables}: abduced_{generalised_atom}, {types}"
            + "}."
        )
        program.append(f"{generalised_atom} :- abduced_{generalised_atom}, {types}.")
        return "\n".join(program)

    def __str__(self):
        return "#modeh " + str(self.atom)


# ---------- modeb ----------- #
class Modeb:
    KEY_WORD = "#modeb"
    WEIGHT_OPERATOR = "="
    PRIORITY_OPERATOR = "@"
    CONSTRAINT_OPERATOR = ":"
    CONSTRAINT_SEPARATOR = "-"
    weight = 1
    priority = 1
    min = 0
    max = 1000000

    def __init__(self, atom: Atom, n: str, negation=False):
        self.atom = atom
        self.n = n
        self.negation = negation

    def setWeight(self, weight):
        self.weight = weight

    def setPriority(self, priority):
        self.priority = priority

    def setMax(self, max):
        self.max = max

    def setMin(self, min):
        self.min = min

    def generalise(self, atom, n=1):
        return generalise_atom(atom, n, replace_outputs=True)

    def createProgram(self):
        if self.negation:
            new_atom = copy.deepcopy(self.atom)
            generalised_atom, _ = self.generalise(new_atom)
            not_pred = "not_" + generalised_atom.predicate
            not_atom = Atom(not_pred, generalised_atom.terms)
            if not generalised_atom.terms:
                program = f"{not_atom} :- not {generalised_atom}."
            else:
                types = ", ".join(generalised_atom.getTypes())
                program = f"{not_atom} :- not {generalised_atom}, {types}."
        else:
            program = ""

        return program

    def __str__(self):
        neg = "not " if self.negation else ""
        return f"#modeb {neg}{self.atom}"
