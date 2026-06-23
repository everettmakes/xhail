import ply.lex as lex
import ply.yacc as yacc

from ..language.structures import Display, Example, Modeb, Modeh
from ..language.terms import (
    Atom,
    Clause,
    Constraint,
    Fact,
    Literal,
    MiscLiteral,
    Normal,
    PlaceMarker,
)


# ---------- exceptions ---------- #
class ParseError(Exception):
    """Raised when the XHAIL parser fails to parse input."""
    pass


# ---------- prepare tokens ----------- #
tokens = (
    "NOT",
    "EXAMPLE_KEY",
    "MODEB_KEY",
    "MODEH_KEY",
    "DISPLAY_KEY",
    "PREDICATE",
    "TERM",
    "LPAREN",
    "RPAREN",
    "COMMA",
    "IMPLIES",
    "DOT",
    "MARKER",
    "OPERATOR",
    "AT",
    "EQ_SINGLE",
    "COLON",
    "SLASH",
)

# ---------- define tokens ----------- #
t_NOT = r"(?<!\S)not(?!\S)"
t_PREDICATE = r"(?!not\b)([a-zA-Z_][a-zA-Z_0-9]*)(?=\()"
t_TERM = r"(?!not\b)[a-zA-Z_][a-zA-Z_0-9]*|[0-9]+"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_COMMA = r","
t_IMPLIES = r":-"
t_DOT = r"\."
t_MARKER = r"\+|\-|\$"
t_OPERATOR = r"(==|!=|<=|>=|<|>)"
t_AT = r"@"
t_EQ_SINGLE = r"="
t_COLON = r":"
t_SLASH = r"/"
t_ignore = " \t\n"


# ── Keyword tokens as functions (higher priority than string tokens) ────────
def t_DISPLAY_KEY(t):
    r"\#display"
    return t


def t_EXAMPLE_KEY(t):
    r"\#example"
    return t


def t_MODEH_KEY(t):
    r"\#modeh"
    return t


def t_MODEB_KEY(t):
    r"\#modeb"
    return t


def t_HASH_MARKER(t):
    r"\#(?=[a-zA-Z_])"
    t.type = "MARKER"
    t.value = "#"
    return t


def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)


def t_ignore_COMMENT(t):
    r"%.*"
    pass


_lex_master = lex.lex()


# ---------- program and clauses ---------- #
def p_program(p):
    """program : program clause
    | clause"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_clause(p):
    """clause : example
    | modeb
    | modeh
    | display
    | fact
    | normal_clause
    | constraint
    """
    p[0] = p[1]


def p_atom(p):
    """atom : PREDICATE LPAREN terms RPAREN"""
    p[0] = Atom(p[1], p[3])


def p_prop_atom(p):
    """prop_atom : TERM"""
    p[0] = Atom(p[1], [])


def p_schema(p):
    """schema : PREDICATE LPAREN schema_terms RPAREN"""
    p[0] = Atom(p[1], p[3])


def p_schema_terms(p):
    """schema_terms : MARKER TERM
    | MARKER TERM COMMA schema_terms
    | schema
    | schema COMMA schema_terms
    """
    if len(p) == 3:
        p[0] = [PlaceMarker(marker=p[1], type=p[2])]
    elif len(p) == 5:
        p[0] = [PlaceMarker(marker=p[1], type=p[2])] + p[4]
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


# ---------- example annotation: optional =weight @priority ---------- #
def p_ex_annotation_both(p):
    """ex_annotation : EQ_SINGLE TERM AT TERM"""
    p[0] = (_to_int(p[2], "example weight"), _to_int(p[4], "example priority"))


def p_ex_annotation_weight(p):
    """ex_annotation : EQ_SINGLE TERM"""
    p[0] = (_to_int(p[2], "example weight"), 1)


def p_ex_annotation_priority(p):
    """ex_annotation : AT TERM"""
    p[0] = (1, _to_int(p[2], "example priority"))


def p_ex_annotation_empty(p):
    """ex_annotation :"""
    p[0] = (1, 1)


# ---------- example ---------- #
def p_example_pos(p):
    """example : EXAMPLE_KEY atom ex_annotation DOT
    | EXAMPLE_KEY prop_atom ex_annotation DOT
    """
    ex = Example(p[2], negation=False)
    ex.setWeight(p[3][0])
    ex.setPriority(p[3][1])
    p[0] = ex


def p_example_neg(p):
    """example : EXAMPLE_KEY NOT atom ex_annotation DOT
    | EXAMPLE_KEY NOT prop_atom ex_annotation DOT
    """
    ex = Example(p[3], negation=True)
    ex.setWeight(p[4][0])
    ex.setPriority(p[4][1])
    p[0] = ex


# ---------- mode annotation: optional :min[-max] =weight @priority ---------- #
def p_mode_annotation_full(p):
    """mode_annotation : COLON TERM MARKER TERM EQ_SINGLE TERM AT TERM"""
    p[0] = (_to_int(p[2], "min"), _to_int(p[4], "max"), _to_int(p[6], "weight"), _to_int(p[8], "priority"))


def p_mode_annotation_range_weight(p):
    """mode_annotation : COLON TERM MARKER TERM EQ_SINGLE TERM"""
    p[0] = (_to_int(p[2], "min"), _to_int(p[4], "max"), _to_int(p[6], "weight"), 1)


def p_mode_annotation_range_priority(p):
    """mode_annotation : COLON TERM MARKER TERM AT TERM"""
    p[0] = (_to_int(p[2], "min"), _to_int(p[4], "max"), 1, _to_int(p[6], "priority"))


def p_mode_annotation_range(p):
    """mode_annotation : COLON TERM MARKER TERM"""
    p[0] = (_to_int(p[2], "min"), _to_int(p[4], "max"), 1, 1)


def p_mode_annotation_min_weight_priority(p):
    """mode_annotation : COLON TERM EQ_SINGLE TERM AT TERM"""
    p[0] = (_to_int(p[2], "min"), 1000000, _to_int(p[4], "weight"), _to_int(p[6], "priority"))


def p_mode_annotation_min_weight(p):
    """mode_annotation : COLON TERM EQ_SINGLE TERM"""
    p[0] = (_to_int(p[2], "min"), 1000000, _to_int(p[4], "weight"), 1)


def p_mode_annotation_min_priority(p):
    """mode_annotation : COLON TERM AT TERM"""
    p[0] = (_to_int(p[2], "min"), 1000000, 1, _to_int(p[4], "priority"))


def p_mode_annotation_min(p):
    """mode_annotation : COLON TERM"""
    val = _to_int(p[2], "min")
    p[0] = (val, val, 1, 1)


def p_mode_annotation_weight_priority(p):
    """mode_annotation : EQ_SINGLE TERM AT TERM"""
    p[0] = (0, 1000000, _to_int(p[2], "weight"), _to_int(p[4], "priority"))


def p_mode_annotation_weight(p):
    """mode_annotation : EQ_SINGLE TERM"""
    p[0] = (0, 1000000, _to_int(p[2], "weight"), 1)


def p_mode_annotation_priority(p):
    """mode_annotation : AT TERM"""
    p[0] = (0, 1000000, 1, _to_int(p[2], "priority"))


def p_mode_annotation_empty(p):
    """mode_annotation :"""
    p[0] = (0, 1000000, 1, 2)


# ---------- modeh ---------- #
def p_modeh(p):
    """modeh : MODEH_KEY schema mode_annotation DOT
    | MODEH_KEY prop_atom mode_annotation DOT
    """
    mh = Modeh(p[2], "*")
    mn, mx, w, pri = p[3]
    mh.setMin(mn)
    mh.setMax(mx)
    mh.setWeight(w)
    mh.setPriority(pri)
    p[0] = mh


# ---------- modeb ---------- #
def p_modeb_pos(p):
    """modeb : MODEB_KEY schema mode_annotation DOT
    | MODEB_KEY prop_atom mode_annotation DOT
    """
    mb = Modeb(p[2], "*", False)
    mn, mx, w, pri = p[3]
    mb.setMin(mn)
    mb.setMax(mx)
    mb.setWeight(w)
    mb.setPriority(pri)
    p[0] = mb


def p_modeb_neg(p):
    """modeb : MODEB_KEY NOT schema mode_annotation DOT
    | MODEB_KEY NOT prop_atom mode_annotation DOT
    """
    mb = Modeb(p[3], "*", True)
    mn, mx, w, pri = p[4]
    mb.setMin(mn)
    mb.setMax(mx)
    mb.setWeight(w)
    mb.setPriority(pri)
    p[0] = mb


# ---------- display ---------- #
def p_display_arity(p):
    """display : DISPLAY_KEY TERM SLASH TERM DOT"""
    p[0] = Display(p[2], _to_int(p[4], "arity"))


def p_display_no_arity(p):
    """display : DISPLAY_KEY TERM DOT"""
    p[0] = Display(p[2], None)


# ---------- terms ---------- #
def p_terms(p):
    """terms : TERM
    | atom
    | TERM COMMA terms
    | atom COMMA terms
    """
    if len(p) == 2 and not isinstance(p[1], Atom):
        p[0] = [Normal(p[1])]
    elif len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 4 and not isinstance(p[1], Atom):
        p[0] = [Normal(p[1])] + p[3]
    else:
        p[0] = [p[1]] + p[3]


def p_fact(p):
    """fact : atom DOT
    | prop_atom DOT
    """
    p[0] = Fact(p[1])


def p_constraint(p):
    """constraint : NOT body DOT
    | IMPLIES body DOT
    """
    p[0] = Constraint(p[2])


def p_normal_clause(p):
    """normal_clause : atom IMPLIES body DOT
    | prop_atom IMPLIES body DOT
    """
    p[0] = Clause(p[1], p[3])


def p_body(p):
    """body : literal COMMA body
    | literal
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[1] + p[3]


def p_literal(p):
    """literal : NOT atom
    | NOT prop_atom
    | atom
    | prop_atom
    | TERM OPERATOR TERM
    """
    if len(p) == 2:
        p[0] = [Literal(p[1], False)]
    elif len(p) == 3:
        p[0] = [Literal(p[2], True)]
    else:
        p[0] = [MiscLiteral(f"{p[1]}{p[2]}{p[3]}")]


# ---------- helpers ---------- #
def _to_int(value: str, label: str) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ParseError(f"Expected integer for {label}, got '{value}'")


def p_error(p):
    if p:
        raise ParseError(f"Syntax error at '{p.value}' on line {p.lineno}")
    else:
        raise ParseError("Syntax error at EOF")


_parser_cache = yacc.yacc(debug=False, write_tables=False)


class Parser:
    def __init__(self):
        self.data = ""
        self.parsedData = []

    def separate(self):
        examples = []
        modehs = []
        modebs = []
        background = []
        displays = []
        for item in self.parsedData:
            if isinstance(item, Example):
                examples.append(item)
            elif isinstance(item, Modeb):
                modebs.append(item)
            elif isinstance(item, Modeh):
                modehs.append(item)
            elif isinstance(item, Display):
                displays.append(item)
            elif isinstance(item, Clause):
                background.append(item)
        return examples, modehs, modebs, background, displays

    def parseProgram(self):
        _lexer = lex.lex()
        self.parsedData = _parser_cache.parse(self.data, lexer=_lexer)
        if self.parsedData is None:
            raise ParseError(
                "Failed to parse program: the parser returned no result. "
                "Check your input for syntax errors."
            )
        return self.parsedData

    def tokenByToken(self):
        _lexer = lex.lex()
        _lexer.input(self.data)
        for token in _lexer:
            print(f"Token type: {token.type}, Token value: {token.value}")

    def loadFile(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            self.data = f.read()
        return self.data

    def loadString(self, s):
        self.data = s
