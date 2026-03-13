"""
rewritelang/grammar.py

Expr dataclass, parse_expr(), expr_to_str(), expr_equal().
This is the foundation. Everything else depends on this being correct.

Axiom 1 — Syntax:
  expr ::= CONST | VAR | SYM "(" EXPRLIST ")"
  CONST ::= uppercase identifier  {T, F, ZERO, ONE, A, B, ...}
  VAR   ::= single lowercase char {x, y, z, u, v, w}
  SYM   ::= lowercase identifier of length >= 2 {not, and, or, succ, f, g, ...}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


class ParseError(Exception):
    pass


@dataclass
class Expr:
    kind: str           # "const" | "var" | "fun"
    name: str           # symbol name
    children: List["Expr"] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expr):
            return False
        return (
            self.kind == other.kind
            and self.name == other.name
            and len(self.children) == len(other.children)
            and all(a == b for a, b in zip(self.children, other.children))
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.name, tuple(hash(c) for c in self.children)))

    def __repr__(self) -> str:
        return expr_to_str(self)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def variables(self) -> set:
        """Return set of variable names occurring in this expression."""
        if self.kind == "var":
            return {self.name}
        result = set()
        for c in self.children:
            result |= c.variables()
        return result

    def is_ground(self) -> bool:
        """True if the expression contains no variables."""
        return len(self.variables()) == 0

    def copy(self) -> "Expr":
        return Expr(self.kind, self.name, [c.copy() for c in self.children])


def _split_args(s: str) -> List[str]:
    """
    Split a comma-separated argument list, respecting nested parentheses.
    e.g. "not(T),or(F,x)" -> ["not(T)", "or(F,x)"]
    """
    args = []
    depth = 0
    current = []
    for ch in s:
        if ch == "(" :
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return [a for a in args if a]  # filter empty strings


def parse_expr(s: str) -> Expr:
    """
    Parse a RewriteLang expression string into an Expr tree.

    Rules:
      - If no '(' present: uppercase -> CONST, single lowercase -> VAR
      - Otherwise: SYM(...) -> FUN node
    """
    s = s.strip()
    if not s:
        raise ParseError("Empty expression string")

    paren_pos = s.find("(")

    if paren_pos == -1:
        # Atom: either CONST or VAR
        if s[0].isupper():
            # Constant — may be multi-char like ZERO, ONE, TRUE
            if not all(c.isalnum() or c == "_" for c in s):
                raise ParseError(f"Invalid constant name: {s!r}")
            return Expr("const", s, [])
        elif len(s) == 1 and s.islower():
            # Variable — single lowercase char only
            return Expr("var", s, [])
        else:
            raise ParseError(
                f"Ambiguous atom {s!r}: multi-char lowercase is a SYM (needs parens), "
                f"single uppercase is CONST, single lowercase is VAR"
            )
    else:
        # Function: SYM "(" EXPRLIST ")"
        sym = s[:paren_pos].strip()
        if not sym:
            raise ParseError(f"Empty symbol name in: {s!r}")
        if not (sym[0].islower() and all(c.isalnum() or c == "_" for c in sym)):
            raise ParseError(f"Invalid symbol name: {sym!r}")
        if s[-1] != ")":
            raise ParseError(f"Mismatched parentheses in: {s!r}")
        inner = s[paren_pos + 1 : -1]
        if not inner.strip():
            # Zero-arity function — treat as constant-like but keep as fun
            return Expr("fun", sym, [])
        children_strs = _split_args(inner)
        children = [parse_expr(c) for c in children_strs]
        return Expr("fun", sym, children)


def expr_to_str(e: Expr) -> str:
    """Convert an Expr tree back to a RewriteLang string."""
    if e.kind in ("const", "var"):
        return e.name
    # fun
    if not e.children:
        return f"{e.name}()"
    args = ", ".join(expr_to_str(c) for c in e.children)
    return f"{e.name}({args})"


def expr_equal(a: Expr, b: Expr) -> bool:
    """Structural equality (same as __eq__, exposed for clarity)."""
    return a == b
