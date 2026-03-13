"""
agent/parser.py

parse_proof_from_output() — robust proof parser.

Handles all failure modes from design doc Section 8:
  - Missing PROOF header
  - Extra spaces, wrong capitalization
  - Invalid rule numbers
  - Truncated output
  - Malformed expressions
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional

from rewritelang import parse_expr, ParseError, Expr


def parse_proof_from_output(
    output: str,
    n_rules: int,
) -> List[Tuple[Optional[Expr], int]]:
    """
    Parse model output into list of (expression, rule_idx) pairs.

    rule_idx is 0-indexed (converts from 1-indexed DSL notation).
    expression is None if parsing failed for that step.

    Returns empty list if no PROOF block found.
    """
    # Strip think block if present
    think_end = output.find("</think>")
    if think_end != -1:
        output = output[think_end + 8:]

    # Find PROOF block
    proof_match = re.search(r"PROOF\s*\n(.*?)(?:\Z)", output, re.DOTALL | re.IGNORECASE)
    if not proof_match:
        # Fallback: try to find steps without PROOF header
        proof_text = output
    else:
        proof_text = proof_match.group(1)

    # Parse each step: S{n}: {expr} RULE {k}
    step_pattern = re.compile(
        r"S\s*\d+\s*:\s*(.+?)\s+RULE\s+(\d+)",
        re.IGNORECASE,
    )

    steps = []
    for m in step_pattern.finditer(proof_text):
        expr_str  = m.group(1).strip()
        rule_num  = int(m.group(2))

        # Convert to 0-indexed; mark invalid rule refs
        if rule_num < 1 or rule_num > n_rules:
            steps.append((None, -1))
            continue

        try:
            expr = parse_expr(expr_str)
            steps.append((expr, rule_num - 1))
        except (ParseError, Exception):
            steps.append((None, rule_num - 1))

    return steps


def extract_think_block(output: str) -> str:
    """Extract the content of <think>...</think> tags, or empty string."""
    m = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def count_think_tokens(output: str) -> int:
    """Approximate token count of think block (whitespace-split words as proxy)."""
    think = extract_think_block(output)
    return len(think.split()) if think else 0
