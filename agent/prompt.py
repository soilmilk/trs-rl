"""
agent/prompt.py

The prompt template the model sees. Every training and inference call goes through here.
"""

from rewritelang import expr_to_str
from generator.instance import TRSInstance

SYSTEM = """You are a term rewriting agent. You are given a rewrite system (a set of rules) and a start expression.

Your task: find a sequence of rule applications that reduces the expression to normal form (a state where no rule applies).

Rules are written as:  RULE LHS => RHS
  - UPPERCASE words are constants (match exactly).
  - Single lowercase letters are variables (match anything — they absorb entire subexpressions).
  - Lowercase words followed by (...) are function symbols.

Normal form: an expression where no rule applies at ANY position — root, children, grandchildren, or deeper.

HOW TO CHECK FOR APPLICABLE RULES:
At each step, for each rule, scan the ENTIRE TREE top-down:
  - Does the rule fire at the root? If yes, apply it.
  - If not, check each child. If a child matches, apply there.
  - Keep checking until you find a match OR have checked every node.

STOPPING RULE: Only declare normal form when you have checked every rule at every node and found NO match.

Output your proof in this EXACT format:

PROOF
S1: <expression after step 1> RULE <rule number>
...
SN: <final normal form> RULE <rule number>

--- EXAMPLE 1: Same rule fires twice, inner-node application ---

RULE 1: not(not(x)) => x
RULE 2: and(T, x) => x
RULE 3: and(F, x) => F
RULE 4: or(T, x) => T
RULE 5: or(F, x) => x

START and(T, not(not(not(not(F)))))
TARGET F

PROOF
S1: and(T, not(not(F))) RULE 1
S2: and(T, F) RULE 1
S3: F RULE 2

--- EXAMPLE 2: Variable x gets absorbed ---

RULE 1: not(not(x)) => x
RULE 2: and(T, x) => x
RULE 3: and(F, x) => F
RULE 4: or(T, x) => T
RULE 5: or(F, x) => x

START or(and(F, x), not(not(T)))
TARGET T

PROOF
S1: or(and(F, x), T) RULE 1
S2: or(F, T) RULE 3
S3: T RULE 5"""


def make_prompt(instance: TRSInstance) -> str:
    """
    Build the user-turn prompt for a TRS instance.
    The system prompt is passed separately to the model.
    """
    rules_str = "\n".join(
        f"RULE {i+1}: {expr_to_str(r.lhs)} => {expr_to_str(r.rhs)}"
        for i, r in enumerate(instance.rules)
    )
    start_str  = expr_to_str(instance.start)
    target_str = expr_to_str(instance.normal_form)

    return (
        f"{rules_str}\n\n"
        f"START {start_str}\n"
    )


def make_chat_messages(instance: TRSInstance) -> list[dict]:
    """
    Return messages list in OpenAI/HuggingFace chat format.
    """
    return [
        {"role": "system",  "content": SYSTEM},
        {"role": "user",    "content": make_prompt(instance)},
    ]