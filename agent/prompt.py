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

CRITICAL — VARIABLES ARE WILDCARDS, NOT CONSTANTS:
  - x, y, z are variables. They match ANY subexpression, including nested functions.
  - and(F, x) => F means: if you see and(F, ANYTHING), the result is F.
  - and(F, x) does NOT mean and(F, x) literally — x disappears and is replaced.
  - You CANNOT apply and(F, x) => F to an expression that still has x as a free variable in the result.
  - After applying and(F, x) => F, the result is just F — not and(F, x) or F(x).

CRITICAL — CHECK THE OUTERMOST EXPRESSION FIRST:
  - Always attempt to apply rules at the ROOT of the expression before going deeper.
  - If not(not(...)) wraps the entire expression, strip it first with not(not(x)) => x.
  - Only recurse into children if no rule fires at the root.

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

--- EXAMPLE 2: Variable x gets absorbed — the variable VANISHES in the result ---
RULE 1: not(not(x)) => x
RULE 2: and(T, x) => x
RULE 3: and(F, x) => F
RULE 4: or(T, x) => T
RULE 5: or(F, x) => x
START or(and(F, x), not(not(T)))
TARGET T
<think>
The root is or(...). Check rule 4: or(T, x) — root is or(and(F,x), ...) not or(T,...). No match at root.
Check children. Left child: and(F, x). Rule 3 matches: and(F, x) => F. Apply there.
Note: x is a variable — and(F, x) becomes F, the variable x is absorbed and gone.
</think>
PROOF
S1: or(F, not(not(T))) RULE 3
S2: or(F, T) RULE 1
S3: T RULE 5

--- EXAMPLE 3: Outer wrapper must be stripped first ---
RULE 1: not(not(x)) => x
RULE 2: and(T, x) => x
RULE 3: and(F, x) => F
RULE 4: or(T, x) => T
RULE 5: or(F, x) => x
START not(not(and(T, F)))
TARGET F
<think>
Root is not(not(...)). Rule 1 fires at root. Strip the outer wrapper first.
</think>
PROOF
S1: and(T, F) RULE 1
S2: F RULE 2"""


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
