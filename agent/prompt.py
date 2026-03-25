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
  - Single lowercase letters are variables (match anything).
  - Lowercase words followed by (...) are function symbols.

Normal form: an expression where no rule applies anywhere in it.

Reason step by step inside <think> tags. Work through which rules could fire, at which positions, and in what order.

Then output your proof in this EXACT format (no extra text after PROOF):

PROOF
S1: <expression after step 1> RULE <rule number>
S2: <expression after step 2> RULE <rule number>
...
SN: <final normal form> RULE <rule number>

Example:
RULE 1: and(T, x) => x
RULE 2: and(F, x) => F

START and(and(T, F), x)
TARGET F

<think>
The start is and(and(T, F), x).
RULE 1 fires on and(T, F) giving and(F, x).
RULE 2 fires on and(F, x) giving F.
F has no rules that apply, so it is in normal form.
</think>

PROOF
S1: and(F, x) RULE 1
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
        f"TARGET {target_str}"
    )


def make_chat_messages(instance: TRSInstance) -> list[dict]:
    """
    Return messages list in OpenAI/HuggingFace chat format.
    """
    return [
        {"role": "system",  "content": SYSTEM},
        {"role": "user",    "content": make_prompt(instance)},
    ]
