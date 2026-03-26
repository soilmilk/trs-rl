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

CRITICAL RULES — YOU MUST FOLLOW THESE:
1. After EVERY step, scan the entire expression for any rule that can still fire.
2. If ANY rule can fire anywhere in the expression — you MUST apply it and continue.
3. Only stop when you have checked every rule at every position and NONE apply.
4. A proof that stops before normal form is WRONG and receives no credit.
5. Do NOT write the final step until you have verified no rules apply.

Reason step by step inside <think> tags:
- Write out the current expression
- Check each rule: can it fire anywhere?
- If yes: apply it, write the new expression, check again
- If no rules fire: write PROOF

Then output your proof in this EXACT format:

PROOF
S1: <expression after step 1> RULE <rule number>
S2: <expression after step 2> RULE <rule number>
...
SN: <final normal form> RULE <rule number>

Example:
RULE 1: not(not(x)) => x
RULE 2: and(T, x) => x
RULE 3: and(F, x) => F
RULE 4: or(T, x) => T
RULE 5: or(F, x) => x

START not(not(and(T, and(F, not(not(T))))))

<think>
Current: not(not(and(T, and(F, not(not(T))))))
Check rules: RULE 1 fires on not(not(and(T, and(F, not(not(T)))))) 
Apply RULE 1: and(T, and(F, not(not(T))))

Current: and(T, and(F, not(not(T))))
Check rules: RULE 1 fires on not(not(T))
Apply RULE 1: and(T, and(F, T))

Current: and(T, and(F, T))
Check rules: RULE 2 fires on and(T, and(F, T))
Apply RULE 2: and(F, T)

Current: and(F, T)
Check rules: RULE 3 fires on and(F, T)
Apply RULE 3: F

Current: F
Check rules: RULE 1 — no. RULE 2 — no. RULE 3 — no. RULE 4 — no. RULE 5 — no.
No rules fire. F is in normal form.
</think>

PROOF
S1: and(T, and(F, not(not(T)))) RULE 1
S2: and(T, and(F, T)) RULE 1
S3: and(F, T) RULE 2
S4: F RULE 3"""


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
