"""
training/reward.py

compute_reward() — the single function called by the GRPO loop.

Input:  model output string + TRSInstance
Output: float in [0.0, 1.0]

The reward has two components (from design doc):
  R_total = 0.4 × (valid_steps / total_steps) + 0.6 × R_final
"""

from __future__ import annotations
from typing import Optional

from rewritelang import verify_proof, verify_proof_detailed
from generator.instance import TRSInstance
from agent.parser import parse_proof_from_output, count_think_tokens


def compute_reward(
    output: str,
    instance: TRSInstance,
    detailed: bool = False,
) -> float | dict:
    """
    Parse model output and compute shaped reward.

    Args:
        output:   Raw model output string (may include <think> block)
        instance: The TRSInstance the model was solving
        detailed: If True, return full debug dict instead of float

    Returns:
        float reward in [0.0, 1.0], or dict if detailed=True
    """
    steps = parse_proof_from_output(output, n_rules=len(instance.rules))

    if not steps:
        if detailed:
            return {
                "reward": 0.0,
                "parse_success": False,
                "valid_steps": 0,
                "total_steps": 0,
                "final_correct": False,
                "think_tokens": count_think_tokens(output),
                "failure": "no proof found in output",
            }
        return 0.0

    if detailed:
        result = verify_proof_detailed(
            rules      = instance.rules,
            start      = instance.start,
            proof_steps= steps,
            target     = instance.normal_form,
        )
        result["parse_success"]  = True
        result["think_tokens"]   = count_think_tokens(output)
        result["n_steps_parsed"] = len(steps)
        return result
    else:
        return verify_proof(
            rules      = instance.rules,
            start      = instance.start,
            proof_steps= steps,
            target     = instance.normal_form,
        )


def compute_reward_batch(
    outputs: list[str],
    instances: list[TRSInstance],
) -> list[float]:
    """
    Vectorised version for GRPO group processing.
    outputs[i] and instances[i] must correspond.
    """
    assert len(outputs) == len(instances)
    return [compute_reward(o, inst) for o, inst in zip(outputs, instances)]
