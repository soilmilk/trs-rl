"""
generator/curriculum.py

CurriculumTracker: automated phase advancement.

Phases progress along three axes simultaneously:
  - rule system complexity (n_rules)
  - expression depth (max_depth)
  - proof length (n_steps)

Advancement is automatic when solve_rate@1 >= threshold on eval set,
measured every eval_interval training steps.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PhaseConfig:
    phase:      int
    n_rules:    int
    max_depth:  int
    n_steps_min: int
    n_steps_max: int
    domain:     str
    description: str = ""

    def sample_n_steps(self, rng) -> int:
        return rng.randint(self.n_steps_min, self.n_steps_max)


# ---------------------------------------------------------------------------
# Phase definitions (matches design doc Section 5.1)
# ---------------------------------------------------------------------------
PHASE_CONFIGS: List[PhaseConfig] = [
    PhaseConfig(
        phase=1,
        n_rules=3,
        max_depth=2,
        n_steps_min=1,
        n_steps_max=1,
        domain="boolean",
        description="Single-step trivial. not(T)→F style.",
    ),
    PhaseConfig(
        phase=2,
        n_rules=5,
        max_depth=3,
        n_steps_min=2,
        n_steps_max=4,
        domain="boolean",
        description="Short chains. not(not(T))→T.",
    ),
    PhaseConfig(
        phase=3,
        n_rules=7,
        max_depth=5,
        n_steps_min=4,
        n_steps_max=8,
        domain="boolean",
        description="Moderate depth. Multiple boolean operators.",
    ),
    PhaseConfig(
        phase=4,
        n_rules=9,    # boolean(7) + arithmetic starts here
        max_depth=8,
        n_steps_min=8,
        n_steps_max=15,
        domain="boolean",   # arithmetic added as separate domain
        description="Deep nested + choice points. Arithmetic introduced.",
    ),
    PhaseConfig(
        phase=5,
        n_rules=12,
        max_depth=10,
        n_steps_min=10,
        n_steps_max=20,
        domain="mixed",
        description="Multi-domain. Boolean + arithmetic + abstract mixing.",
    ),
]


@dataclass
class CurriculumState:
    current_phase:    int  = 1
    training_step:    int  = 0
    phase_start_step: int  = 0

    # eval history: list of {step, phase, solve_rate}
    eval_history:     List[dict] = field(default_factory=list)

    # best solve rate seen in current phase
    best_solve_rate:  float = 0.0

    # timestamps for wall-clock tracking
    phase_start_time: float = field(default_factory=time.time)


class CurriculumTracker:
    """
    Manages phase progression. Fully automated — call .record_eval() after
    each evaluation and it will auto-advance when threshold is met.
    """

    def __init__(
        self,
        advance_threshold: float = 0.75,
        eval_interval:     int   = 250,
        n_phases:          int   = 5,
    ):
        self.advance_threshold = advance_threshold
        self.eval_interval     = eval_interval
        self.n_phases          = n_phases
        self.state             = CurriculumState()

    @property
    def current_phase(self) -> int:
        return self.state.current_phase

    @property
    def current_config(self) -> PhaseConfig:
        return PHASE_CONFIGS[self.state.current_phase - 1]

    def step(self) -> bool:
        """
        Called once per training step.
        Returns True if an eval should be run now.
        """
        self.state.training_step += 1
        return (self.state.training_step % self.eval_interval) == 0

    def record_eval(self, solve_rate: float) -> dict:
        """
        Record an eval result. Automatically advances phase if threshold met.

        Returns a dict describing what happened:
          {"advanced": bool, "old_phase": int, "new_phase": int, "solve_rate": float}
        """
        step = self.state.training_step
        phase = self.state.current_phase

        record = {
            "step":       step,
            "phase":      phase,
            "solve_rate": solve_rate,
            "timestamp":  time.time(),
        }
        self.state.eval_history.append(record)

        if solve_rate > self.state.best_solve_rate:
            self.state.best_solve_rate = solve_rate

        result = {
            "advanced":   False,
            "old_phase":  phase,
            "new_phase":  phase,
            "solve_rate": solve_rate,
        }

        # Advance if threshold met and not already at final phase
        if solve_rate >= self.advance_threshold and phase < self.n_phases:
            self.state.current_phase += 1
            self.state.phase_start_step = step
            self.state.best_solve_rate  = 0.0
            self.state.phase_start_time = time.time()

            result["advanced"]  = True
            result["new_phase"] = self.state.current_phase

            print(
                f"\n{'='*60}\n"
                f"  PHASE ADVANCE: {phase} → {self.state.current_phase}\n"
                f"  At training step {step}, solve_rate={solve_rate:.3f}\n"
                f"  New config: {self.current_config.description}\n"
                f"{'='*60}\n"
            )

        return result

    def get_instance_kwargs(self, rng) -> dict:
        """
        Return kwargs for generate_instance() for the current phase.
        Automatically handles mixed-domain phases.
        """
        cfg = self.current_config
        kwargs = {
            "n_rules":   cfg.n_rules,
            "max_depth": cfg.max_depth,
            "n_steps":   cfg.sample_n_steps(rng),
            "domain":    cfg.domain,
        }
        if cfg.domain == "mixed":
            kwargs["domains_for_mixed"] = ["boolean", "arithmetic", "abstract"]
        return kwargs

    def should_add_arithmetic(self) -> bool:
        return self.state.current_phase >= 4

    def should_add_abstract(self) -> bool:
        return self.state.current_phase >= 5

    def summary(self) -> dict:
        return {
            "current_phase":   self.state.current_phase,
            "training_step":   self.state.training_step,
            "best_solve_rate": self.state.best_solve_rate,
            "phase_config":    {
                "n_rules":   self.current_config.n_rules,
                "max_depth": self.current_config.max_depth,
                "n_steps":   f"{self.current_config.n_steps_min}-{self.current_config.n_steps_max}",
                "domain":    self.current_config.domain,
            },
            "eval_history_len": len(self.state.eval_history),
        }

    def save(self, path: str) -> None:
        import json, dataclasses
        with open(path, "w") as f:
            json.dump({
                "state": dataclasses.asdict(self.state),
                "config": {
                    "advance_threshold": self.advance_threshold,
                    "eval_interval":     self.eval_interval,
                    "n_phases":          self.n_phases,
                },
            }, f, indent=2)

    @staticmethod
    def load(path: str) -> "CurriculumTracker":
        with open(path) as f:
            d = json.load(f)
        tracker = CurriculumTracker(**d["config"])
        s = d["state"]
        tracker.state = CurriculumState(
            current_phase    = s["current_phase"],
            training_step    = s["training_step"],
            phase_start_step = s["phase_start_step"],
            eval_history     = s["eval_history"],
            best_solve_rate  = s["best_solve_rate"],
            phase_start_time = s["phase_start_time"],
        )
        return tracker
