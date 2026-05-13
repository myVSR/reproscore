"""
src/scoring/ros.py
==================
Reproducibility Outcome Score (ROS) — computed from execution evidence.

ROS is optional and only available when sandboxed execution has been performed.
It normalises over whichever subset of the five components is available.

Run standalone:
    from src.scoring.ros import ROSScorer, ExecutionEvidence
    scorer = ROSScorer()
    ev = ExecutionEvidence(install_success=True, notebook_exec_rate=0.75)
    ros = scorer.score(ev)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .rubric import Rubric, load_rubric


@dataclass
class ExecutionEvidence:
    """
    Evidence collected from sandboxed execution.
    None means the component was not attempted / not available.
    All scores are in [0, 100].
    """
    install_success: Optional[bool] = None       # I: pip/conda install exit code
    execution_success: Optional[bool] = None     # X: entry-point exit code
    output_determinism: Optional[float] = None   # Δ: hash equality across 3 runs (0-100)
    notebook_exec_rate: Optional[float] = None   # N: fraction of .ipynb that ran to completion (0-1)
    import_success_rate: Optional[float] = None  # E: fraction free from ImportError/ModuleNotFoundError (0-1)
    test_pass_rate: Optional[float] = None       # T: pytest/testthat pass rate (0-1)


@dataclass
class ROSResult:
    ros: Optional[float]                  # None if no components available
    available_components: list[str]
    component_scores: Dict[str, float]
    coverage_weight_sum: float            # Σvj for available components


class ROSScorer:
    """
    Computes the Reproducibility Outcome Score (ROS).

    Uses normalised weighted sum over available components only (Equation 8).
    Returns ros=None when no components are available.
    """

    def __init__(self, rubric: Optional[Rubric] = None):
        self.rubric = rubric or load_rubric()

    def score(self, ev: ExecutionEvidence) -> ROSResult:
        """Compute ROS from execution evidence."""
        weights = self.rubric.ros_components

        # Map component symbol → (score 0-100, weight)
        candidates: Dict[str, tuple[float, float]] = {}

        if ev.install_success is not None:
            candidates["I"] = (100.0 if ev.install_success else 0.0,
                               weights["I"]["weight"])

        if ev.execution_success is not None:
            candidates["X"] = (100.0 if ev.execution_success else 0.0,
                               weights["X"]["weight"])

        if ev.output_determinism is not None:
            candidates["delta"] = (float(ev.output_determinism),
                                   weights["delta"]["weight"])

        if ev.notebook_exec_rate is not None:
            candidates["N"] = (float(ev.notebook_exec_rate) * 100.0,
                               weights["N"]["weight"])

        if ev.import_success_rate is not None and "E" in weights:
            candidates["E"] = (float(ev.import_success_rate) * 100.0,
                               weights["E"]["weight"])

        if ev.test_pass_rate is not None:
            candidates["T"] = (float(ev.test_pass_rate) * 100.0,
                               weights["T"]["weight"])

        if not candidates:
            return ROSResult(
                ros=None,
                available_components=[],
                component_scores={},
                coverage_weight_sum=0.0,
            )

        weight_sum = sum(w for _, w in candidates.values())
        ros = sum(score * weight for score, weight in candidates.values()) / weight_sum

        return ROSResult(
            ros=round(ros, 2),
            available_components=list(candidates.keys()),
            component_scores={sym: round(score, 2) for sym, (score, _) in candidates.items()},
            coverage_weight_sum=round(weight_sum, 4),
        )
