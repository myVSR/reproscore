"""
src/scoring/rcs.py
==================
Reproducibility Composite Score (RCS).

RCS blends RRS (static readiness) and ROS (execution outcome) via a
coverage weight α proportional to the fraction of execution evidence
collected. Collapses to RRS when no ROS is available.

Run standalone:
    from src.scoring.rcs import RCSScorer
    scorer = RCSScorer()
    result = scorer.score(rrs=72.0, ros=78.8, coverage_weight_sum=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .rubric import Rubric, load_rubric


@dataclass
class RCSResult:
    rcs: float
    rrs: float
    ros: Optional[float]
    alpha: float          # coverage weight applied to ROS
    alpha_max: float
    coverage_level: str   # descriptive label


class RCSScorer:
    """
    Computes the Reproducibility Composite Score (Equation 11).

    α = min(coverage_weight_sum, 1) * alpha_max
    RCS = (1-α)*RRS + α*ROS  when ROS available
        = RRS                 when ROS is None
    """

    def __init__(self, rubric: Optional[Rubric] = None):
        self.rubric = rubric or load_rubric()

    def score(
        self,
        rrs: float,
        ros: Optional[float],
        coverage_weight_sum: float = 0.0,
    ) -> RCSResult:
        """
        Compute RCS.

        Parameters
        ----------
        rrs : float
            Reproducibility Readiness Score (0-100).
        ros : float | None
            Reproducibility Outcome Score (0-100), or None if not available.
        coverage_weight_sum : float
            Sum of ros_component weights for available components (0-1).
            Pass ROSResult.coverage_weight_sum.
        """
        alpha_max = self.rubric.rcs["alpha_max"]
        alpha_min = self.rubric.rcs["alpha_min"]

        if ros is None or coverage_weight_sum == 0.0:
            return RCSResult(
                rcs=round(rrs, 2),
                rrs=round(rrs, 2),
                ros=None,
                alpha=0.0,
                alpha_max=alpha_max,
                coverage_level="No execution data",
            )

        # Equation 9: α = min(Σvj, 1) * alpha_max
        alpha = min(coverage_weight_sum, 1.0) * alpha_max
        # Equation 10: floor when any ROS component is available
        alpha = max(alpha, alpha_min)

        rcs = (1.0 - alpha) * rrs + alpha * ros
        rcs = max(0.0, min(100.0, rcs))

        level = _coverage_level(coverage_weight_sum)

        return RCSResult(
            rcs=round(rcs, 2),
            rrs=round(rrs, 2),
            ros=round(ros, 2),
            alpha=round(alpha, 4),
            alpha_max=alpha_max,
            coverage_level=level,
        )


def _coverage_level(w: float) -> str:
    if w <= 0:
        return "No execution data"
    elif w <= 0.10:
        return "Notebooks only"
    elif w <= 0.65:
        return "Install + execution"
    elif w <= 0.80:
        return "All except determinism"
    else:
        return "Full ROS"
