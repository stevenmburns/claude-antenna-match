"""Broadband matching network design with co-optimizable antenna geometry."""

from .antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    evaluate_rational,
    fan_dipole_impedance,
)
from .matching import (
    JointResult,
    LadderResult,
    MatchingProblem,
    SECTION_KINDS,
    baseline_gain,
    cascade_zin,
    format_ladder,
    solve_joint_lengths_and_ladder,
    solve_lc_ladder,
    transducer_gain,
)
from .pr_utils import check_pr_vectorized

__all__ = [
    "C_LIGHT",
    "VF_BARE_WIRE",
    "DipoleElement",
    "JointResult",
    "LadderResult",
    "MatchingProblem",
    "SECTION_KINDS",
    "baseline_gain",
    "cascade_zin",
    "check_pr_vectorized",
    "evaluate_rational",
    "fan_dipole_impedance",
    "format_ladder",
    "solve_joint_lengths_and_ladder",
    "solve_lc_ladder",
    "transducer_gain",
]
