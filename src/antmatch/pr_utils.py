"""Positive-real utility checks.

Copied from claude-driving-point-synthesis (`dpsynth.pr_utils`). Kept as
a standalone helper to avoid taking that whole package as a dependency
for the simple PR sanity check used by the antenna model tests.
"""

from __future__ import annotations

import numpy as np


def check_pr_vectorized(
    num: np.ndarray, den: np.ndarray, n_points: int = 4000
) -> tuple[bool, float]:
    """Fast PR check: verify Re[Z(jw)] >= 0 on a sampled frequency grid.

    Returns (is_pr, min_re) where min_re is the minimum of Re[Z(jw)].
    """
    omegas = np.concatenate(
        [
            np.array([0.0]),
            np.logspace(-4, 4, n_points),
        ]
    )
    s_vals = 1j * omegas

    num_vals = np.zeros_like(s_vals)
    for c in reversed(num):
        num_vals = num_vals * s_vals + c
    den_vals = np.zeros_like(s_vals)
    for c in reversed(den):
        den_vals = den_vals * s_vals + c

    mask = np.abs(den_vals) > 1e-15
    z_vals = np.full_like(s_vals, np.inf)
    z_vals[mask] = num_vals[mask] / den_vals[mask]

    min_re = float(np.min(z_vals[mask].real))
    return min_re >= -1e-8, min_re
