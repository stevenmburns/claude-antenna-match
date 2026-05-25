"""Tests for broadband matching network optimization (matching module).

The focus is on the working v1: direct LC-ladder topology optimization.
We verify the cascade impedance evaluator against analytical L-section
behavior and check that the optimizer improves over baseline on the
fan-dipole / 5-HF-band benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    evaluate_rational,
    fan_dipole_impedance,
)
from antmatch.matching import (
    MatchingProblem,
    baseline_gain,
    cascade_zin,
    solve_joint_lengths_and_ladder,
    solve_lc_ladder,
)


def _leg_for(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


# ---------------------------------------------------------------------------
# cascade_zin
# ---------------------------------------------------------------------------


def test_cascade_passthrough_with_no_sections():
    z_l = np.array([10 + 5j, 50 - 20j])
    w = np.array([1e7, 1e8])
    z_in = cascade_zin([], np.array([]), z_l, w)
    assert np.allclose(z_in, z_l)


def test_cascade_series_l_adds_jwl():
    z_l = np.array([50 + 0j])
    w = np.array([2e8])
    L_h = 1e-7  # 100 nH
    z_in = cascade_zin(["L_series"], np.array([L_h]), z_l, w)
    expected = 50 + 1j * w[0] * L_h
    assert np.allclose(z_in, expected)


def test_cascade_shunt_c_parallels_correctly():
    z_l = np.array([50 + 0j])
    w = np.array([1e8])
    c = 1e-10  # 100 pF
    z_in = cascade_zin(["C_shunt"], np.array([c]), z_l, w)
    z_c = 1.0 / (1j * w[0] * c)
    expected = (50 * z_c) / (50 + z_c)
    assert np.allclose(z_in, expected)


# ---------------------------------------------------------------------------
# End-to-end: fan dipole + ladder optimizer
# ---------------------------------------------------------------------------


def _fan_dipole_problem() -> tuple[MatchingProblem, np.ndarray]:
    """20m+10m fan dipole on 5 HF amateur band centers."""
    el20 = DipoleElement(leg_length_m=_leg_for(14.3))
    el10 = DipoleElement(leg_length_m=_leg_for(28.47))
    num, den = fan_dipole_impedance([el20, el10])

    f_mhz = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
    w = 2 * np.pi * f_mhz * 1e6
    z = evaluate_rational(num, den, 1j * w)

    problem = MatchingProblem(
        omegas_design=w,
        z_load_design=z,
        omegas_grid=w,
        z_load_grid=z,
        omegas_breakpts=w,
        r_source=50.0,
    )
    return problem, f_mhz


def test_baseline_worst_band_is_15m():
    """Sanity check on the baseline numbers used in our other assertions."""
    problem, _ = _fan_dipole_problem()
    t = baseline_gain(problem.z_load_design, problem.r_source)
    # The 15m band (idx 2) is between the two dipole resonances at high |Z|;
    # baseline T should be the worst there.
    assert int(np.argmin(t)) == 2
    assert t.min() < 0.15  # worse than SWR ~25


def test_4_element_ladder_improves_worst_case_substantially():
    """4-element ladder should meaningfully beat baseline on worst-case T."""
    problem, _ = _fan_dipole_problem()
    baseline_worst = baseline_gain(problem.z_load_design, problem.r_source).min()

    result = solve_lc_ladder(
        problem,
        ["L_shunt", "C_series", "L_shunt", "C_series"],
        n_restarts=20,
        max_iter=500,
    )
    # Baseline worst-case T ≈ 0.12; 4-element ladder should clear T > 0.3
    # (SWR < ~9) — a real, if not great, improvement.
    assert result.worst_gain > 0.3
    assert result.worst_gain > 3 * baseline_worst


def test_6_element_ladder_at_least_as_good_as_4():
    """More elements should give at least as good worst-case gain."""
    problem, _ = _fan_dipole_problem()
    r4 = solve_lc_ladder(
        problem,
        ["L_shunt", "C_series", "L_shunt", "C_series"],
        n_restarts=20,
        max_iter=500,
    )
    r6 = solve_lc_ladder(
        problem,
        ["C_shunt", "L_series", "C_shunt", "L_series", "C_shunt", "L_series"],
        n_restarts=20,
        max_iter=500,
    )
    # Allow some slack — local minima can leave 6 slightly behind 4 on a
    # bad restart seed. Reject only large regressions.
    assert r6.worst_gain >= r4.worst_gain - 0.05


# ---------------------------------------------------------------------------
# Joint optimization: lengths + ladder
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_joint_3freq_achieves_excellent_match():
    """3 frequencies + 2 dipoles + 4-element ladder is essentially solvable."""
    f_mhz = np.array([14.300, 21.383, 28.470])
    w = 2 * np.pi * f_mhz * 1e6
    result = solve_joint_lengths_and_ladder(
        omegas_design=w,
        n_dipoles=2,
        ladder_kinds=["L_shunt", "C_series", "L_shunt", "C_series"],
        n_restarts=40,
        max_iter=800,
    )
    # Empirically reaches SWR ~1.4 (T ~0.97). Be conservative.
    assert result.worst_gain > 0.85, (
        f"3-freq joint optimum should give T > 0.85; got {result.worst_gain:.3f}"
    )


def test_joint_lengths_only_beats_fixed_baseline():
    """Just letting the lengths move (no matching net) should improve things."""
    f_mhz = np.array([14.300, 21.383, 28.470])
    w = 2 * np.pi * f_mhz * 1e6

    # Fixed-length baseline at the extreme resonances:
    el20 = DipoleElement(leg_length_m=_leg_for(14.3))
    el10 = DipoleElement(leg_length_m=_leg_for(28.47))
    num, den = fan_dipole_impedance([el20, el10])
    z_fixed = evaluate_rational(num, den, 1j * w)
    t_fixed_worst = baseline_gain(z_fixed, 50.0).min()

    # Length-only joint optimization with a small number of restarts so
    # the test fits inside the default pytest timeout. The 2x bound
    # leaves plenty of slack — empirically gets ~2.5x.
    result = solve_joint_lengths_and_ladder(
        omegas_design=w,
        n_dipoles=2,
        ladder_kinds=[],
        n_restarts=8,
        max_iter=300,
    )
    assert result.worst_gain > 2 * t_fixed_worst
