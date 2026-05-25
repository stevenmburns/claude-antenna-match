"""Tests for the Yarman-Aksen v2 matching network bound."""

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
from antmatch.matching import baseline_gain
from antmatch.yarman_aksen import (
    build_belevitch,
    f_from_transmission_zeros,
    hurwitz_spectral_factor,
    poly_mul,
    poly_neg_s,
    solve_yarman_aksen,
    transducer_gain_yarman,
)


# ---------------------------------------------------------------------------
# Spectral factor
# ---------------------------------------------------------------------------


def test_spectral_factor_constant():
    g = hurwitz_spectral_factor(np.array([1.09]))
    assert np.allclose(g, [np.sqrt(1.09)])


def test_spectral_factor_one_minus_s_squared():
    # 1 - s^2 = (1+s)(1-s); Hurwitz factor is (1+s) = [1,1] ascending.
    g = hurwitz_spectral_factor(np.array([1.0, 0.0, -1.0]))
    assert np.allclose(g, [1.0, 1.0])


def test_spectral_factor_s_squared_round_trip():
    # f(s) = s^2, so f.f* = s^4. Hurwitz factor should be s^2.
    f = np.array([0.0, 0.0, 1.0])
    ff = poly_mul(f, poly_neg_s(f))
    g = hurwitz_spectral_factor(ff)
    assert np.allclose(g, [0.0, 0.0, 1.0])
    gg = poly_mul(g, poly_neg_s(g))
    assert np.allclose(gg, ff)


def test_spectral_factor_general_round_trip():
    # Use a known Hurwitz g and verify g·g* round-trips through the factor.
    g_true = np.array([2.0, 3.0, 1.0])  # 2 + 3s + s^2 = (s+1)(s+2), Hurwitz
    gg = poly_mul(g_true, poly_neg_s(g_true))
    g = hurwitz_spectral_factor(gg)
    # Should match up to sign; leading coefficient is + by construction
    assert np.allclose(g, g_true)


# ---------------------------------------------------------------------------
# Transducer gain — identity-network sanity
# ---------------------------------------------------------------------------


def test_identity_network_recovers_baseline():
    """h=0 with f=1 builds a trivial 'wire' (s11=0, s21=1). The transducer
    gain should equal the direct source-to-load baseline."""
    triple = build_belevitch(np.array([0.0]), np.array([1.0]), sigma=-1)
    z_l = np.array([50 + 0j, 200 + 100j, 10 - 30j])
    omegas = np.array([1.0, 1.0, 1.0])
    t = transducer_gain_yarman(triple, z_l, omegas, r_source=50.0)
    expected = baseline_gain(z_l, 50.0)
    assert np.allclose(t, expected)


def test_transducer_gain_bounded_in_zero_one():
    """Random Belevitch triples should always produce 0 ≤ T ≤ 1."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        h = rng.normal(0.0, 1.0, size=3)
        f = f_from_transmission_zeros(n_at_origin=1)
        triple = build_belevitch(h, f, sigma=-1)
        z_l = np.array([50 + 30j, 100 - 80j, 20 + 10j])
        omegas = np.array([0.5, 1.0, 2.0])
        t = transducer_gain_yarman(triple, z_l, omegas, r_source=50.0)
        assert np.all(t >= -1e-9)
        assert np.all(t <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# End-to-end: v2 on fan dipole problem
# ---------------------------------------------------------------------------


def _leg_for(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def _three_freq_fan_dipole() -> tuple[np.ndarray, np.ndarray]:
    el20 = DipoleElement(leg_length_m=_leg_for(14.3))
    el10 = DipoleElement(leg_length_m=_leg_for(28.47))
    num, den = fan_dipole_impedance([el20, el10])
    f_mhz = np.array([14.300, 21.383, 28.470])
    w = 2 * np.pi * f_mhz * 1e6
    z = evaluate_rational(num, den, 1j * w)
    return w, z


def test_yarman_beats_baseline_at_modest_degree():
    """v2 with h_degree=4 should strongly beat the no-match baseline."""
    w, z = _three_freq_fan_dipole()
    t_base = float(baseline_gain(z, 50.0).min())
    res = solve_yarman_aksen(w, z, h_degree=4, n_at_origin=2, n_restarts=10, max_iter=300)
    assert res.worst_gain > 0.7  # SWR < ~3
    assert res.worst_gain > 4 * t_base


@pytest.mark.slow
def test_yarman_high_degree_approaches_perfect():
    """At h_degree=6 the bound should be close to perfect on the 3-freq problem."""
    w, z = _three_freq_fan_dipole()
    res = solve_yarman_aksen(w, z, h_degree=6, n_at_origin=3, n_restarts=20, max_iter=400)
    assert res.worst_gain > 0.9  # SWR < ~2
