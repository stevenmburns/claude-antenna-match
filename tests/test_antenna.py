"""Tests for the analytical fan dipole model."""

from __future__ import annotations

import numpy as np

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    evaluate_rational,
    fan_dipole_impedance,
)
from antmatch.pr_utils import check_pr_vectorized


def _leg_for(f_mhz: float, vf: float = VF_BARE_WIRE) -> float:
    return vf * C_LIGHT / (4.0 * f_mhz * 1e6)


def test_single_element_resonance():
    el = DipoleElement(leg_length_m=_leg_for(14.3))
    assert abs(el.f0_hz - 14.3e6) < 1e3  # within 1 kHz of 14.3 MHz

    num, den = el.impedance_coeffs()
    z_at_res = evaluate_rational(num, den, np.array([1j * el.w0]))[0]
    assert abs(z_at_res.imag) < 1e-6  # purely real at resonance
    assert abs(z_at_res.real - el.r_rad) < 1e-6


def test_single_element_positive_real():
    el = DipoleElement(leg_length_m=_leg_for(14.3))
    num, den = el.impedance_coeffs()
    is_pr, min_re = check_pr_vectorized(num, den)
    assert is_pr, f"single dipole should be PR, min Re = {min_re}"


def test_fan_dipole_two_resonances():
    el20 = DipoleElement(leg_length_m=_leg_for(14.3))
    el10 = DipoleElement(leg_length_m=_leg_for(28.47))
    num, den = fan_dipole_impedance([el20, el10])

    # Z should peak (parallel resonance) somewhere and dip (series-like, low
    # imag) near each element's resonance. We just check resonance frequencies
    # are reachable and that Re[Z(jω)] is positive there.
    for f in (14.3e6, 28.47e6):
        w = 2 * np.pi * f
        z = evaluate_rational(num, den, np.array([1j * w]))[0]
        assert z.real > 0


def test_fan_dipole_positive_real():
    el20 = DipoleElement(leg_length_m=_leg_for(14.3))
    el10 = DipoleElement(leg_length_m=_leg_for(28.47))
    num, den = fan_dipole_impedance([el20, el10])
    is_pr, min_re = check_pr_vectorized(num, den)
    assert is_pr, f"fan dipole should be PR, min Re = {min_re}"
