"""Simple analytical models for wire antennas.

These models are deliberately crude — they exist to give the matching-network
synthesis pipeline a realistic *shape* of Z_a(s) to work against, not to
substitute for NEC or a measured S11. Use them for design exploration; check
final element dimensions against a real EM simulator before cutting wire.

Conventions
-----------
- All lengths in meters, frequencies in Hz, impedances in ohms.
- Z_a(s) is returned as ``(num, den)`` ascending-power coefficient arrays
  (numpy.ndarray of float), suitable for feeding to ``hazony_synthesis`` or
  ``bott_duffin_synthesis``.

Single-element dipole model
---------------------------
Near its first resonance, a thin wire dipole behaves like a series RLC:

    Z(s) = R + s·L + 1/(s·C)

with R ≈ R_rad (radiation resistance at the feedpoint),
ω₀ = 1/√(L·C) the first resonance, and Q = ω₀·L/R the loaded Q of the
resonance. Length is mapped to frequency via the standard rule of thumb

    L_total ≈ VF · c / (2·f₀)      (VF ≈ 0.95 for bare wire)

so f₀ = VF·c/(2·L_total) = VF·c/(4·L_leg).

The model **does not** capture: mutual coupling between elements, height-
above-ground effects on R_rad, wire-diameter-dependent Q, or harmonic
resonances. Override ``r_rad`` and ``q`` per element if you have better
estimates from NEC or measurement.

Fan dipole
----------
Two parallel dipoles fed from a common feedpoint:

    Y_a(s) = Y_1(s) + Y_2(s)   ⇒   Z_a(s) = 1 / Y_a(s)

This gives a 4th-order positive-real rational. Mutual coupling between the
two wires is **ignored** — for realistic spacings (1-3 wire diameters at the
feedpoint, fanning out at ~30°) this typically shifts each resonance by
1-3 % and slightly couples the Q's; acceptable for a first matching design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_LIGHT = 299_792_458.0  # m/s
VF_BARE_WIRE = 0.95  # velocity factor for thin bare wire in air


# ---------------------------------------------------------------------------
# Single-element model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DipoleElement:
    """Series-RLC model of one half-wave wire dipole near its first resonance."""

    leg_length_m: float
    r_rad: float = 65.0  # ohms; 73 free-space, lower at HF heights
    q: float = 10.0  # loaded Q of the first resonance
    vf: float = VF_BARE_WIRE

    @property
    def f0_hz(self) -> float:
        return self.vf * C_LIGHT / (4.0 * self.leg_length_m)

    @property
    def w0(self) -> float:
        return 2.0 * np.pi * self.f0_hz

    @property
    def l_h(self) -> float:
        return self.q * self.r_rad / self.w0

    @property
    def c_f(self) -> float:
        return 1.0 / (self.w0**2 * self.l_h)

    def impedance_coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (num, den) in ascending powers of s for Z(s) = R + sL + 1/(sC).

        Multiply numerator and denominator by s·C to clear the 1/s term:
            Z(s) = (1 + s·R·C + s²·L·C) / (s·C)
        """
        r, l, c = self.r_rad, self.l_h, self.c_f
        num = np.array([1.0, r * c, l * c])
        den = np.array([0.0, c])
        return num, den


# ---------------------------------------------------------------------------
# Fan dipole (parallel combination)
# ---------------------------------------------------------------------------


def _poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[: len(a)] += a
    out[: len(b)] += b
    return out


def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(a, b)


def _poly_trim(p: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Strip trailing zeros (highest powers) within tolerance."""
    last = len(p)
    while last > 1 and abs(p[last - 1]) <= tol:
        last -= 1
    return p[:last]


def fan_dipole_impedance(
    elements: list[DipoleElement],
) -> tuple[np.ndarray, np.ndarray]:
    """Z_a(s) of N dipoles in parallel at a common feedpoint.

    Returns (num, den) ascending-power coefficients. Mutual coupling between
    elements is ignored (see module docstring).
    """
    if not elements:
        raise ValueError("Need at least one element.")

    # Y_a = sum_i 1/Z_i = sum_i den_i / num_i.
    # Common denominator is prod_i num_i.
    nums = [el.impedance_coeffs()[0] for el in elements]
    dens = [el.impedance_coeffs()[1] for el in elements]

    num_common = np.array([1.0])
    for n in nums:
        num_common = _poly_mul(num_common, n)

    y_a_num = np.array([0.0])
    for i, d in enumerate(dens):
        other = np.array([1.0])
        for j, n in enumerate(nums):
            if j == i:
                continue
            other = _poly_mul(other, n)
        y_a_num = _poly_add(y_a_num, _poly_mul(d, other))

    # Z_a = 1 / Y_a = num_common / y_a_num
    z_num = _poly_trim(num_common)
    z_den = _poly_trim(y_a_num)
    return z_num, z_den


# ---------------------------------------------------------------------------
# Convenience: spot-frequency evaluation
# ---------------------------------------------------------------------------


def evaluate_rational(
    num: np.ndarray, den: np.ndarray, s_vals: np.ndarray
) -> np.ndarray:
    """Evaluate Z(s) = num(s)/den(s) at complex points s_vals via Horner."""
    num_v = np.zeros_like(s_vals, dtype=complex)
    for c in reversed(num):
        num_v = num_v * s_vals + c
    den_v = np.zeros_like(s_vals, dtype=complex)
    for c in reversed(den):
        den_v = den_v * s_vals + c
    return num_v / den_v
