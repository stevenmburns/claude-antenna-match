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
        r, L, c = self.r_rad, self.l_h, self.c_f
        num = np.array([1.0, r * c, L * c])
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


def evaluate_rational(num: np.ndarray, den: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """Evaluate Z(s) = num(s)/den(s) at complex points s_vals via Horner."""
    num_v = np.zeros_like(s_vals, dtype=complex)
    for c in reversed(num):
        num_v = num_v * s_vals + c
    den_v = np.zeros_like(s_vals, dtype=complex)
    for c in reversed(den):
        den_v = den_v * s_vals + c
    return num_v / den_v


# ---------------------------------------------------------------------------
# Fan dipole with mutual coupling (numerical, not rational)
# ---------------------------------------------------------------------------


def geometric_coupling_matrix(n: int, k_nn: float, decay: float | None = None) -> np.ndarray:
    """N×N coupling-coefficient matrix with nearest-neighbor strength k_nn.

    Off-diagonal entries decay geometrically with index distance:
        k_ij = k_nn * decay^(|i-j|-1)   for |i-j| >= 1
        k_ii = 0
    `decay` defaults to `k_nn` (so k_ij = k_nn^|i-j|), which keeps the
    matrix positive-definite for any 0 ≤ k_nn < 1.
    """
    if not 0 <= k_nn < 1:
        raise ValueError(f"k_nn must be in [0, 1), got {k_nn}")
    if decay is None:
        decay = k_nn
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = k_nn * decay ** (abs(i - j) - 1)
    return mat


def fan_dipole_impedance_coupled(
    elements: list[DipoleElement],
    coupling: np.ndarray,
    omegas: np.ndarray,
    coupling_R: np.ndarray | None = None,
) -> np.ndarray:
    """Z_a(jω) for N parallel dipoles with mutual coupling.

    `coupling[i,j]` is the inductive coupling coefficient k_L between the
    inductances of elements i and j: M_ij = k_L·√(L_i·L_j). Must be
    symmetric with zero diagonal.

    `coupling_R[i,j]` (optional) is the *resistive* coupling coefficient
    α_R for mutual radiation resistance: R_ij = α_R·√(R_i·R_j). Defaults
    to zero (pure magnetic coupling). For parallel half-wave dipoles at
    d/λ << 1 (the dominant case in a multi-spacer fan dipole), α_R is
    nearly 1 and is the DOMINANT coupling mechanism.

    Each frequency yields N×N complex Z:
        Z_ii(jω) = R_i + jω·L_i + 1/(jω·C_i)
        Z_ij(jω) = α_R_ij·√(R_i·R_j) + jω·k_L_ij·√(L_i·L_j)   (i≠j)

    All elements share feedpoint voltage V; Y_a = 1ᵀ·Z⁻¹·1.
    Returns Z_a(jω) at each frequency (numerical, not rational form).
    """
    n = len(elements)
    if coupling.shape != (n, n):
        raise ValueError(f"coupling shape {coupling.shape} != ({n}, {n})")
    if not np.allclose(coupling, coupling.T):
        raise ValueError("coupling matrix must be symmetric")
    if not np.allclose(np.diag(coupling), 0.0):
        raise ValueError("coupling matrix diagonal must be 0")
    if coupling_R is not None:
        if coupling_R.shape != (n, n):
            raise ValueError(f"coupling_R shape {coupling_R.shape} != ({n}, {n})")
        if not np.allclose(coupling_R, coupling_R.T):
            raise ValueError("coupling_R must be symmetric")
        if not np.allclose(np.diag(coupling_R), 0.0):
            raise ValueError("coupling_R diagonal must be 0")

    L = np.array([el.l_h for el in elements])
    C = np.array([el.c_f for el in elements])
    R = np.array([el.r_rad for el in elements])
    M = coupling * np.sqrt(np.outer(L, L))  # N×N mutual inductances
    R_mut = coupling_R * np.sqrt(np.outer(R, R)) if coupling_R is not None else np.zeros((n, n))

    omegas = np.asarray(omegas, dtype=float)
    z_a = np.zeros_like(omegas, dtype=complex)
    ones = np.ones(n)
    for idx, w in enumerate(omegas):
        z_mat = R_mut + 1j * w * M
        z_diag = R + 1j * w * L + 1.0 / (1j * w * C)
        z_mat[np.diag_indices(n)] = z_diag
        x = np.linalg.solve(z_mat, ones)
        z_a[idx] = 1.0 / np.sum(x)
    return z_a
