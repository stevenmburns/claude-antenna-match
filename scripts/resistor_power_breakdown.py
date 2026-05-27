#!/usr/bin/env python
"""Per-resistor power dissipation in the lossy matcher, given 100 W input.

Loads the optimum from the prior run (R_rad=65, joint length opt at
~2 dB worst loss) and forward-cascades voltage/current through each
section to get the actual power dropped in each resistor at every
design frequency.

Assumes 100 W AVAILABLE from a 50 Ω Thévenin source.

Reports per band:
  - reflected power (back into source)
  - power dissipated in each resistor
  - power delivered to load (antenna)
"""

from __future__ import annotations

import numpy as np

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    evaluate_rational,
    fan_dipole_impedance,
)


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0
P_AVAIL_W = 100.0

# ---- Result from the prior (R_rad=65) joint-length 2 dB run ----
F_LOW_MHZ = 16.157
F_HIGH_MHZ = 24.970
R_RAD = 65.0  # the run was with default R_rad

KINDS = [
    "R_shunt",
    "C_shunt",
    "L_series",
    "C_shunt",
    "R_shunt",
    "L_series",
    "C_shunt",
    "L_series",
]
VALUES_SI = np.array(
    [
        461.0,  # R_shunt (Ω)
        92.6e-12,  # C_shunt (F)
        0.496e-6,  # L_series (H)
        131.0e-12,  # C_shunt (F)
        562.0,  # R_shunt (Ω)
        1.10e-6,  # L_series (H)
        60.5e-12,  # C_shunt (F)
        0.56e-6,  # L_series (H)
    ]
)


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def z_load_at(omega: float) -> complex:
    el_lo = DipoleElement(leg_length_m=leg(F_LOW_MHZ), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(F_HIGH_MHZ), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return complex(evaluate_rational(num, den, np.array([1j * omega]))[0])


def section_abcd_at(kind: str, val: float, omega: float) -> np.ndarray:
    if kind == "L_series":
        ze = 1j * omega * val
        return np.array([[1, ze], [0, 1]], dtype=complex)
    if kind == "C_series":
        ze = 1.0 / (1j * omega * val)
        return np.array([[1, ze], [0, 1]], dtype=complex)
    if kind == "R_series":
        return np.array([[1, val], [0, 1]], dtype=complex)
    if kind == "L_shunt":
        ye = 1.0 / (1j * omega * val)
        return np.array([[1, 0], [ye, 1]], dtype=complex)
    if kind == "C_shunt":
        ye = 1j * omega * val
        return np.array([[1, 0], [ye, 1]], dtype=complex)
    if kind == "R_shunt":
        return np.array([[1, 0], [1.0 / val, 1]], dtype=complex)
    raise ValueError(kind)


def cascade_at(omega: float) -> np.ndarray:
    M = np.eye(2, dtype=complex)
    for k, v in zip(KINDS, VALUES_SI, strict=True):
        M = M @ section_abcd_at(k, v, omega)
    return M


def analyse(omega: float, p_avail_W: float = P_AVAIL_W) -> dict:
    z_l = z_load_at(omega)
    M = cascade_at(omega)
    z_in = (M[0, 0] * z_l + M[0, 1]) / (M[1, 0] * z_l + M[1, 1])

    # Normalize: V_s peak such that available power = p_avail_W
    # P_avail = |V_s|^2 / (8 R_s)  => |V_s| = sqrt(8 R_s P_avail)
    v_s_mag = np.sqrt(8.0 * Z0 * p_avail_W)
    v_s = complex(v_s_mag)  # phase 0
    # V1 (input to network), I1 (into port 1)
    i_in = v_s / (Z0 + z_in)
    v_in = v_s - Z0 * i_in
    # Power dissipated in source R_s:
    p_source = 0.5 * np.abs(i_in) ** 2 * Z0
    # Real power into network (port 1)
    p_into_net = 0.5 * (v_in * np.conj(i_in)).real

    # Walk forward through sections, tracking (V, I) at each internal node
    v, i = v_in, i_in
    per_section_power = []
    for kind, val in zip(KINDS, VALUES_SI, strict=True):
        if kind in ("L_series", "C_series"):
            ze = 1j * omega * val if kind == "L_series" else 1.0 / (1j * omega * val)
            v_new = v - ze * i
            i_new = i
            p_diss = 0.0
        elif kind == "R_series":
            v_new = v - val * i
            i_new = i
            p_diss = 0.5 * np.abs(i) ** 2 * val
        elif kind in ("L_shunt", "C_shunt"):
            ye = 1.0 / (1j * omega * val) if kind == "L_shunt" else 1j * omega * val
            v_new = v
            i_new = i - ye * v
            p_diss = 0.0
        elif kind == "R_shunt":
            v_new = v
            i_new = i - v / val
            p_diss = 0.5 * np.abs(v) ** 2 / val
        else:
            raise ValueError(kind)
        per_section_power.append(p_diss)
        v, i = v_new, i_new

    # After last section, (v, i) should be (V_load, I_load)
    v_load, i_load = v, i
    p_load = 0.5 * (v_load * np.conj(i_load)).real

    return {
        "z_in": z_in,
        "z_load": z_l,
        "p_source": float(p_source),
        "p_into_net": float(p_into_net),
        "p_load": float(p_load),
        "per_section_power": [float(p) for p in per_section_power],
    }


def main() -> None:
    print("Per-resistor dissipation at 100 W available source power")
    print(f"Topology: {' - '.join(KINDS)}")
    print(f"R positions: {[i for i, k in enumerate(KINDS) if k.startswith('R')]}")
    print(
        f"R values:    {[f'{VALUES_SI[i]:.1f} Ω' for i, k in enumerate(KINDS) if k.startswith('R')]}"
    )
    print()

    header = f"{'band':>5}  {'P_src(R_s)':>10}  {'P→net':>7}  {'P→load':>7}"
    r_idx = [i for i, k in enumerate(KINDS) if k.startswith("R")]
    for i in r_idx:
        header += f"  {'P[R' + str(i + 1) + '=' + str(int(VALUES_SI[i])) + 'Ω]':>16}"
    header += f"  {'sum check':>9}"
    print(header)
    print("-" * (len(header) + 2))

    max_per_r = {i: 0.0 for i in r_idx}
    for lbl, f_mhz in zip(BAND_LABEL, F_DESIGN_MHZ):
        omega = 2 * np.pi * f_mhz * 1e6
        a = analyse(omega)
        per_r_powers = [a["per_section_power"][i] for i in r_idx]
        for ri, pri in zip(r_idx, per_r_powers):
            max_per_r[ri] = max(max_per_r[ri], pri)
        sum_check = a["p_source"] + sum(a["per_section_power"]) + a["p_load"]
        row = f"  {lbl:>3}  {a['p_source']:>10.2f}  {a['p_into_net']:>7.2f}  {a['p_load']:>7.2f}"
        for pri in per_r_powers:
            row += f"  {pri:>16.2f}"
        row += f"  {sum_check:>9.2f}"
        print(row)

    print()
    print("WORST-CASE per-resistor dissipation across all 5 bands (for component sizing):")
    for i in r_idx:
        print(
            f"  R at position {i + 1} ({VALUES_SI[i]:.0f} Ω):  "
            f"max {max_per_r[i]:.2f} W  →  pick a ≥{int(max_per_r[i] * 2)} W resistor"
        )


if __name__ == "__main__":
    main()
