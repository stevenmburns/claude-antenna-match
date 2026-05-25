#!/usr/bin/env python
"""Joint dipole-lengths + matching-ladder optimization.

For each scenario, we report three rows:
  (1) default lengths (resonant at extremes) + no matching network
  (2) optimum lengths + no matching network    — what lengths alone buy you
  (3) optimum lengths + a 4-element ladder     — full joint optimum

Run two scenarios:
  A. Easy:  3 frequencies (14.3, 21.383, 28.47 MHz), 2 dipoles
  B. Hard:  5 frequencies (20/17/15/12/10 m),       2 dipoles
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
from antmatch.matching import (
    baseline_gain,
    solve_joint_lengths_and_ladder,
)


def leg_for(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def t_to_swr(t: float) -> float:
    g = float(np.sqrt(max(0.0, 1.0 - t)))
    return (1 + g) / (1 - g) if g < 1 else float("inf")


def print_band_table(f_mhz: np.ndarray, t: np.ndarray, prefix: str = "  ") -> None:
    print(f"{prefix}{'f (MHz)':>10} {'T':>8} {'SWR':>8}")
    for f, ti in zip(f_mhz, t, strict=False):
        print(f"{prefix}{f:>10.3f} {ti:>8.4f} {t_to_swr(ti):>8.2f}")
    worst = float(t.min())
    print(f"{prefix}worst-case T = {worst:.4f}   worst SWR = {t_to_swr(worst):.2f}")


def scenario(f_mhz: np.ndarray, ladder_kinds: list[str], default_legs_m: np.ndarray) -> None:
    w = 2 * np.pi * f_mhz * 1e6

    # (1) default lengths, no match
    els = [DipoleElement(leg_length_m=float(L)) for L in default_legs_m]
    num, den = fan_dipole_impedance(els)
    z_default = evaluate_rational(num, den, 1j * w)
    t_default = baseline_gain(z_default, r_source=50.0)
    default_f_mhz = [VF_BARE_WIRE * C_LIGHT / (4.0 * L) / 1e6 for L in default_legs_m]
    print(
        f"(1) default lengths {default_legs_m.tolist()} m  (resonances near "
        f"{', '.join(f'{f:.2f}' for f in default_f_mhz)} MHz), no matching:"
    )
    print_band_table(f_mhz, t_default)
    print()

    # (2) optimum lengths, no match
    r_lonly = solve_joint_lengths_and_ladder(
        omegas_design=w,
        n_dipoles=len(default_legs_m),
        ladder_kinds=[],
        n_restarts=30,
        max_iter=500,
    )
    opt_f_mhz = [VF_BARE_WIRE * C_LIGHT / (4.0 * L) / 1e6 for L in r_lonly.leg_lengths_m]
    print(
        f"(2) optimum lengths {[round(L, 3) for L in r_lonly.leg_lengths_m]} m "
        f"(resonances near {', '.join(f'{f:.2f}' for f in opt_f_mhz)} MHz), no matching:"
    )
    print_band_table(f_mhz, r_lonly.gain_design)
    print()

    # (3) optimum lengths + ladder
    r_joint = solve_joint_lengths_and_ladder(
        omegas_design=w,
        n_dipoles=len(default_legs_m),
        ladder_kinds=ladder_kinds,
        n_restarts=40,
        max_iter=800,
    )
    opt_f_mhz_j = [VF_BARE_WIRE * C_LIGHT / (4.0 * L) / 1e6 for L in r_joint.leg_lengths_m]
    print(
        f"(3) joint optimum: lengths {[round(L, 3) for L in r_joint.leg_lengths_m]} m "
        f"(resonances near {', '.join(f'{f:.2f}' for f in opt_f_mhz_j)} MHz)"
    )
    print(f"    ladder ({' + '.join(ladder_kinds)}):")
    for k, v in zip(ladder_kinds, r_joint.ladder_values, strict=True):
        unit = "nH" if k.startswith("L") else "pF"
        scale = 1e9 if k.startswith("L") else 1e12
        print(f"      {k:9s}  {v * scale:.4g} {unit}")
    print_band_table(f_mhz, r_joint.gain_design)
    print()


def main() -> None:
    # =============================================================
    # SCENARIO A: 3 freqs, easy case
    # =============================================================
    print("=" * 70)
    print("SCENARIO A: 3 frequencies (14.300, 21.383, 28.470 MHz), 2 dipoles")
    print("=" * 70 + "\n")
    f_A = np.array([14.300, 21.383, 28.470])
    default_legs_A = np.array([leg_for(14.3), leg_for(28.47)])
    scenario(f_A, ["L_shunt", "C_series", "L_shunt", "C_series"], default_legs_A)

    # =============================================================
    # SCENARIO B: 5 freqs, hard case
    # =============================================================
    print("=" * 70)
    print("SCENARIO B: 5 frequencies (20/17/15/12/10 m), 2 dipoles")
    print("=" * 70 + "\n")
    f_B = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
    default_legs_B = np.array([leg_for(14.3), leg_for(28.47)])
    scenario(f_B, ["L_shunt", "C_series", "L_shunt", "C_series"], default_legs_B)


if __name__ == "__main__":
    main()
