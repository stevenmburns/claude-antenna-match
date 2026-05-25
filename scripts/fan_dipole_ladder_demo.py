#!/usr/bin/env python
"""End-to-end demo: 20m+10m fan dipole + LC-ladder match on 5 HF bands.

For each band (20/17/15/12/10 m) we report:
  * baseline transducer gain with no matching network (source ↔ Z_L directly)
  * gain after a 2-element L-section
  * gain after a 4-element ladder
  * gain after a 6-element ladder

Reveals how matching quality scales with network complexity, and how badly
a fixed lossless network is limited at off-resonance bands (Bode-Fano in
action).
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
    MatchingProblem,
    baseline_gain,
    format_ladder,
    solve_lc_ladder,
)


def leg_for(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def t_to_swr(t: float) -> float:
    gamma = float(np.sqrt(max(0.0, 1.0 - t)))
    return (1 + gamma) / (1 - gamma) if gamma < 1 else float("inf")


def print_per_band(label: str, f_mhz: np.ndarray, t: np.ndarray) -> None:
    print(label)
    print(f"{'f (MHz)':>10} {'T':>8} {'SWR':>8}")
    for f, ti in zip(f_mhz, t, strict=False):
        print(f"{f:>10.3f} {ti:>8.4f} {t_to_swr(ti):>8.2f}")
    print(
        f"  worst-case T = {t.min():.4f}   worst SWR = {t_to_swr(float(t.min())):.2f}"
    )
    print()


def try_ladder(problem: MatchingProblem, kinds: list[str], f_mhz: np.ndarray) -> None:
    result = solve_lc_ladder(problem, kinds, max_iter=2000)
    print(f"{len(kinds)}-element ladder ({' + '.join(kinds)}):")
    print(format_ladder(result))
    print_per_band("  per-band:", f_mhz, result.gain_design)


def main() -> None:
    # 1. Fan dipole built for 20m and 10m
    el20 = DipoleElement(leg_length_m=leg_for(14.3))
    el10 = DipoleElement(leg_length_m=leg_for(28.47))
    num, den = fan_dipole_impedance([el20, el10])

    # 2. Five ham band centers
    f_centers_mhz = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
    w_design = 2 * np.pi * f_centers_mhz * 1e6
    z_design = evaluate_rational(num, den, 1j * w_design)

    print("Load impedance Z_L(jω) at each band center:")
    print(f"{'f (MHz)':>10} {'Re Z_L':>10} {'Im Z_L':>10}")
    for f, z in zip(f_centers_mhz, z_design, strict=False):
        print(f"{f:>10.3f} {z.real:>10.2f} {z.imag:>10.2f}")
    print()

    # 3. Baseline
    print_per_band(
        "Baseline (no matching network):",
        f_centers_mhz,
        baseline_gain(z_design, r_source=50.0),
    )

    problem = MatchingProblem(
        omegas_design=w_design,
        z_load_design=z_design,
        omegas_grid=w_design,  # unused by solve_lc_ladder
        z_load_grid=z_design,  # unused
        omegas_breakpts=w_design,  # unused
        r_source=50.0,
    )

    # 4. Try ladders of increasing length
    try_ladder(problem, ["L_shunt", "C_series"], f_centers_mhz)
    try_ladder(
        problem,
        ["L_shunt", "C_series", "L_shunt", "C_series"],
        f_centers_mhz,
    )
    try_ladder(
        problem,
        ["C_shunt", "L_series", "C_shunt", "L_series", "C_shunt", "L_series"],
        f_centers_mhz,
    )


if __name__ == "__main__":
    main()
