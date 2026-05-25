#!/usr/bin/env python
"""v1 LC-ladder vs v2 Yarman-Aksen bound on the fan-dipole problem.

Asks: how much of the SWR plateau seen in v1 is *fundamental*
(Bode-Fano-bounded for the given fixed antenna and design frequencies)
vs how much is just the v1 topology being too restrictive?

v2 (Yarman-Aksen) parameterizes any lossless 2-port matching network
of a chosen complexity directly via Belevitch polynomials, with
realizability enforced by construction. Its optimum is an upper
bound on what any passive lossless matching network of that order
can achieve. The gap (v2 - v1) tells us whether the LC-ladder
topology was leaving headroom on the table.

Note: v2 currently reports T values only — it does not synthesize
component values from the optimum (h, f, g). That's a separate
Darlington/Brune extraction step. Building the actual matching
network from a v2 solution is future work.
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
    solve_lc_ladder,
)
from antmatch.yarman_aksen import solve_yarman_aksen


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def t_to_swr(t: float) -> float:
    g = float(np.sqrt(max(0.0, 1.0 - t)))
    return (1 + g) / (1 - g) if g < 1 else float("inf")


def best_v2(w: np.ndarray, z_l: np.ndarray, degree: int) -> tuple[float, int]:
    """Try all three n_at_origin choices (low-pass, band-pass-ish, high-pass)
    and return the best worst-case T plus the winning n_at_origin."""
    best_t, best_k = 0.0, 0
    for k in [0, degree // 2, degree]:
        try:
            res = solve_yarman_aksen(
                w, z_l, h_degree=degree, n_at_origin=k, n_restarts=12, max_iter=300
            )
            if res.worst_gain > best_t:
                best_t = res.worst_gain
                best_k = k
        except Exception:
            pass
    return best_t, best_k


def lc_ladder(problem: MatchingProblem, kinds: list[str]) -> float:
    res = solve_lc_ladder(problem, kinds, n_restarts=20, max_iter=500)
    return res.worst_gain


def run_scenario(name: str, f_mhz: np.ndarray) -> None:
    w = 2 * np.pi * f_mhz * 1e6
    el20 = DipoleElement(leg_length_m=leg(14.3))
    el10 = DipoleElement(leg_length_m=leg(28.47))
    num, den = fan_dipole_impedance([el20, el10])
    z_l = evaluate_rational(num, den, 1j * w)

    problem = MatchingProblem(
        omegas_design=w,
        z_load_design=z_l,
        omegas_grid=w,
        z_load_grid=z_l,
        omegas_breakpts=w,
        r_source=50.0,
    )

    print("=" * 70)
    print(f"{name} — fan dipole tuned for 14.3 and 28.47 MHz (fixed lengths)")
    print("=" * 70)

    t_base = float(baseline_gain(z_l, 50.0).min())
    print(f"Baseline (no match):        T={t_base:.4f}  SWR={t_to_swr(t_base):.2f}")

    # v1 LC-ladder at 2/4/6 elements
    print("\nv1: direct LC-ladder optimization")
    print(f"  {'topology':35s} {'T':>8} {'SWR':>8}")
    for kinds in [
        ["L_shunt", "C_series"],
        ["L_shunt", "C_series", "L_shunt", "C_series"],
        ["C_shunt", "L_series", "C_shunt", "L_series", "C_shunt", "L_series"],
    ]:
        t = lc_ladder(problem, kinds)
        print(f"  {' + '.join(kinds):35s} {t:>8.4f} {t_to_swr(t):>8.2f}")

    # v2 Yarman-Aksen at matching degrees
    print("\nv2: Yarman-Aksen Bode-Fano upper bound (any lossless N of degree d)")
    print(f"  {'degree d':35s} {'T':>8} {'SWR':>8}  {'n_at_origin':>12}")
    for d in [2, 4, 6, 8]:
        t, k = best_v2(w, z_l, d)
        print(f"  {'h_degree=' + str(d):35s} {t:>8.4f} {t_to_swr(t):>8.2f}  {k:>12}")
    print()


def main() -> None:
    run_scenario("3 frequencies", np.array([14.300, 21.383, 28.470]))
    run_scenario("5 frequencies", np.array([14.300, 18.1575, 21.383, 24.970, 28.470]))


if __name__ == "__main__":
    main()
