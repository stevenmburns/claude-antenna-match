#!/usr/bin/env python
"""Diagnostic for v3 Darlington synthesis design.

Runs v2 (Yarman-Aksen) on the 5-band fan-dipole problem at several
degrees, then dissects the optimal Belevitch triple (h, f, g) to answer
the questions v3 design needs:

  1. Which n_at_origin wins at each degree? (transmission-zero structure)
  2. Are transmission zeros only at 0 / infinity, or off-axis? (decides
     whether v3 needs Type C/D sections or just plain LC ladder)
  3. Is g(s) cleanly Hurwitz? (sanity for our spectral factorization)
  4. Where are the reflection zeros (roots of h)? (Brune extraction order)
  5. How many sections will the Darlington recipe produce? (= deg(g))
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
from antmatch.yarman_aksen import (
    solve_yarman_aksen,
    transducer_gain_yarman,
)


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def t_to_swr(t: float) -> float:
    g = float(np.sqrt(max(0.0, 1.0 - t)))
    return (1 + g) / (1 - g) if g < 1 else float("inf")


def poly_roots_ascending(p: np.ndarray) -> np.ndarray:
    """np.roots wants descending; antmatch stores ascending."""
    p = np.asarray(p, dtype=float)
    # strip trailing zeros
    while len(p) > 1 and abs(p[-1]) < 1e-30:
        p = p[:-1]
    if len(p) <= 1:
        return np.array([], dtype=complex)
    return np.roots(p[::-1])


def fmt_root(r: complex) -> str:
    return f"{r.real:+.4f}{r.imag:+.4f}j"


def classify_root(r: complex, tol: float = 1e-6) -> str:
    if abs(r.real) < tol:
        return "axis"
    elif r.real < 0:
        return "LHP"
    else:
        return "RHP"


def diagnose(omegas: np.ndarray, z_l: np.ndarray, degree: int) -> None:
    print(f"\n--- degree d = {degree} -----------------------------------------------", flush=True)

    results = {}
    for k in range(degree + 1):
        try:
            res = solve_yarman_aksen(
                omegas, z_l, h_degree=degree, n_at_origin=k, n_restarts=8, max_iter=200
            )
            results[k] = res
            print(
                f"  n_at_origin={k}: T={res.worst_gain:.4f}  SWR={t_to_swr(res.worst_gain):.2f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  n_at_origin={k}: FAILED ({type(e).__name__}: {e})", flush=True)

    if not results:
        print("  no successful solve")
        return

    k_best = max(results, key=lambda k: results[k].worst_gain)
    result = results[k_best]
    t_worst = result.worst_gain
    print(
        f"\n  winner: n_at_origin={k_best}  "
        f"(n_at_inf={degree - k_best})  "
        f"T_worst={t_worst:.4f}  SWR={t_to_swr(t_worst):.2f}",
        flush=True,
    )

    # Dissect the winning triple
    triple = result.triple
    print(f"\n  h coeffs (ascending): {np.array2string(triple.h, precision=4)}")
    print(f"  f coeffs (ascending): {np.array2string(triple.f, precision=4)}")
    print(f"  g coeffs (ascending): {np.array2string(triple.g, precision=4)}")

    # Transmission zeros: by construction f(s) = s^n_at_origin, so all
    # zeros at origin. n_at_infinity is implicit (deg(g) - deg(f)).
    f_roots = poly_roots_ascending(triple.f)
    deg_f = len(triple.f) - 1
    deg_g = len(triple.g) - 1
    n_inf = deg_g - deg_f
    print(f"  transmission zeros: {k_best} at s=0, {n_inf} at s=∞")
    if len(f_roots) > 0:
        # Should all be effectively zero
        max_axis_abs = max(abs(r) for r in f_roots)
        print(f"    f-root max |r|: {max_axis_abs:.2e} (should be ~0 since f = s^k)")

    # Reflection zeros: roots of h(s) — these are zeros of s11
    h_roots = poly_roots_ascending(triple.h)
    print(f"  reflection zeros (roots of h, n={len(h_roots)}):")
    for r in h_roots:
        print(f"    {fmt_root(r)}  [{classify_root(r)}]")

    # Hurwitz check on g
    g_roots = poly_roots_ascending(triple.g)
    print(f"  g roots (must be LHP for Hurwitz, n={len(g_roots)}):")
    rhp_count = 0
    axis_count = 0
    for r in g_roots:
        cls = classify_root(r)
        if cls == "RHP":
            rhp_count += 1
        elif cls == "axis":
            axis_count += 1
        print(f"    {fmt_root(r)}  [{cls}]")
    if rhp_count > 0:
        print(f"  WARNING: {rhp_count} RHP root(s) in g — spectral factor BROKEN")
    elif axis_count > 0:
        print(f"  note: {axis_count} on-axis g root(s) — marginal Hurwitz")
    else:
        print("  g is strictly Hurwitz")

    # Sanity: does evaluating the triple give the same gain?
    omega_0 = float(np.exp(np.mean(np.log(omegas))))
    t_check = transducer_gain_yarman(triple, z_l, omegas / omega_0)
    print(f"  T(ω_design) check: {np.array2string(t_check, precision=4)}")

    # How many sections will Darlington extract?
    print(f"  expected v3 section count: {deg_g}  (degree of g)")
    if k_best == 0 or k_best == deg_g:
        print("  extraction kind: pure LP or HP ladder (all zeros at one end)")
    else:
        print("  extraction kind: mixed (some zeros at 0, some at ∞)")


def main() -> None:
    f_mhz = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
    w = 2 * np.pi * f_mhz * 1e6
    el20 = DipoleElement(leg_length_m=leg(14.3))
    el10 = DipoleElement(leg_length_m=leg(28.47))
    num, den = fan_dipole_impedance([el20, el10])
    z_l = evaluate_rational(num, den, 1j * w)

    print("=" * 70)
    print("v3 Darlington-synthesis diagnostic — 5-band fan dipole")
    print("=" * 70)
    print(f"design freqs (MHz): {f_mhz}")
    print("Z_L at design freqs:")
    for f, z in zip(f_mhz, z_l):
        print(f"  {f:7.3f}  {z.real:+8.2f} {z.imag:+8.2f}j")

    for d in [2, 4, 6, 8]:
        diagnose(w, z_l, d)


if __name__ == "__main__":
    main()
