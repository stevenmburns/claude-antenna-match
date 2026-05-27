#!/usr/bin/env python
"""Smith chart for the 4-band fan dipole — with and without the N=5 matcher.

Two traces over a 16-30 MHz sweep:
  - bare antenna: Γ_load (no matcher), shows how far the antenna is from 50 Ω
  - matched: Γ_in at the matcher port, shows what the radio sees

4 design freqs marked on each trace. SWR=2 reference circle dashed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    evaluate_rational,
    fan_dipole_impedance,
)


F_DESIGN_MHZ = np.array([18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["17m", "15m", "12m", "10m"]
Z0 = 50.0
R_RAD = 50.0

F_LOW_MHZ = 19.144
F_HIGH_MHZ = 26.933

KINDS = ["R_shunt", "L_shunt", "C_series", "L_shunt", "C_series"]
VALUES_SI = np.array([153.0, 0.673e-6, 82.7e-12, 0.55e-6, 263.0e-12])


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def section_abcd(kind: str, val: float, omegas: np.ndarray) -> np.ndarray:
    n = len(omegas)
    if kind == "L_series":
        ze = 1j * omegas * val
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "C_series":
        ze = 1.0 / (1j * omegas * val)
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "L_shunt":
        ye = 1.0 / (1j * omegas * val)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "C_shunt":
        ye = 1j * omegas * val
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "R_shunt":
        ye = np.full(n, 1.0 / val, dtype=complex)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    raise ValueError(kind)


def cascade_abcd(values, omegas):
    n = len(omegas)
    M = np.array([[np.ones(n), np.zeros(n)], [np.zeros(n), np.ones(n)]], dtype=complex)
    for kind, val in zip(KINDS, values, strict=True):
        M = np.einsum("ilk,ljk->ijk", M, section_abcd(kind, val, omegas))
    return M


def z_load_of(omegas):
    el_lo = DipoleElement(leg_length_m=leg(F_LOW_MHZ), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(F_HIGH_MHZ), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return evaluate_rational(num, den, 1j * omegas)


def smith_skrf(ax, gamma: np.ndarray, freqs_mhz: np.ndarray, **plot_kw) -> None:
    s = gamma.reshape(-1, 1, 1)
    net = rf.Network(
        frequency=rf.Frequency.from_f(freqs_mhz, unit="MHz"),
        s=s,
        z0=Z0,
        name=plot_kw.pop("name", ""),
    )
    net.plot_s_smith(ax=ax, draw_labels=False, show_legend=False, **plot_kw)


def main() -> None:
    f_sweep_mhz = np.linspace(15.0, 32.0, 3000)
    w_sweep = 2 * np.pi * f_sweep_mhz * 1e6
    w_design = 2 * np.pi * F_DESIGN_MHZ * 1e6

    # Bare antenna
    z_load_sweep = z_load_of(w_sweep)
    g_load_sweep = (z_load_sweep - Z0) / (z_load_sweep + Z0)
    z_load_design = z_load_of(w_design)
    g_load_design = (z_load_design - Z0) / (z_load_design + Z0)

    # Matched (input to matcher loaded by antenna)
    M = cascade_abcd(VALUES_SI, w_sweep)
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    z_in_sweep = (A * z_load_sweep + B) / (C * z_load_sweep + D)
    g_in_sweep = (z_in_sweep - Z0) / (z_in_sweep + Z0)
    M_d = cascade_abcd(VALUES_SI, w_design)
    A_d, B_d, C_d, D_d = M_d[0, 0], M_d[0, 1], M_d[1, 0], M_d[1, 1]
    z_in_design = (A_d * z_load_design + B_d) / (C_d * z_load_design + D_d)
    g_in_design = (z_in_design - Z0) / (z_in_design + Z0)

    swr_load_design = (1 + np.abs(g_load_design)) / (1 - np.abs(g_load_design))
    swr_in_design = (1 + np.abs(g_in_design)) / (1 - np.abs(g_in_design))

    print(f"{'band':>5}  {'f (MHz)':>8}  {'bare SWR':>10}  {'matched SWR':>12}")
    print("-" * 50)
    for lbl, f, sw_l, sw_i in zip(BAND_LABEL, F_DESIGN_MHZ, swr_load_design, swr_in_design):
        print(f"  {lbl:>3}  {f:>8.3f}  {sw_l:>10.2f}  {sw_i:>12.2f}")
    print()

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 7))
    ax_bare = fig.add_subplot(1, 2, 1)
    ax_match = fig.add_subplot(1, 2, 2)

    # SWR=2 reference circle (|Γ| = 1/3)
    theta = np.linspace(0, 2 * np.pi, 200)
    for ax in (ax_bare, ax_match):
        ax.plot((1.0 / 3.0) * np.cos(theta), (1.0 / 3.0) * np.sin(theta), "k--", lw=0.7, alpha=0.5)

    # --- Bare antenna panel ---
    smith_skrf(ax_bare, g_load_sweep, f_sweep_mhz, color="C0", lw=1.5)
    smith_skrf(
        ax_bare, g_load_design, F_DESIGN_MHZ, color="C1", marker="o", markersize=11, linestyle=""
    )
    for f, g, lbl, sw in zip(F_DESIGN_MHZ, g_load_design, BAND_LABEL, swr_load_design):
        ax_bare.annotate(
            f"{lbl}\nSWR={sw:.1f}",
            xy=(g.real, g.imag),
            xytext=(12, 8),
            textcoords="offset points",
            fontsize=10,
            color="C1",
            fontweight="bold",
        )
    ax_bare.set_title(
        f"BARE antenna (no matcher) — sweep 15-32 MHz\n"
        f"f_low={F_LOW_MHZ:.2f} MHz, f_high={F_HIGH_MHZ:.2f} MHz, R_rad={R_RAD:.0f} Ω\n"
        "dashed circle = SWR 2",
        fontsize=11,
    )

    # --- Matched panel ---
    smith_skrf(ax_match, g_in_sweep, f_sweep_mhz, color="C2", lw=1.5)
    smith_skrf(
        ax_match, g_in_design, F_DESIGN_MHZ, color="C1", marker="o", markersize=11, linestyle=""
    )
    for f, g, lbl, sw in zip(F_DESIGN_MHZ, g_in_design, BAND_LABEL, swr_in_design):
        ax_match.annotate(
            f"{lbl}\nSWR={sw:.2f}",
            xy=(g.real, g.imag),
            xytext=(12, 8),
            textcoords="offset points",
            fontsize=10,
            color="C1",
            fontweight="bold",
        )
    ax_match.set_title(
        "WITH N=5 matcher (2 dB loss budget) — sweep 15-32 MHz\n"
        "R=153 Ω, L=0.67/0.55 µH, C=82.7/263 pF\n"
        "dashed circle = SWR 2",
        fontsize=11,
    )

    fig.tight_layout()
    out = "n5_4band_smith.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
