#!/usr/bin/env python
"""Smith chart for the L-tuner design (N=2: C_series + L_shunt).

Each band has its own C_series value (switched), so there are 4
matched curves — one per band — overlaid on a single chart along
with the bare antenna sweep for reference.

Per-band C values (variable per band):
  17m: 50.2 pF
  15m: 528  pF
  12m: 72.2 pF
  10m: 330  pF
Shared: L_shunt = 0.701 µH
Antenna: f_low = 21.65 MHz, f_high = 28.74 MHz
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
BAND_COLORS = ["C2", "C1", "C4", "C0"]
Z0 = 50.0
R_RAD = 50.0
F_LOW_MHZ = 21.65
F_HIGH_MHZ = 28.74

# L-tuner: C_series + L_shunt, source-to-load order
L_SHUNT_H = 0.701e-6
# Per-band C_series (one per band)
C_PER_BAND_F = {
    "17m": 50.2e-12,
    "15m": 528e-12,
    "12m": 72.2e-12,
    "10m": 330e-12,
}


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def z_load_at(omegas):
    el_lo = DipoleElement(leg_length_m=leg(F_LOW_MHZ), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(F_HIGH_MHZ), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return evaluate_rational(num, den, 1j * omegas)


def z_in_matched(omegas, c_series_F, l_shunt_H):
    """ABCD cascade C_series then L_shunt, terminated in antenna load."""
    z_l = z_load_at(omegas)
    # Series C: V_in = V_out + Z_C * I_out, I_in = I_out
    z_c = 1.0 / (1j * omegas * c_series_F)
    # Shunt L: V_in = V_out, I_in = Y_L * V_out + I_out
    y_l = 1.0 / (1j * omegas * l_shunt_H)
    # Cascade: M_C @ M_L:
    # M_C = [[1, Z_C],[0,1]]; M_L = [[1,0],[Y_L,1]]
    # M = [[1 + Z_C*Y_L, Z_C], [Y_L, 1]]
    A = 1.0 + z_c * y_l
    B = z_c
    C = y_l
    D = np.ones_like(omegas, dtype=complex)
    z_in = (A * z_l + B) / (C * z_l + D)
    return z_in, z_l


def smith_skrf(ax, gamma, freqs_mhz, **plot_kw):
    s = gamma.reshape(-1, 1, 1)
    net = rf.Network(
        frequency=rf.Frequency.from_f(freqs_mhz, unit="MHz"),
        s=s,
        z0=Z0,
        name=plot_kw.pop("name", ""),
    )
    net.plot_s_smith(ax=ax, draw_labels=False, show_legend=False, **plot_kw)


def main():
    # Bare antenna sweep
    f_sweep_mhz = np.linspace(15.0, 32.0, 3000)
    w_sweep = 2 * np.pi * f_sweep_mhz * 1e6
    z_bare = z_load_at(w_sweep)
    g_bare = (z_bare - Z0) / (z_bare + Z0)

    # Per-band: small sweep around design freq using THAT band's C value
    band_traces = {}
    for lbl, f_des, color in zip(BAND_LABEL, F_DESIGN_MHZ, BAND_COLORS):
        c_val = C_PER_BAND_F[lbl]
        f_local = np.linspace(f_des - 0.5, f_des + 0.5, 100)
        w_local = 2 * np.pi * f_local * 1e6
        z_in, _ = z_in_matched(w_local, c_val, L_SHUNT_H)
        g_local = (z_in - Z0) / (z_in + Z0)
        # Also evaluate at the exact design freq
        z_in_d, _ = z_in_matched(np.array([2 * np.pi * f_des * 1e6]), c_val, L_SHUNT_H)
        g_d = ((z_in_d - Z0) / (z_in_d + Z0))[0]
        swr_d = (1 + abs(g_d)) / (1 - abs(g_d))
        band_traces[lbl] = {
            "f_local": f_local,
            "g_local": g_local,
            "g_design": g_d,
            "swr_design": float(swr_d),
            "c_val": c_val,
            "color": color,
            "f_des": f_des,
        }

    print("L-tuner per-band match check:")
    for lbl in BAND_LABEL:
        t = band_traces[lbl]
        print(
            f"  {lbl} @ {t['f_des']:.3f} MHz   C_series = {t['c_val'] * 1e12:.1f} pF   SWR = {t['swr_design']:.3f}"
        )
    print()

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 7))
    ax_bare = fig.add_subplot(1, 2, 1)
    ax_match = fig.add_subplot(1, 2, 2)

    # SWR=1.5 reference circle (|Γ|=0.2)
    theta = np.linspace(0, 2 * np.pi, 200)
    for ax, r in [(ax_bare, 1.0 / 3.0), (ax_match, 0.2)]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), "k--", lw=0.7, alpha=0.5)

    # --- BARE antenna ---
    smith_skrf(ax_bare, g_bare, f_sweep_mhz, color="C0", lw=1.4)
    z_bare_design = z_load_at(2 * np.pi * F_DESIGN_MHZ * 1e6)
    g_bare_design = (z_bare_design - Z0) / (z_bare_design + Z0)
    smith_skrf(
        ax_bare, g_bare_design, F_DESIGN_MHZ, color="C1", marker="o", markersize=11, linestyle=""
    )
    swr_bare = (1 + np.abs(g_bare_design)) / (1 - np.abs(g_bare_design))
    for f, g, lbl, sw in zip(F_DESIGN_MHZ, g_bare_design, BAND_LABEL, swr_bare):
        ax_bare.annotate(
            f"{lbl}\nSWR={sw:.1f}",
            xy=(g.real, g.imag),
            xytext=(11, 11),
            textcoords="offset points",
            fontsize=10,
            color="C1",
            fontweight="bold",
        )
    ax_bare.set_title(
        f"BARE antenna (no matcher) — sweep 15-32 MHz\n"
        f"f_low={F_LOW_MHZ:.2f} MHz, f_high={F_HIGH_MHZ:.2f} MHz, R_rad=50 Ω\n"
        "dashed circle = SWR 2",
        fontsize=11,
    )

    # --- L-tuner matched (per band) ---
    for lbl in BAND_LABEL:
        t = band_traces[lbl]
        smith_skrf(ax_match, t["g_local"], t["f_local"], color=t["color"], lw=2)
        # Mark design freq
        ax_match.scatter(
            t["g_design"].real,
            t["g_design"].imag,
            s=120,
            marker="o",
            color=t["color"],
            edgecolors="black",
            lw=1.5,
            zorder=10,
        )
        ax_match.annotate(
            f"{lbl}\n{t['f_des']:.2f} MHz\nC={t['c_val'] * 1e12:.0f}pF\nSWR={t['swr_design']:.2f}",
            xy=(t["g_design"].real, t["g_design"].imag),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=9,
            color=t["color"],
            fontweight="bold",
        )
    ax_match.set_title(
        "L-TUNER matched — each band uses its own C_series value\n"
        "(short sweep ±0.5 MHz around each band design freq)\n"
        "fixed L_shunt = 0.701 µH; dashed circle = SWR 1.5",
        fontsize=11,
    )

    fig.tight_layout()
    out = "l_tuner_smith.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
