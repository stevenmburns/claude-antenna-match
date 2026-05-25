#!/usr/bin/env python
"""Smith chart of the 5-element fan dipole with mutual coupling, showing
the hybridized 4-resonance signature.

In the moderate-coupling regime (k_L_adj=0.10, α_R_adj=0.75 with best
pentagon ordering), the 5 element modes hybridize down to 4 visible
resonances on the Smith chart. The center band (12m or 15m, depending
on details) loses its independent dip.

Plots:
  - Uncoupled 5-element trace (faint, blue): 5 loops near the center.
  - Coupled 5-element trace (red): 4 loops near the center.
  - Design freqs marked. Detected resonance freqs (local SWR minima)
    also marked, to make the 5 vs 4 count obvious.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from scipy.signal import argrelmin
from scipy.optimize import minimize_scalar

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    fan_dipole_impedance_coupled,
)


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0
R_RAD = 50.0


def leg_for_f(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def make_elements(legs: np.ndarray) -> list[DipoleElement]:
    return [DipoleElement(leg_length_m=L, r_rad=R_RAD) for L in legs]


def build_pentagon_coupling(
    order: tuple[int, ...], kL_adj: float, kL_diag: float, aR_adj: float, aR_diag: float
) -> tuple[np.ndarray, np.ndarray]:
    adj = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    KL = np.zeros((5, 5))
    KR = np.zeros((5, 5))
    for v in range(5):
        for u in range(5):
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            kL = kL_adj if pair in adj else kL_diag
            aR = aR_adj if pair in adj else aR_diag
            i, j = order[v], order[u]
            KL[i, j] = kL
            KR[i, j] = aR
    return KL, KR


def smith_skrf(ax, gamma: np.ndarray, freqs_mhz: np.ndarray, **plot_kw) -> None:
    """Plot a Γ trace via scikit-rf so the chart graticule matches the
    rest of the project's Smith plots."""
    s = gamma.reshape(-1, 1, 1)
    net = rf.Network(
        frequency=rf.Frequency.from_f(freqs_mhz, unit="MHz"),
        s=s,
        z0=Z0,
        name=plot_kw.pop("name", ""),
    )
    net.plot_s_smith(ax=ax, draw_labels=False, show_legend=False, **plot_kw)


def main() -> None:
    legs = np.array([leg_for_f(f) for f in F_DESIGN_MHZ])

    # Pentagon vertex assignment: best ordering 20m-12m-15m-17m-10m
    # → band_index_at_vertex = (0, 3, 2, 1, 4)
    order = (0, 3, 2, 1, 4)
    KL, KR = build_pentagon_coupling(order, kL_adj=0.16, kL_diag=0.08, aR_adj=0.55, aR_diag=0.50)
    K_zero = np.zeros((5, 5))

    # Dense sweep across the HF region
    f_sweep_mhz = np.linspace(12.0, 32.0, 3000)
    w_sweep = 2 * np.pi * f_sweep_mhz * 1e6

    elements = make_elements(legs)
    z_uncoupled = fan_dipole_impedance_coupled(elements, K_zero, w_sweep, coupling_R=K_zero)
    z_coupled = fan_dipole_impedance_coupled(elements, KL, w_sweep, coupling_R=KR)

    # Apply per-element iterative retune (Strategy A — what physical tuning does)
    w_design = 2 * np.pi * F_DESIGN_MHZ * 1e6
    legs_tuned = legs.copy()
    for _pass in range(6):
        for i in range(5):

            def obj(L_i: float, i=i) -> float:
                trial = legs_tuned.copy()
                trial[i] = L_i
                z = fan_dipole_impedance_coupled(make_elements(trial), KL, w_design, coupling_R=KR)
                swr = (1 + np.abs((z - Z0) / (z + Z0))) / (1 - np.abs((z - Z0) / (z + Z0)))
                return float(swr[i])

            L0 = legs_tuned[i]
            res = minimize_scalar(
                obj,
                bounds=(0.70 * L0, 1.30 * L0),
                method="bounded",
                options={"xatol": 1e-5},
            )
            legs_tuned[i] = res.x
    print(
        "\nRetuned length shifts (cm): "
        + "  ".join(f"{(L - L0) * 100:+.2f}" for L, L0 in zip(legs_tuned, legs))
    )

    elements_tuned = make_elements(legs_tuned)
    z_tuned = fan_dipole_impedance_coupled(elements_tuned, KL, w_sweep, coupling_R=KR)

    g_uncoupled = (z_uncoupled - Z0) / (z_uncoupled + Z0)
    g_coupled = (z_coupled - Z0) / (z_coupled + Z0)
    g_tuned = (z_tuned - Z0) / (z_tuned + Z0)

    swr_uncoupled = (1 + np.abs(g_uncoupled)) / (1 - np.abs(g_uncoupled))
    swr_coupled = (1 + np.abs(g_coupled)) / (1 - np.abs(g_coupled))
    swr_tuned = (1 + np.abs(g_tuned)) / (1 - np.abs(g_tuned))

    # Find actual resonance minima of |Γ| in each trace
    min_idx_uncoupled = argrelmin(np.abs(g_uncoupled), order=20)[0]
    min_idx_coupled = argrelmin(np.abs(g_coupled), order=20)[0]
    min_idx_tuned = argrelmin(np.abs(g_tuned), order=20)[0]
    # Filter spurious tiny minima (require |Γ| < 0.5)
    min_idx_uncoupled = [i for i in min_idx_uncoupled if abs(g_uncoupled[i]) < 0.5]
    min_idx_coupled = [i for i in min_idx_coupled if abs(g_coupled[i]) < 0.5]
    min_idx_tuned = [i for i in min_idx_tuned if abs(g_tuned[i]) < 0.5]

    print(
        f"Uncoupled: {len(min_idx_uncoupled)} resonance minima at "
        f"{[f'{f_sweep_mhz[i]:.2f}' for i in min_idx_uncoupled]} MHz"
    )
    print(
        f"Coupled  : {len(min_idx_coupled)} resonance minima at "
        f"{[f'{f_sweep_mhz[i]:.2f}' for i in min_idx_coupled]} MHz"
    )
    print(
        f"Tuned    : {len(min_idx_tuned)} resonance minima at "
        f"{[f'{f_sweep_mhz[i]:.2f}' for i in min_idx_tuned]} MHz"
    )

    # ---------- two-panel figure: SWR sweep + Smith chart ----------
    fig = plt.figure(figsize=(18, 9))
    ax_swr = fig.add_subplot(1, 2, 1)
    ax_sm = fig.add_subplot(1, 2, 2)

    # --- SWR panel ---
    ax_swr.semilogy(f_sweep_mhz, swr_uncoupled, "C0", lw=1.0, alpha=0.5, label="uncoupled (5 dips)")
    ax_swr.semilogy(
        f_sweep_mhz,
        swr_coupled,
        "C3",
        lw=1.3,
        alpha=0.55,
        label="coupled, nominal lengths (4 dips, displaced)",
    )
    ax_swr.semilogy(
        f_sweep_mhz,
        swr_tuned,
        "C2",
        lw=2.0,
        label="coupled, per-element retuned (still 4 dips, pulled toward bands)",
    )
    ax_swr.axhline(2.0, color="grey", ls=":", lw=0.7)
    for f, lbl in zip(F_DESIGN_MHZ, BAND_LABEL):
        ax_swr.axvline(f, color="grey", ls=":", lw=0.4, alpha=0.6)
        ax_swr.text(
            f,
            0.97,
            lbl,
            transform=ax_swr.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="grey",
        )
    # Mark detected resonance points
    for i in min_idx_uncoupled:
        ax_swr.scatter(
            f_sweep_mhz[i], swr_uncoupled[i], color="C0", s=25, zorder=5, marker="v", alpha=0.6
        )
    for i in min_idx_coupled:
        ax_swr.scatter(
            f_sweep_mhz[i], swr_coupled[i], color="C3", s=35, zorder=5, marker="o", alpha=0.6
        )
    for i in min_idx_tuned:
        ax_swr.scatter(
            f_sweep_mhz[i],
            swr_tuned[i],
            color="C2",
            s=60,
            zorder=6,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
        )
    ax_swr.set_ylim(1, 30)
    ax_swr.set_xlabel("frequency  (MHz)")
    ax_swr.set_ylabel("SWR  (log)")
    ax_swr.set_title("Coupled vs uncoupled — count the resonance dips")
    ax_swr.legend(loc="upper right", fontsize=9)
    ax_swr.grid(True, which="both", alpha=0.3)

    # --- Smith panel ---
    smith_skrf(ax_sm, g_uncoupled, f_sweep_mhz, color="C0", lw=0.9, alpha=0.35)
    smith_skrf(ax_sm, g_coupled, f_sweep_mhz, color="C3", lw=1.1, alpha=0.45)
    smith_skrf(ax_sm, g_tuned, f_sweep_mhz, color="C2", lw=1.8)

    # Mark design frequencies on the TUNED trace (after iterative tuning)
    w_design = 2 * np.pi * F_DESIGN_MHZ * 1e6
    z_design_tuned = fan_dipole_impedance_coupled(elements_tuned, KL, w_design, coupling_R=KR)
    g_design_tuned = (z_design_tuned - Z0) / (z_design_tuned + Z0)
    smith_skrf(
        ax_sm, g_design_tuned, F_DESIGN_MHZ, color="C1", marker="o", markersize=11, linestyle=""
    )
    for f, g, lbl in zip(F_DESIGN_MHZ, g_design_tuned, BAND_LABEL):
        ax_sm.annotate(
            f"{lbl}",
            xy=(g.real, g.imag),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=13,
            color="C1",
            fontweight="bold",
        )

    # Mark detected resonance minima on the TUNED trace
    for i in min_idx_tuned:
        g = g_tuned[i]
        ax_sm.scatter(
            g.real,
            g.imag,
            marker="*",
            s=300,
            color="black",
            edgecolors="white",
            linewidths=2,
            zorder=10,
        )
        ax_sm.annotate(
            f"{f_sweep_mhz[i]:.1f} MHz",
            xy=(g.real, g.imag),
            xytext=(-18, -22),
            textcoords="offset points",
            fontsize=11,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", lw=0.7),
        )

    ax_sm.set_title(
        f"Smith chart (Z₀=50 Ω):  faint=uncoupled ({len(min_idx_uncoupled)}),  "
        f"thin=coupled ({len(min_idx_coupled)}),  bold=coupled+retuned ({len(min_idx_tuned)})\n"
        "★ = actual resonances of tuned antenna     ● = design freqs after tuning",
        fontsize=12,
    )
    fig.tight_layout()
    out = "hybridized_smith.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")

    # Print numeric summary — before and after retune
    z_design_nominal = fan_dipole_impedance_coupled(elements, KL, w_design, coupling_R=KR)
    g_design_nominal = (z_design_nominal - Z0) / (z_design_nominal + Z0)
    swr_nominal = (1 + np.abs(g_design_nominal)) / (1 - np.abs(g_design_nominal))
    swr_t = (1 + np.abs(g_design_tuned)) / (1 - np.abs(g_design_tuned))
    print()
    print(f"{'band':>5}  {'f_design':>10}  {'SWR nominal':>12}  {'SWR retuned':>12}")
    print("-" * 50)
    for lbl, f, s_n, s_t in zip(BAND_LABEL, F_DESIGN_MHZ, swr_nominal, swr_t):
        print(f"{lbl:>5}  {f:>10.3f}  {s_n:>12.2f}  {s_t:>12.2f}")


if __name__ == "__main__":
    main()
