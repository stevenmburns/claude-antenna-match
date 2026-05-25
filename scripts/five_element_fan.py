#!/usr/bin/env python
"""5-element fan dipole — one element per design frequency.

Models the limit case where each amateur-band design frequency gets
its own series-RLC dipole. Plots Z_a Bode + Smith and reports SWR
at the design freqs (with NO matching network applied).
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


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def main() -> None:
    f_design_mhz = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
    elements = [DipoleElement(leg_length_m=leg(f)) for f in f_design_mhz]
    num, den = fan_dipole_impedance(elements)

    # 2-element baseline for comparison
    num2, den2 = fan_dipole_impedance(
        [DipoleElement(leg_length_m=leg(14.3)), DipoleElement(leg_length_m=leg(28.47))]
    )

    print(f"5-element Z_a(s): num degree {len(num) - 1}, den degree {len(den) - 1}")
    print(f"2-element Z_a(s): num degree {len(num2) - 1}, den degree {len(den2) - 1}")
    print()
    print("Element leg lengths:")
    for f, el in zip(f_design_mhz, elements):
        print(
            f"  {f:7.4f} MHz → leg = {el.leg_length_m * 100:6.2f} cm "
            f"(L = {el.l_h * 1e6:6.2f} µH, C = {el.c_f * 1e12:6.2f} pF)"
        )

    z0 = 50.0
    f_sweep = np.linspace(5e6, 35e6, 4000)
    w_sweep = 2 * np.pi * f_sweep
    z5 = evaluate_rational(num, den, 1j * w_sweep)
    z2 = evaluate_rational(num2, den2, 1j * w_sweep)
    g5 = (z5 - z0) / (z5 + z0)
    g2 = (z2 - z0) / (z2 + z0)

    w_design = 2 * np.pi * f_design_mhz * 1e6
    z5_d = evaluate_rational(num, den, 1j * w_design)
    z2_d = evaluate_rational(num2, den2, 1j * w_design)
    g5_d = (z5_d - z0) / (z5_d + z0)
    g2_d = (z2_d - z0) / (z2_d + z0)

    def swr(g):
        return (1 + np.abs(g)) / (1 - np.abs(g))

    print()
    print("Raw-antenna SWR at design freqs (NO matching network):")
    print(f"  {'f (MHz)':>10} {'2-elem SWR':>12} {'5-elem SWR':>12}")
    for f, gA, gB in zip(f_design_mhz, g2_d, g5_d):
        print(f"  {f:>10.3f} {swr(gA):>12.2f} {swr(gB):>12.2f}")

    # ----- Bode magnitude -----
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fmhz = f_sweep / 1e6
    axes[0].semilogx(
        fmhz, 20 * np.log10(np.abs(z2)), color="C0", lw=1.2, label="2 elements (current)", alpha=0.7
    )
    axes[0].semilogx(
        fmhz, 20 * np.log10(np.abs(z5)), color="C3", lw=1.6, label="5 elements (one per band)"
    )
    axes[0].axhline(20 * np.log10(50), color="grey", ls=":", lw=0.8, label="50 Ω")
    for f in f_design_mhz:
        axes[0].axvline(f, color="grey", ls="--", lw=0.5, alpha=0.5)
    axes[0].scatter(f_design_mhz, 20 * np.log10(np.abs(z5_d)), color="C3", s=40, zorder=5)
    axes[0].set_ylabel("|Z_a|  (dBΩ)")
    axes[0].set_title("Z_a(jω): 2-element vs 5-element fan dipole")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(fmhz, swr(g2), color="C0", lw=1.2, label="2 elements", alpha=0.7)
    axes[1].semilogx(fmhz, swr(g5), color="C3", lw=1.6, label="5 elements")
    axes[1].axhline(1.0, color="black", lw=0.4)
    axes[1].axhline(2.0, color="grey", ls=":", lw=0.8, label="SWR 2")
    for f in f_design_mhz:
        axes[1].axvline(f, color="grey", ls="--", lw=0.5, alpha=0.5)
    axes[1].scatter(f_design_mhz, swr(g5_d), color="C3", s=40, zorder=5)
    axes[1].scatter(f_design_mhz, swr(g2_d), color="C0", s=40, zorder=5, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1, 100)
    axes[1].set_ylabel("SWR  (log)")
    axes[1].set_xlabel("frequency  (MHz)")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("five_element_bode.png", dpi=140, bbox_inches="tight")
    print("\nwrote five_element_bode.png")

    # ----- Smith chart -----
    fig2, ax = plt.subplots(figsize=(8, 8))
    net5 = rf.Network(
        frequency=rf.Frequency.from_f(f_sweep / 1e6, unit="MHz"),
        s=g5.reshape(-1, 1, 1),
        z0=z0,
        name="5-element sweep",
    )
    net5_d = rf.Network(
        frequency=rf.Frequency.from_f(f_design_mhz, unit="MHz"),
        s=g5_d.reshape(-1, 1, 1),
        z0=z0,
        name="design freqs",
    )
    net5.plot_s_smith(ax=ax, draw_labels=False, show_legend=False, color="C3", lw=1.2)
    net5_d.plot_s_smith(
        ax=ax,
        draw_labels=True,
        show_legend=False,
        marker="o",
        markersize=9,
        linestyle="",
        color="C1",
    )
    for f_hz, g in zip(f_design_mhz * 1e6, g5_d):
        ax.annotate(
            f"{f_hz / 1e6:.2f} MHz",
            xy=(g.real, g.imag),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="C1",
        )
    ax.set_title(
        "5-element fan dipole Smith chart (Z₀ = 50 Ω)\nsweep 5-35 MHz; orange dots = design freqs"
    )
    fig2.tight_layout()
    fig2.savefig("five_element_smith.png", dpi=140, bbox_inches="tight")
    print("wrote five_element_smith.png")


if __name__ == "__main__":
    main()
