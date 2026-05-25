#!/usr/bin/env python
"""Smith chart of the fan-dipole antenna impedance via scikit-rf.

Same Z_a(s) model as antenna_bode.py — two parallel series-RLC dipoles
tuned for 14.3 and 28.47 MHz. Normalized to 50 Ω.
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
    el20 = DipoleElement(leg_length_m=leg(14.3))
    el10 = DipoleElement(leg_length_m=leg(28.47))
    num, den = fan_dipole_impedance([el20, el10])

    z0 = 50.0

    # Dense sweep over the HF band for the trace
    f_sweep = np.linspace(5e6, 35e6, 1500)
    z_sweep = evaluate_rational(num, den, 1j * 2 * np.pi * f_sweep)
    gamma_sweep = (z_sweep - z0) / (z_sweep + z0)

    # Design frequencies (5-band)
    f_design = np.array([14.300, 18.1575, 21.383, 24.970, 28.470]) * 1e6
    z_design = evaluate_rational(num, den, 1j * 2 * np.pi * f_design)
    gamma_design = (z_design - z0) / (z_design + z0)

    # Build scikit-rf Networks: a continuous sweep + a sparse "design" net
    s_sweep = gamma_sweep.reshape(-1, 1, 1)
    net_sweep = rf.Network(
        frequency=rf.Frequency.from_f(f_sweep / 1e6, unit="MHz"),
        s=s_sweep,
        z0=z0,
        name="fan dipole (sweep)",
    )
    s_design = gamma_design.reshape(-1, 1, 1)
    net_design = rf.Network(
        frequency=rf.Frequency.from_f(f_design / 1e6, unit="MHz"),
        s=s_design,
        z0=z0,
        name="design freqs",
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    net_sweep.plot_s_smith(ax=ax, draw_labels=False, show_legend=False, color="C0", lw=1.4)
    net_design.plot_s_smith(
        ax=ax,
        draw_labels=True,
        show_legend=False,
        marker="o",
        markersize=8,
        linestyle="",
        color="C1",
    )

    # Annotate each design point with its frequency in MHz
    for f_hz, g in zip(f_design, gamma_design):
        ax.annotate(
            f"{f_hz / 1e6:.2f} MHz",
            xy=(g.real, g.imag),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="C1",
        )

    ax.set_title(
        "Fan dipole Smith chart (Z₀ = 50 Ω)\n"
        "elements 14.30 / 28.47 MHz in parallel — sweep 5-35 MHz"
    )
    fig.tight_layout()
    out = "antenna_smith.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # SWR summary at design frequencies
    swr = (1 + np.abs(gamma_design)) / (1 - np.abs(gamma_design))
    print()
    print(f"{'f (MHz)':>10} {'|Γ|':>8} {'SWR':>8} {'Re Z':>10} {'Im Z':>10}")
    print("-" * 50)
    for f_hz, g, s in zip(f_design, gamma_design, swr):
        z = z0 * (1 + g) / (1 - g)
        print(f"{f_hz / 1e6:>10.3f} {abs(g):>8.3f} {s:>8.2f} {z.real:>+10.2f} {z.imag:>+10.2f}")


if __name__ == "__main__":
    main()
