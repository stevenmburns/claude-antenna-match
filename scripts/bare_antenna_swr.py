#!/usr/bin/env python
"""Plot SWR vs frequency for the bare 2-element fan dipole (no matcher).

Compares the original 20m/10m-resonance antenna to the optimized
inverted-V at ~17 MHz / ~26 MHz, both at R_rad=50 Ω.

This is what the matcher's lossy 2 dB job has to undo.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
R_RAD = 50.0


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def bare_z(f_low_mhz: float, f_high_mhz: float, omegas: np.ndarray) -> np.ndarray:
    el_lo = DipoleElement(leg_length_m=leg(f_low_mhz), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(f_high_mhz), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return evaluate_rational(num, den, 1j * omegas)


def swr_from_z(z: np.ndarray) -> np.ndarray:
    g = (z - Z0) / (z + Z0)
    abs_g = np.minimum(np.abs(g), 0.999)
    return (1 + abs_g) / (1 - abs_g)


def main() -> None:
    f_sweep = np.linspace(12.0, 32.0, 4000)
    w_sweep = 2 * np.pi * f_sweep * 1e6
    w_design = 2 * np.pi * F_DESIGN_MHZ * 1e6

    configs = [
        ("orig: 14.30 / 28.47 MHz (the antenna you built)", 14.300, 28.470, "C0"),
        ("17m+12m: 18.16 / 24.97 MHz", 18.1575, 24.970, "C2"),
        ("N=5-optimized: ~17.0 / 25.8 MHz", 17.0, 25.8, "C3"),
    ]

    print(f"{'antenna':>40}  " + "  ".join(f"{lbl:>8}" for lbl in BAND_LABEL))
    print(f"{'f_low / f_high':>40}  " + "  ".join(f"{f:>8.3f}" for f in F_DESIGN_MHZ) + "  MHz")
    print(f"{'  (bare SWR per design freq)':>40}")
    print("-" * 100)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for lbl, f_lo, f_hi, color in configs:
        z = bare_z(f_lo, f_hi, w_sweep)
        z_d = bare_z(f_lo, f_hi, w_design)
        swr = swr_from_z(z)
        swr_d = swr_from_z(z_d)
        print(
            f"{lbl:>40}  "
            + "  ".join(f"{s:>8.2f}" for s in swr_d)
            + f"  (worst: {swr_d.max():.2f})"
        )
        axes[0].semilogy(f_sweep, swr, color=color, lw=1.6, label=lbl)
        axes[0].scatter(F_DESIGN_MHZ, swr_d, color=color, s=50, zorder=5, edgecolors="black")

        # Mag + phase
        axes[1].plot(f_sweep, np.abs(z), color=color, lw=1.2, alpha=0.5, label=f"|Z|, {lbl}")

    axes[0].axhline(1.0, color="black", lw=0.4)
    axes[0].axhline(2.0, color="grey", ls=":", lw=0.7, label="SWR=2")
    axes[0].axhline(2.5, color="grey", ls=":", lw=0.4)
    for f, lbl in zip(F_DESIGN_MHZ, BAND_LABEL):
        axes[0].axvline(f, color="grey", ls=":", lw=0.4, alpha=0.6)
        axes[0].text(
            f,
            0.97,
            lbl,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="grey",
        )
    axes[0].set_ylabel("SWR (log)")
    axes[0].set_ylim(1, 100)
    axes[0].set_title("Bare 2-element fan dipole SWR (no matching network), R_rad = 50 Ω")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    for f, _ in zip(F_DESIGN_MHZ, BAND_LABEL):
        axes[1].axvline(f, color="grey", ls=":", lw=0.4, alpha=0.6)
    axes[1].axhline(50, color="grey", ls=":", lw=0.7, label="50 Ω")
    axes[1].set_xlabel("frequency (MHz)")
    axes[1].set_ylabel("|Z| (Ω)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("bare_antenna_swr.png", dpi=140, bbox_inches="tight")
    print()
    print("wrote bare_antenna_swr.png")


if __name__ == "__main__":
    main()
