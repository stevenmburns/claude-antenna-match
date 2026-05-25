#!/usr/bin/env python
"""Bode plot of the fan-dipole antenna impedance Z_a(jω).

Shows magnitude (log), phase, and Re/Im decomposition across the HF
range, with the two element resonances and the 5 design frequencies
marked.
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


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def main() -> None:
    f_20m = 14.3
    f_10m = 28.47
    el20 = DipoleElement(leg_length_m=leg(f_20m))
    el10 = DipoleElement(leg_length_m=leg(f_10m))
    num, den = fan_dipole_impedance([el20, el10])

    # Sweep 5 to 35 MHz on a dense log grid
    f_sweep = np.logspace(np.log10(5e6), np.log10(35e6), 4000)
    w = 2 * np.pi * f_sweep
    z = evaluate_rational(num, den, 1j * w)

    f_design = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
    w_design = 2 * np.pi * f_design * 1e6
    z_design = evaluate_rational(num, den, 1j * w_design)

    f_mhz = f_sweep / 1e6
    mag_db = 20 * np.log10(np.abs(z))
    phase_deg = np.degrees(np.angle(z))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # ----- magnitude -----
    axes[0].semilogx(f_mhz, mag_db, color="C0", lw=1.6)
    axes[0].axhline(20 * np.log10(50), color="grey", ls=":", lw=0.8, label="50 Ω")
    for f in [f_20m, f_10m]:
        axes[0].axvline(f, color="C3", ls="--", lw=0.8, alpha=0.7)
    axes[0].scatter(
        f_design,
        20 * np.log10(np.abs(z_design)),
        color="C1",
        zorder=5,
        s=40,
        label="5 design freqs",
    )
    axes[0].set_ylabel("|Z_a|  (dBΩ)")
    axes[0].set_title(
        "Fan dipole Z_a(jω) — series-RLC tuned for "
        f"{f_20m:.2f} MHz (20 m) and {f_10m:.2f} MHz (10 m), in parallel"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    # ----- phase -----
    axes[1].semilogx(f_mhz, phase_deg, color="C2", lw=1.6)
    axes[1].axhline(0, color="grey", ls=":", lw=0.8)
    for f in [f_20m, f_10m]:
        axes[1].axvline(f, color="C3", ls="--", lw=0.8, alpha=0.7)
    axes[1].scatter(f_design, np.degrees(np.angle(z_design)), color="C1", zorder=5, s=40)
    axes[1].set_ylabel("∠Z_a  (deg)")
    axes[1].set_ylim(-95, 95)
    axes[1].grid(True, which="both", alpha=0.3)

    # ----- Re / Im decomposition -----
    axes[2].semilogx(f_mhz, z.real, color="C0", lw=1.4, label="Re Z_a")
    axes[2].semilogx(f_mhz, z.imag, color="C3", lw=1.4, label="Im Z_a")
    axes[2].axhline(50, color="grey", ls=":", lw=0.8, label="50 Ω")
    axes[2].axhline(0, color="black", lw=0.4)
    for f in [f_20m, f_10m]:
        axes[2].axvline(f, color="C3", ls="--", lw=0.8, alpha=0.5)
    axes[2].set_yscale("symlog", linthresh=50)
    axes[2].set_xlabel("frequency  (MHz)")
    axes[2].set_ylabel("Re/Im Z_a  (Ω, symlog)")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = "antenna_bode.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # Print numeric summary
    print()
    print(f"{'f (MHz)':>10} {'|Z|':>10} {'∠Z (deg)':>10} {'Re Z':>10} {'Im Z':>10}")
    print("-" * 56)
    for f, zd in zip(f_design, z_design):
        print(
            f"{f:>10.3f} {abs(zd):>10.2f} {np.degrees(np.angle(zd)):>+10.2f} "
            f"{zd.real:>+10.2f} {zd.imag:>+10.2f}"
        )
    print()
    print(
        f"element 1 (20 m): leg={el20.leg_length_m:.3f} m  f0={el20.f0_hz / 1e6:.3f} MHz  Q={el20.q}  R_rad={el20.r_rad} Ω"
    )
    print(
        f"element 2 (10 m): leg={el10.leg_length_m:.3f} m  f0={el10.f0_hz / 1e6:.3f} MHz  Q={el10.q}  R_rad={el10.r_rad} Ω"
    )
    print(f"Z_a(s) order: num deg = {len(num) - 1}, den deg = {len(den) - 1}")


if __name__ == "__main__":
    main()
