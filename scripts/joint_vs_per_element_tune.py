#!/usr/bin/env python
"""Test joint-length and decouple-the-neighbors strategies vs simple
per-element coordinate descent, on the hybridized-15m regime.

In the inductive-heavy coupling regime, the 15m element loses its
independent resonance because both adjacent bands (17m, 12m) hybridize
with it. Single-element retuning leaves 15m stuck. This script shows
whether joint optimization or deliberate neighbor-detuning recovers it.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    fan_dipole_impedance_coupled,
)


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
W_DESIGN = 2 * np.pi * F_DESIGN_MHZ * 1e6
Z0 = 50.0
R_RAD = 50.0


def leg_for_f(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def make_elements(legs: np.ndarray) -> list[DipoleElement]:
    return [DipoleElement(leg_length_m=L, r_rad=R_RAD) for L in legs]


def swr_from_z(z: np.ndarray) -> np.ndarray:
    g = (z - Z0) / (z + Z0)
    return (1 + np.abs(g)) / (1 - np.abs(g))


def z_a_at(legs: np.ndarray, KL: np.ndarray, KR: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    return fan_dipole_impedance_coupled(make_elements(legs), KL, omegas, coupling_R=KR)


def swr_at_design(legs, KL, KR):
    return swr_from_z(z_a_at(legs, KL, KR, W_DESIGN))


def per_element_tune(legs, KL, KR, n_passes=6):
    """Strategy A: classic iterative — each element tunes to its own band."""
    legs = legs.copy()
    for _ in range(n_passes):
        for i in range(len(legs)):

            def obj(L_i, i=i):
                t = legs.copy()
                t[i] = L_i
                return float(swr_at_design(t, KL, KR)[i])

            L0 = legs[i]
            res = minimize_scalar(
                obj, bounds=(0.70 * L0, 1.30 * L0), method="bounded", options={"xatol": 1e-5}
            )
            legs[i] = res.x
    return legs


def joint_tune(legs0, KL, KR):
    """Strategy B: minimize worst-case SWR across all 5 design bands
    jointly over all 5 lengths."""

    def obj(legs):
        return float(np.max(swr_at_design(legs, KL, KR)))

    bounds = [(0.70 * L0, 1.30 * L0) for L0 in legs0]
    # Multi-restart: try the nominal point and a few perturbations
    best = None
    rng = np.random.default_rng(0)
    for trial in range(8):
        if trial == 0:
            x0 = legs0.copy()
        else:
            x0 = legs0 * (1.0 + 0.05 * rng.normal(size=5))
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(
            obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200, "ftol": 1e-10}
        )
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def decouple_neighbors(legs0, KL, KR):
    """Strategy C: deliberately shift 17m and 12m AWAY from 15m to free
    its resonance. Sweep the relative frequency offset; for each shift,
    re-tune the other 3 elements to their own design freqs."""
    f0 = F_DESIGN_MHZ.copy()
    best = (np.inf, None, None)
    # Try shifting 17m down by Δ_low and 12m up by Δ_high
    for d_low in np.linspace(0.0, 1.0, 11):  # MHz to shift 17m DOWN
        for d_high in np.linspace(0.0, 1.0, 11):  # MHz to shift 12m UP
            f_shifted = f0.copy()
            f_shifted[1] -= d_low  # 17m
            f_shifted[3] += d_high  # 12m
            # Start each element at λ/4 for its (possibly shifted) target
            legs_start = np.array([leg_for_f(f) for f in f_shifted])
            # Per-element retune at the SHIFTED frequencies
            global W_DESIGN_TUNE
            W_DESIGN_TUNE = 2 * np.pi * f_shifted * 1e6
            tuned = per_element_tune_at(legs_start, KL, KR, W_DESIGN_TUNE)
            # But evaluate worst-case SWR at the ORIGINAL design freqs
            swr_orig = swr_from_z(z_a_at(tuned, KL, KR, W_DESIGN))
            worst = float(np.max(swr_orig))
            if worst < best[0]:
                best = (worst, (d_low, d_high), tuned)
    return best


def per_element_tune_at(legs, KL, KR, w_targets, n_passes=4):
    legs = legs.copy()
    for _ in range(n_passes):
        for i in range(len(legs)):

            def obj(L_i, i=i):
                t = legs.copy()
                t[i] = L_i
                z = z_a_at(t, KL, KR, w_targets)
                return float(swr_from_z(z)[i])

            L0 = legs[i]
            res = minimize_scalar(
                obj, bounds=(0.70 * L0, 1.30 * L0), method="bounded", options={"xatol": 1e-5}
            )
            legs[i] = res.x
    return legs


def main() -> None:
    # Best-ordering pentagon coupling matrices for the inductive-heavy regime
    # (the one that reproduces the 15m-stuck observation).
    adj = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    # Map band index i to pentagon vertex per the best ordering: 20-12-15-17-10
    # Bands at vertices: V0=20m(0), V1=12m(3), V2=15m(2), V3=17m(1), V4=10m(4)
    order = (0, 3, 2, 1, 4)

    # Inductive coupling matrix (k_L on adj/diag pairs)
    KL = np.zeros((5, 5))
    KR = np.zeros((5, 5))
    kL_adj, kL_diag = 0.10, 0.05
    aR_adj, aR_diag = 0.75, 0.70
    for v in range(5):
        for u in range(5):
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            kL = kL_adj if pair in adj else kL_diag
            aR = aR_adj if pair in adj else aR_diag
            KL[order[v], order[u]] = kL
            KR[order[v], order[u]] = aR

    nominal_legs = np.array([leg_for_f(f) for f in F_DESIGN_MHZ])

    print("=" * 78)
    print("Inductive-heavy regime (k_L adj/diag=0.20/0.10, α_R adj/diag=0.10/0.05)")
    print("Best ordering: 20m-12m-15m-17m-10m around the pentagon")
    print("=" * 78)
    swr0 = swr_at_design(nominal_legs, KL, KR)
    print("\n(0) Nominal lengths, no tuning:")
    print("    SWR: " + "  ".join(f"{BAND_LABEL[i]}:{swr0[i]:.2f}" for i in range(5)))

    legs_A = per_element_tune(nominal_legs, KL, KR)
    swr_A = swr_at_design(legs_A, KL, KR)
    print("\n(A) Strategy A — per-element iterative tuning (what you did physically):")
    print("    SWR: " + "  ".join(f"{BAND_LABEL[i]}:{swr_A[i]:.2f}" for i in range(5)))
    print(
        "    length shifts (cm): "
        + "  ".join(f"{(L - L0) * 100:+.2f}" for L, L0 in zip(legs_A, nominal_legs))
    )

    legs_B = joint_tune(nominal_legs, KL, KR)
    swr_B = swr_at_design(legs_B, KL, KR)
    print("\n(B) Strategy B — JOINT length optimization (minimax SWR over all 5 bands):")
    print("    SWR: " + "  ".join(f"{BAND_LABEL[i]}:{swr_B[i]:.2f}" for i in range(5)))
    print(
        "    length shifts (cm): "
        + "  ".join(f"{(L - L0) * 100:+.2f}" for L, L0 in zip(legs_B, nominal_legs))
    )

    worst_C, shifts_C, legs_C = decouple_neighbors(nominal_legs, KL, KR)
    swr_C = swr_at_design(legs_C, KL, KR)
    print("\n(C) Strategy C — DELIBERATELY detune 17m and 12m to free 15m:")
    print(f"    best shifts: 17m -{shifts_C[0]:.2f} MHz, 12m +{shifts_C[1]:.2f} MHz")
    print(
        "    SWR at original 5 design freqs: "
        + "  ".join(f"{BAND_LABEL[i]}:{swr_C[i]:.2f}" for i in range(5))
    )
    print(
        "    length shifts (cm): "
        + "  ".join(f"{(L - L0) * 100:+.2f}" for L, L0 in zip(legs_C, nominal_legs))
    )

    # Plot SWR sweep across the HF band for each strategy
    f_sweep = np.linspace(13e6, 30e6, 2000)
    w_sweep = 2 * np.pi * f_sweep

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, legs, ls in [
        ("nominal lengths", nominal_legs, "--"),
        ("A: per-element", legs_A, "-"),
        ("B: joint", legs_B, "-"),
        ("C: decouple", legs_C, "-"),
    ]:
        z = z_a_at(legs, KL, KR, w_sweep)
        ax.semilogy(f_sweep / 1e6, swr_from_z(z), lw=1.6, label=label, linestyle=ls)
    for f, lbl in zip(F_DESIGN_MHZ, BAND_LABEL):
        ax.axvline(f, color="grey", ls=":", lw=0.5, alpha=0.6)
        ax.text(
            f,
            0.95,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="grey",
        )
    ax.axhline(2.0, color="grey", ls=":", lw=0.8)
    ax.axhline(1.5, color="grey", ls=":", lw=0.5)
    ax.set_ylim(1, 20)
    ax.set_xlabel("frequency  (MHz)")
    ax.set_ylabel("SWR  (log)")
    ax.set_title("Tuning strategies on the 15m-hybridized 5-element fan dipole")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("tuning_strategies.png", dpi=140, bbox_inches="tight")
    print("\nwrote tuning_strategies.png")


if __name__ == "__main__":
    main()
