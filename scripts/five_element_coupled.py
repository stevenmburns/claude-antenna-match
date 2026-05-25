#!/usr/bin/env python
"""5-element fan dipole WITH mutual coupling between elements.

Reproduces the observation that iterative re-tuning of individual
element lengths can drive 4 of 5 bands to acceptable SWR, but the
CENTER band remains stuck. The center element has two near neighbors
pulling on it (one inductive, one capacitive at f_3), while end
elements have only one neighbor — with one length d.o.f. per element
you can compensate one pull but not two.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    fan_dipole_impedance_coupled,
    geometric_coupling_matrix,
)


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
W_DESIGN = 2 * np.pi * F_DESIGN_MHZ * 1e6
Z0 = 50.0


def leg_for_f(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def swr_from_z(z: np.ndarray, z0: float = Z0) -> np.ndarray:
    g = (z - z0) / (z + z0)
    return (1 + np.abs(g)) / (1 - np.abs(g))


def swr_at_design(legs: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    elements = [DipoleElement(leg_length_m=L) for L in legs]
    z = fan_dipole_impedance_coupled(elements, coupling, W_DESIGN)
    return swr_from_z(z)


def retune_each_element(legs: np.ndarray, coupling: np.ndarray, n_passes: int = 6) -> np.ndarray:
    """Coordinate descent: for each element i, find leg length that
    minimizes SWR at design freq i, with all other lengths fixed.
    Repeat for n_passes sweeps over all elements."""
    legs = legs.copy()
    for _pass in range(n_passes):
        for i in range(len(legs)):

            def obj(L_i: float, i=i) -> float:
                trial = legs.copy()
                trial[i] = L_i
                return float(swr_at_design(trial, coupling)[i])

            # Bounded ±25% around current length (handles non-monotone landscapes)
            L0 = legs[i]
            res = minimize_scalar(
                obj,
                bounds=(0.75 * L0, 1.25 * L0),
                method="bounded",
                options={"xatol": 1e-5},
            )
            legs[i] = res.x
    return legs


def uniform_coupling_matrix(n: int, k: float) -> np.ndarray:
    """N×N matrix with every off-diagonal entry = k. Models the case
    where all elements meet at a common feedpoint with similar spacing,
    so pairwise coupling barely depends on index distance.
    """
    if not 0 <= k < 1.0 / max(n - 1, 1):
        raise ValueError(f"k must satisfy k(n-1) < 1, got k={k}, n={n}")
    mat = np.full((n, n), k)
    np.fill_diagonal(mat, 0.0)
    return mat


def main() -> None:
    legs_nominal = np.array([leg_for_f(f) for f in F_DESIGN_MHZ])

    # Two coupling models, three strengths each
    n = 5
    cases = [
        ("uncoupled", geometric_coupling_matrix(n, 0.00)),
        ("geom k=0.15", geometric_coupling_matrix(n, 0.15)),
        ("uniform k=0.10", uniform_coupling_matrix(n, 0.10)),
        ("uniform k=0.15", uniform_coupling_matrix(n, 0.15)),
        ("uniform k=0.20", uniform_coupling_matrix(n, 0.20)),
    ]

    f_sweep = np.linspace(10e6, 32e6, 800)
    w_sweep = 2 * np.pi * f_sweep

    print("\n=== Step 1: nominal lengths (each = λ/4 at its design freq), with coupling ===")
    print(f"  {'case':>14} {' '.join(f'{f:>6.2f}' for f in F_DESIGN_MHZ)}  (SWR)")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, K in cases:
        swr_design = swr_at_design(legs_nominal, K)
        print(f"  {label:>14} " + "  ".join(f"{s:>5.2f}" for s in swr_design))

        elements = [DipoleElement(leg_length_m=L) for L in legs_nominal]
        z_sweep = fan_dipole_impedance_coupled(elements, K, w_sweep)
        axes[0].semilogy(f_sweep / 1e6, swr_from_z(z_sweep), lw=1.4, label=label)
    axes[0].set_ylim(1, 100)
    axes[0].set_ylabel("SWR  (log)")
    axes[0].set_title("Nominal lengths (λ/4 at each design freq) — effect of coupling")
    for f in F_DESIGN_MHZ:
        axes[0].axvline(f, color="grey", ls="--", lw=0.5, alpha=0.5)
    axes[0].axhline(2.0, color="grey", ls=":", lw=0.8)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)

    print("\n=== Step 2: re-tune each element length to minimize its own band's SWR ===")
    print("            (coordinate descent, 6 passes)")
    print(f"  {'case':>14} {' '.join(f'{f:>6.2f}' for f in F_DESIGN_MHZ)}  (post-tune SWR)")
    for label, K in cases:
        legs_tuned = retune_each_element(legs_nominal, K)
        swr_design = swr_at_design(legs_tuned, K)
        print(f"  {label:>14} " + "  ".join(f"{s:>5.2f}" for s in swr_design))

        # Report length shifts in cm
        shift_cm = (legs_tuned - legs_nominal) * 100
        print(f"      length shifts (cm): {' '.join(f'{s:+.2f}' for s in shift_cm)}")

        elements_tuned = [DipoleElement(leg_length_m=L) for L in legs_tuned]
        z_sweep = fan_dipole_impedance_coupled(elements_tuned, K, w_sweep)
        axes[1].semilogy(f_sweep / 1e6, swr_from_z(z_sweep), lw=1.4, label=label)
    axes[1].set_ylim(1, 100)
    axes[1].set_ylabel("SWR  (log)")
    axes[1].set_xlabel("frequency  (MHz)")
    axes[1].set_title("After per-element retuning — center band typically stays high")
    for f in F_DESIGN_MHZ:
        axes[1].axvline(f, color="grey", ls="--", lw=0.5, alpha=0.5)
    axes[1].axhline(2.0, color="grey", ls=":", lw=0.8)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig("five_element_coupled.png", dpi=140, bbox_inches="tight")
    print("\nwrote five_element_coupled.png")


if __name__ == "__main__":
    main()
