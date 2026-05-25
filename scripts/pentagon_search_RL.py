#!/usr/bin/env python
"""Pentagon ordering search with BOTH resistive and inductive coupling.

The 5 wires are parallel (held by multiple pentagonal spacers along
their length), so mutual radiation resistance R_ij ≈ R_self for every
pair (d/λ is tiny: 0.007-0.024 across the bands). Inductive coupling
varies more with spacing (logarithmically) but is secondary.

Resistive coupling is the dominant mechanism in this geometry.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.optimize import minimize_scalar

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
R_RAD_INV_V = 50.0


def leg_for_f(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def make_elements(legs: np.ndarray) -> list[DipoleElement]:
    return [DipoleElement(leg_length_m=L, r_rad=R_RAD_INV_V) for L in legs]


def swr_from_z(z: np.ndarray, z0: float = Z0) -> np.ndarray:
    g = (z - z0) / (z + z0)
    return (1 + np.abs(g)) / (1 - np.abs(g))


def pentagon_matrices(
    order: tuple[int, ...], kL_adj: float, kL_diag: float, aR_adj: float, aR_diag: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build (K_L, K_R) for pentagon ordering.

    `order[v]` = band index at vertex v. Pentagon edges are
    (0,1),(1,2),(2,3),(3,4),(0,4); the other 5 pairs are diagonals.
    """
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


def swr_at_design(legs: np.ndarray, KL: np.ndarray, KR: np.ndarray) -> np.ndarray:
    z = fan_dipole_impedance_coupled(make_elements(legs), KL, W_DESIGN, coupling_R=KR)
    return swr_from_z(z)


def retune(legs: np.ndarray, KL: np.ndarray, KR: np.ndarray, n_passes: int = 5) -> np.ndarray:
    legs = legs.copy()
    for _p in range(n_passes):
        for i in range(len(legs)):

            def obj(L_i: float, i=i) -> float:
                trial = legs.copy()
                trial[i] = L_i
                return float(swr_at_design(trial, KL, KR)[i])

            L0 = legs[i]
            res = minimize_scalar(
                obj,
                bounds=(0.70 * L0, 1.30 * L0),
                method="bounded",
                options={"xatol": 1e-5},
            )
            legs[i] = res.x
    return legs


def canonicalize(order: tuple[int, ...]) -> tuple[int, ...]:
    n = len(order)
    rots = [tuple(order[i:] + order[:i]) for i in range(n)]
    rots += [tuple(reversed(r)) for r in rots]
    return min(rots)


def main() -> None:
    nominal_legs = np.array([leg_for_f(f) for f in F_DESIGN_MHZ])

    # Sanity: uncoupled baseline
    K0 = np.zeros((5, 5))
    swr0 = swr_at_design(nominal_legs, K0, K0)
    print(
        f"Uncoupled baseline (R_rad={R_RAD_INV_V} Ω):  "
        + "  ".join(f"{BAND_LABEL[i]}:{swr0[i]:.2f}" for i in range(5))
    )

    # Three physically-motivated regimes:
    #
    #   - "resistive-only" tests the textbook prediction: at d/λ << 1,
    #     R_ij → R_self for every pair, and inductive M_ij is negligible
    #     relative to R. Both adj and diag share α_R near 1.
    #
    #   - "mixed" adds small inductive coupling that distinguishes adj/diag.
    #
    #   - "weak resistive + inductive" lets the inductive term matter more,
    #     more like a sparsely-spaced fan dipole.
    regimes = [
        # (label,                 kL_adj kL_diag aR_adj aR_diag)
        ("R only, very strong", 0.000, 0.000, 0.95, 0.92),
        ("R only, moderate   ", 0.000, 0.000, 0.80, 0.75),
        ("R only, light      ", 0.000, 0.000, 0.50, 0.45),
        ("mixed              ", 0.08, 0.04, 0.80, 0.75),
        ("inductive heavy    ", 0.20, 0.10, 0.10, 0.05),
    ]

    for label, kL_a, kL_d, aR_a, aR_d in regimes:
        print()
        print("=" * 78)
        print(f"Regime: {label}  (kL adj/diag={kL_a}/{kL_d},  α_R adj/diag={aR_a}/{aR_d})")
        print("=" * 78)

        seen = set()
        results = []
        for perm in permutations(range(5)):
            canon = canonicalize(perm)
            if canon in seen:
                continue
            seen.add(canon)
            KL, KR = pentagon_matrices(perm, kL_a, kL_d, aR_a, aR_d)
            tuned = retune(nominal_legs, KL, KR)
            swr = swr_at_design(tuned, KL, KR)
            results.append((float(swr.max()), perm, swr, int(np.argmax(swr))))

        results.sort(key=lambda x: x[0])

        print(
            f"\n{'rank':>4}  {'pentagon order':>32}  {'max':>5}  {'stuck':>6}  {'SWR per band':>40}"
        )
        print("-" * 100)
        for rank, (worst, perm, swr, stuck) in enumerate(results):
            order_str = " ".join(BAND_LABEL[i] for i in perm)
            swr_str = " ".join(f"{BAND_LABEL[i]}:{swr[i]:.2f}" for i in range(5))
            stuck_lbl = BAND_LABEL[stuck]
            tag = "  <-- BEST" if rank == 0 else ("  <-- WORST" if rank == len(results) - 1 else "")
            print(
                f"  {rank + 1:>2}  {order_str:>32}  {worst:>5.2f}  {stuck_lbl:>6}  {swr_str}{tag}"
            )


if __name__ == "__main__":
    main()
