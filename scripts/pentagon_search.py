#!/usr/bin/env python
"""Search the best pentagon ordering for a 5-element fan dipole.

Five wires sit at the vertices of a regular pentagon with side ~15 cm,
so each wire has 2 'adjacent' neighbors (closer, k_adj) and 2
'diagonal' neighbors (farther, k_diag). The assignment of bands to
vertices is a free choice; we want the ordering that minimizes the
worst-case post-retune SWR.

Physical hypothesis: the residual SWR on band i is dominated by
coupling from bands j whose currents are large at f_i, i.e. bands
adjacent in frequency. If those bands are also adjacent at pentagon
vertices (stronger coupling), the perturbation maxes out. Best
ordering places each band's frequency-near neighbors at pentagon
DIAGONAL vertices, not adjacent ones.
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


def leg_for_f(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


# Inverted-V geometry: drops R_rad from ~73 (flat dipole, free space) to ~50 Ω
# at typical apex angle (~120-150°). User chose this so each band's resonance
# lands natively matched to a 50 Ω feedline.
R_RAD_INV_V = 50.0


def make_elements(legs: np.ndarray) -> list[DipoleElement]:
    return [DipoleElement(leg_length_m=L, r_rad=R_RAD_INV_V) for L in legs]


def swr_from_z(z: np.ndarray, z0: float = Z0) -> np.ndarray:
    g = (z - z0) / (z + z0)
    return (1 + np.abs(g)) / (1 - np.abs(g))


def pentagon_coupling_matrix(order: tuple[int, ...], k_adj: float, k_diag: float) -> np.ndarray:
    """5×5 coupling matrix for bands placed at pentagon vertices.

    `order[v]` = band index placed at vertex v ∈ {0..4}. Pentagon
    adjacencies are (0,1),(1,2),(2,3),(3,4),(4,0); diagonals are the
    other pairs. Returned matrix is indexed by BAND, not vertex.
    """
    adj_pairs = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    K = np.zeros((5, 5))
    for v in range(5):
        for u in range(5):
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            k = k_adj if pair in adj_pairs else k_diag
            i, j = order[v], order[u]
            K[i, j] = k
    return K


def swr_at_design(legs: np.ndarray, K: np.ndarray) -> np.ndarray:
    elements = make_elements(legs)
    z = fan_dipole_impedance_coupled(elements, K, W_DESIGN)
    return swr_from_z(z)


def retune(legs: np.ndarray, K: np.ndarray, n_passes: int = 5) -> np.ndarray:
    legs = legs.copy()
    for _p in range(n_passes):
        for i in range(len(legs)):

            def obj(L_i: float, i=i) -> float:
                trial = legs.copy()
                trial[i] = L_i
                return float(swr_at_design(trial, K)[i])

            L0 = legs[i]
            res = minimize_scalar(
                obj,
                bounds=(0.75 * L0, 1.25 * L0),
                method="bounded",
                options={"xatol": 1e-5},
            )
            legs[i] = res.x
    return legs


def canonicalize(order: tuple[int, ...]) -> tuple[int, ...]:
    """Reduce a pentagon ordering to its lexicographically smallest
    rotation/reflection. Cuts 120 -> 12 unique orderings."""
    n = len(order)
    rotations = [tuple(order[i:] + order[:i]) for i in range(n)]
    rotations += [tuple(reversed(r)) for r in rotations]
    return min(rotations)


def main() -> None:
    nominal_legs = np.array([leg_for_f(f) for f in F_DESIGN_MHZ])

    # Uncoupled baseline sanity check (R_rad = 50 → SWR should be ~1.0 at each band)
    K0 = np.zeros((5, 5))
    swr0 = swr_at_design(nominal_legs, K0)
    print(
        f"Uncoupled baseline (R_rad = {R_RAD_INV_V:.0f} Ω):  "
        + "  ".join(f"{BAND_LABEL[i]}:{swr0[i]:.2f}" for i in range(5))
    )
    print()

    # Coupling values: pentagon side ~15 cm, diagonal ~25 cm. Real coupling
    # coefficients aren't derivable from our behavioral Q=10 RLC model, so we
    # use values fit to the user's observation that one band sticks at SWR~2.5
    # while the rest reach ~1.3. Try a few regimes.
    regimes = [
        ("very light (0.05/0.025)", 0.05, 0.025),
        ("light      (0.08/0.04) ", 0.08, 0.04),
        ("medium     (0.12/0.06) ", 0.12, 0.06),
    ]

    for label, k_adj, k_diag in regimes:
        print("=" * 72)
        print(f"Coupling regime: {label}   (k_adj={k_adj}, k_diag={k_diag})")
        print("=" * 72)

        seen: set[tuple[int, ...]] = set()
        results: list[tuple[float, tuple[int, ...], np.ndarray]] = []
        for perm in permutations(range(5)):
            canon = canonicalize(perm)
            if canon in seen:
                continue
            seen.add(canon)
            K = pentagon_coupling_matrix(perm, k_adj, k_diag)
            tuned = retune(nominal_legs, K)
            swr = swr_at_design(tuned, K)
            results.append((float(swr.max()), perm, swr))

        results.sort(key=lambda x: x[0])

        print(f"\n{'rank':>4}  {'pentagon vertex order':>40}  {'max':>5}  {'SWR per band':>40}")
        print("-" * 100)
        for rank, (worst, perm, swr) in enumerate(results):
            order_str = " - ".join(BAND_LABEL[i] for i in perm)
            swr_str = " ".join(f"{BAND_LABEL[i]}:{swr[i]:.2f}" for i in range(5))
            tag = "  <-- BEST" if rank == 0 else ("  <-- WORST" if rank == len(results) - 1 else "")
            print(f"  {rank + 1:>2}  {order_str:>40}  {worst:>5.2f}  {swr_str}{tag}")

        # Explain why the best ordering is best
        best_perm = results[0][1]
        print(f"\nBest ordering: {' - '.join(BAND_LABEL[i] for i in best_perm)}")
        print("Vertex-adjacency table (each band & its 2 adjacent pentagon neighbors):")
        adj_pairs = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
        for v in range(5):
            band = best_perm[v]
            neighbors_v = [u for u in range(5) if (min(u, v), max(u, v)) in adj_pairs]
            adj_bands = [best_perm[u] for u in neighbors_v]
            df = [F_DESIGN_MHZ[b] - F_DESIGN_MHZ[band] for b in adj_bands]
            print(
                f"  vertex {v}: {BAND_LABEL[band]:>4s} @ {F_DESIGN_MHZ[band]:.2f} MHz "
                f"-> adj: {[BAND_LABEL[b] for b in adj_bands]} "
                f"(Δf = {df[0]:+.2f}, {df[1]:+.2f} MHz)"
            )
        print()


if __name__ == "__main__":
    main()
