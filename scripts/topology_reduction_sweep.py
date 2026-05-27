#!/usr/bin/env python
"""Topology reduction sweep — no resistor, find the smallest LC matcher
that still achieves a reasonable per-band-tuned match.

For each candidate topology (N components, all L/C, no R):
  - try each position as the per-band-variable component
  - shared values + per-band variable + joint antenna lengths
  - minimize worst-case SWR across the 4 design bands
  - (no loss budget — no R means no dissipation)

Report the best variable position and the achieved SWR for each topology.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    evaluate_rational,
    fan_dipole_impedance,
)


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "topology_reduction_sweep.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("topo")


F_DESIGN_MHZ = np.array([18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["17m", "15m", "12m", "10m"]
Z0 = 50.0
R_RAD = 50.0
F_LOW_BOUNDS = (17.0, 22.0)
F_HIGH_BOUNDS = (24.0, 29.0)

# All candidate topologies (no R). Source-to-load order.
TOPOLOGIES = [
    # N=4
    ("N=4: LπLπ", ["L_shunt", "C_series", "L_shunt", "C_series"]),
    ("N=4: πLπL", ["C_shunt", "L_series", "C_shunt", "L_series"]),
    # N=3
    ("N=3: LπL  (T-with-shunts)", ["L_shunt", "C_series", "L_shunt"]),
    ("N=3: πLπ  (T-with-series)", ["C_series", "L_shunt", "C_series"]),
    ("N=3: πCπ-style (C-shunt T)", ["C_shunt", "L_series", "C_shunt"]),
    ("N=3: LCL-style (L-series T)", ["L_series", "C_shunt", "L_series"]),
    # N=2 — classic L-matches
    ("N=2: L-match (Lsh + Cse)", ["L_shunt", "C_series"]),
    ("N=2: L-match (Cse + Lsh)", ["C_series", "L_shunt"]),
    ("N=2: L-match (Csh + Lse)", ["C_shunt", "L_series"]),
    ("N=2: L-match (Lse + Csh)", ["L_series", "C_shunt"]),
]


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def section_abcd_at(kind: str, val: float, omega: float) -> np.ndarray:
    if kind == "L_series":
        return np.array([[1, 1j * omega * val], [0, 1]], dtype=complex)
    if kind == "C_series":
        return np.array([[1, 1.0 / (1j * omega * val)], [0, 1]], dtype=complex)
    if kind == "L_shunt":
        return np.array([[1, 0], [1.0 / (1j * omega * val), 1]], dtype=complex)
    if kind == "C_shunt":
        return np.array([[1, 0], [1j * omega * val, 1]], dtype=complex)
    raise ValueError(kind)


def cascade_at(kinds, values, omega):
    M = np.eye(2, dtype=complex)
    for k, v in zip(kinds, values, strict=True):
        M = M @ section_abcd_at(k, v, omega)
    return M


def z_load_at(omega, f_low, f_high):
    el_lo = DipoleElement(leg_length_m=leg(f_low), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(f_high), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return complex(evaluate_rational(num, den, np.array([1j * omega]))[0])


def evaluate_band_swr(kinds, shared_log, var_log, var_pos, f_low, f_high) -> np.ndarray:
    """Compute SWR for each of the 4 design bands with shared + per-band var values."""
    n_k = len(kinds)
    nonvar = [i for i in range(n_k) if i != var_pos]
    swrs = np.zeros(len(F_DESIGN_MHZ))
    for b, f_mhz in enumerate(F_DESIGN_MHZ):
        omega = 2 * np.pi * f_mhz * 1e6
        full_vals = np.empty(n_k)
        for j, pos in enumerate(nonvar):
            full_vals[pos] = np.exp(shared_log[j])
        full_vals[var_pos] = np.exp(var_log[b])
        z_l = z_load_at(omega, f_low, f_high)
        M = cascade_at(kinds, full_vals, omega)
        z_in = (M[0, 0] * z_l + M[0, 1]) / (M[1, 0] * z_l + M[1, 1])
        g_in = (z_in - Z0) / (z_in + Z0)
        abs_g = min(abs(g_in), 0.9999)
        swrs[b] = (1 + abs_g) / (1 - abs_g)
    return swrs


def bounds_for(kinds, var_pos):
    nonvar = [i for i in range(len(kinds)) if i != var_pos]
    shared_b = []
    for pos in nonvar:
        k = kinds[pos]
        if k.startswith("L"):
            shared_b.append((np.log(1e-10), np.log(1e-2)))
        else:
            shared_b.append((np.log(1e-15), np.log(1e-5)))
    var_k = kinds[var_pos]
    if var_k.startswith("L"):
        var_b_each = (np.log(1e-10), np.log(1e-2))
    else:
        var_b_each = (np.log(1e-15), np.log(1e-5))
    return shared_b + [var_b_each] * len(F_DESIGN_MHZ) + [F_LOW_BOUNDS, F_HIGH_BOUNDS]


def initial_x(kinds, var_pos, omega_center):
    nonvar = [i for i in range(len(kinds)) if i != var_pos]
    shared = []
    for pos in nonvar:
        k = kinds[pos]
        if k.startswith("L"):
            shared.append(Z0 / omega_center)
        else:
            shared.append(1.0 / (Z0 * omega_center))
    var_k = kinds[var_pos]
    if var_k.startswith("L"):
        var_init = [Z0 / omega_center] * len(F_DESIGN_MHZ)
    else:
        var_init = [1.0 / (Z0 * omega_center)] * len(F_DESIGN_MHZ)
    return np.concatenate([np.log(shared), np.log(var_init), [19.5, 26.5]])


def unpack(x, n_k, var_pos):
    n_shared = n_k - 1
    n_bands = len(F_DESIGN_MHZ)
    shared = x[:n_shared]
    variable = x[n_shared : n_shared + n_bands]
    f_low = float(x[n_shared + n_bands])
    f_high = float(x[n_shared + n_bands + 1])
    return shared, variable, f_low, f_high


def objective(x, kinds, var_pos):
    n_k = len(kinds)
    shared, variable, f_low, f_high = unpack(x, n_k, var_pos)
    swrs = evaluate_band_swr(kinds, shared, variable, var_pos, f_low, f_high)
    return float(swrs.max())


def solve_one(kinds, var_pos, n_restarts=15):
    omega_center = float(np.exp(np.mean(np.log(2 * np.pi * F_DESIGN_MHZ * 1e6))))
    rng = np.random.default_rng(var_pos)
    n_k = len(kinds)
    x0 = initial_x(kinds, var_pos, omega_center)
    bnds = bounds_for(kinds, var_pos)

    starts = [x0.copy()]
    for _ in range(n_restarts):
        s = x0.copy()
        s[: n_k - 1] += rng.normal(0.0, 0.5, size=n_k - 1)
        s[n_k - 1 : n_k - 1 + len(F_DESIGN_MHZ)] += rng.normal(0.0, 0.5, size=len(F_DESIGN_MHZ))
        s[-2] += rng.normal(0.0, 1.0)
        s[-1] += rng.normal(0.0, 1.0)
        starts.append(s)
    # A few corner starts on antenna lengths
    for f_lo in [18.0, 20.0, 21.5]:
        for f_hi in [25.0, 26.5, 28.0]:
            s = x0.copy()
            s[-2] = f_lo
            s[-1] = f_hi
            starts.append(s)

    best_x = None
    best_score = np.inf
    for s in starts:
        s_clip = np.array([np.clip(s[i], b[0] + 0.01, b[1] - 0.01) for i, b in enumerate(bnds)])
        try:
            res = minimize(
                objective,
                s_clip,
                args=(kinds, var_pos),
                method="L-BFGS-B",
                bounds=bnds,
                options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8},
            )
            if res.fun < best_score:
                best_score = float(res.fun)
                best_x = res.x
        except Exception:  # noqa: BLE001
            continue

    shared, variable, f_low, f_high = unpack(best_x, n_k, var_pos)
    swrs = evaluate_band_swr(kinds, shared, variable, var_pos, f_low, f_high)
    return {
        "kinds": kinds,
        "var_pos": var_pos,
        "shared_log": shared,
        "variable_log": variable,
        "f_low": f_low,
        "f_high": f_high,
        "swrs": swrs,
        "worst_swr": float(swrs.max()),
    }


def fmt(kind: str, val: float) -> str:
    if kind.startswith("L"):
        return f"{val * 1e6:.3g}µH"
    return f"{val * 1e12:.3g}pF"


def main():
    log.info("=" * 78)
    log.info("Topology reduction sweep — no resistor, find min worst-SWR per topology")
    log.info("for each component being per-band variable")
    log.info("=" * 78)

    overall_best = []
    for topo_label, kinds in TOPOLOGIES:
        log.info("")
        log.info(">>> %s : %s", topo_label, " - ".join(kinds))
        t0 = time.time()
        best_for_topo = None
        for pos in range(len(kinds)):
            r = solve_one(kinds, pos)
            log.info(
                "  var_pos=%d (%s)  worst SWR=%.3f  per-band: %s  f=(%.2f, %.2f)",
                pos,
                kinds[pos],
                r["worst_swr"],
                "/".join(f"{s:.2f}" for s in r["swrs"]),
                r["f_low"],
                r["f_high"],
            )
            if best_for_topo is None or r["worst_swr"] < best_for_topo["worst_swr"]:
                best_for_topo = r
        log.info(
            "  best for %s: var_pos=%d (%s), SWR=%.3f, in %.1fs",
            topo_label,
            best_for_topo["var_pos"],
            kinds[best_for_topo["var_pos"]],
            best_for_topo["worst_swr"],
            time.time() - t0,
        )

        # Decode variable values
        nonvar = [i for i in range(len(kinds)) if i != best_for_topo["var_pos"]]
        shared_vals = np.exp(best_for_topo["shared_log"])
        var_vals = np.exp(best_for_topo["variable_log"])
        log.info(
            "    shared: %s",
            ", ".join(f"{kinds[p]}={fmt(kinds[p], shared_vals[j])}" for j, p in enumerate(nonvar)),
        )
        log.info(
            "    variable %s per band (17m/15m/12m/10m): %s",
            kinds[best_for_topo["var_pos"]],
            " / ".join(fmt(kinds[best_for_topo["var_pos"]], v) for v in var_vals),
        )
        overall_best.append(
            {
                "label": topo_label,
                "kinds": kinds,
                "n": len(kinds),
                **best_for_topo,
            }
        )

    log.info("")
    log.info("=" * 78)
    log.info("SUMMARY — best per topology")
    log.info("=" * 78)
    log.info("  %-40s  %3s  %12s  %10s", "topology", "N", "best var pos", "worst SWR")
    for b in overall_best:
        log.info(
            "  %-40s  %3d  %12s  %10.3f",
            b["label"],
            b["n"],
            b["kinds"][b["var_pos"]],
            b["worst_swr"],
        )

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [b["label"] for b in overall_best]
    swrs = [b["worst_swr"] for b in overall_best]
    colors = ["C0" if b["n"] == 4 else ("C2" if b["n"] == 3 else "C3") for b in overall_best]
    ax.barh(range(len(overall_best)), swrs, color=colors)
    for i, (lbl, swr) in enumerate(zip(labels, swrs)):
        ax.text(swr + 0.05, i, f"{swr:.2f}", va="center", fontsize=10)
    ax.set_yticks(range(len(overall_best)))
    ax.set_yticklabels(labels)
    ax.axvline(1.0, color="grey", ls=":", lw=0.5)
    ax.axvline(1.5, color="grey", ls=":", lw=0.5)
    ax.axvline(2.0, color="grey", ls="--", lw=0.7)
    ax.set_xlabel("worst-band SWR (4-band, no R, 1 component variable)")
    ax.set_title("Smallest LC matcher with one per-band-variable component (lossless)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("topology_reduction_sweep.png", dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote topology_reduction_sweep.png")


if __name__ == "__main__":
    main()
