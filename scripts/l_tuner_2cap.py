#!/usr/bin/env python
"""L-tuner with only 2 distinct C_series values (alternating pairing).

Pair 1: 17m + 12m use C_A
Pair 2: 15m + 10m use C_B

Variables: L_shunt (fixed) + C_A + C_B + f_low + f_high = 5.
Goal: minimize worst-case SWR across 4 bands, lossless (no R).

Also runs the OTHER 2 pairings as a sanity check:
  - (17m, 15m) low pair vs (12m, 10m) high pair
  - (17m, 10m) outer pair vs (15m, 12m) inner pair
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
LOG_PATH = os.path.join(LOG_DIR, "l_tuner_2cap.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("2cap")


F_DESIGN_MHZ = np.array([18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["17m", "15m", "12m", "10m"]
Z0 = 50.0
R_RAD = 50.0
F_LOW_BOUNDS = (17.0, 22.0)
F_HIGH_BOUNDS = (24.0, 29.0)

# Three candidate pairings: each is a 4-tuple of "which cap (A or B) per band"
# Band order: 17m, 15m, 12m, 10m
PAIRINGS = [
    ("alternating (17m+12m | 15m+10m)", [0, 1, 0, 1]),  # user's request
    ("low/high     (17m+15m | 12m+10m)", [0, 0, 1, 1]),
    ("outer/inner  (17m+10m | 15m+12m)", [0, 1, 1, 0]),
]


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def z_load_at(omega: float, f_low: float, f_high: float) -> complex:
    el_lo = DipoleElement(leg_length_m=leg(f_low), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(f_high), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return complex(evaluate_rational(num, den, np.array([1j * omega]))[0])


def swr_band(
    l_shunt: float, c_series: float, f_design_mhz: float, f_low: float, f_high: float
) -> float:
    """SWR at one band freq for the L-tuner (C_series then L_shunt)."""
    omega = 2 * np.pi * f_design_mhz * 1e6
    z_l = z_load_at(omega, f_low, f_high)
    # ABCD for C_series then L_shunt cascade:
    z_c = 1.0 / (1j * omega * c_series)
    y_l = 1.0 / (1j * omega * l_shunt)
    # M = M_C @ M_L = [[1+z_c*y_l, z_c],[y_l, 1]]
    A = 1.0 + z_c * y_l
    B = z_c
    C = y_l
    D = 1.0
    z_in = (A * z_l + B) / (C * z_l + D)
    g = (z_in - Z0) / (z_in + Z0)
    abs_g = min(abs(g), 0.9999)
    return (1 + abs_g) / (1 - abs_g)


def objective(x, pairing):
    log_l, log_ca, log_cb, f_low, f_high = x
    l_shunt = np.exp(log_l)
    c_vals = [np.exp(log_ca), np.exp(log_cb)]
    swrs = [swr_band(l_shunt, c_vals[pairing[b]], F_DESIGN_MHZ[b], f_low, f_high) for b in range(4)]
    return float(max(swrs))


def solve(pairing, n_restarts=40):
    rng = np.random.default_rng(hash(tuple(pairing)) & 0xFFFFFFFF)
    omega_center = 2 * np.pi * 22e6
    x0 = np.array(
        [
            np.log(Z0 / omega_center),  # L_shunt ~ 0.36 µH
            np.log(1.0 / (Z0 * omega_center)),  # C_A ~ 145 pF
            np.log(1.0 / (Z0 * omega_center)),  # C_B
            19.5,
            26.5,
        ]
    )
    bnds = [
        (np.log(1e-10), np.log(1e-2)),  # L
        (np.log(1e-15), np.log(1e-5)),  # C_A
        (np.log(1e-15), np.log(1e-5)),  # C_B
        F_LOW_BOUNDS,
        F_HIGH_BOUNDS,
    ]

    starts = [x0.copy()]
    for _ in range(n_restarts):
        s = x0.copy()
        s[0] += rng.normal(0.0, 1.0)
        s[1] += rng.normal(0.0, 2.0)
        s[2] += rng.normal(0.0, 2.0)
        s[3] += rng.normal(0.0, 1.5)
        s[4] += rng.normal(0.0, 1.5)
        starts.append(s)
    # Some structured starts: try C values from the 4-cap optimum
    seed_c_vals_pF = [50.2, 528, 72.2, 330]
    for c_a_pF in seed_c_vals_pF:
        for c_b_pF in seed_c_vals_pF:
            s = x0.copy()
            s[1] = np.log(c_a_pF * 1e-12)
            s[2] = np.log(c_b_pF * 1e-12)
            starts.append(s)

    best_x = None
    best_score = np.inf
    for s in starts:
        s_clip = np.array([np.clip(s[i], b[0] + 0.01, b[1] - 0.01) for i, b in enumerate(bnds)])
        try:
            res = minimize(
                objective,
                s_clip,
                args=(pairing,),
                method="L-BFGS-B",
                bounds=bnds,
                options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-9},
            )
            if res.fun < best_score:
                best_score = float(res.fun)
                best_x = res.x
        except Exception:  # noqa: BLE001
            continue

    log_l, log_ca, log_cb, f_low, f_high = best_x
    l_shunt = np.exp(log_l)
    c_a = np.exp(log_ca)
    c_b = np.exp(log_cb)
    swrs = [
        swr_band(l_shunt, [c_a, c_b][pairing[b]], F_DESIGN_MHZ[b], f_low, f_high) for b in range(4)
    ]
    return {
        "l_shunt": l_shunt,
        "c_a": c_a,
        "c_b": c_b,
        "f_low": f_low,
        "f_high": f_high,
        "swrs": np.array(swrs),
        "worst_swr": float(max(swrs)),
    }


def main():
    log.info("=" * 78)
    log.info("L-tuner with TWO distinct C values across 4 bands")
    log.info("=" * 78)

    results = []
    for label, pairing in PAIRINGS:
        log.info("")
        log.info(">>> %s    pairing=%s", label, pairing)
        t0 = time.time()
        r = solve(pairing)
        log.info("    DONE in %.1fs", time.time() - t0)
        log.info("    L_shunt = %.3f µH", r["l_shunt"] * 1e6)
        log.info(
            "    C_A     = %.1f pF (used on bands: %s)",
            r["c_a"] * 1e12,
            [BAND_LABEL[b] for b in range(4) if pairing[b] == 0],
        )
        log.info(
            "    C_B     = %.1f pF (used on bands: %s)",
            r["c_b"] * 1e12,
            [BAND_LABEL[b] for b in range(4) if pairing[b] == 1],
        )
        log.info("    f_low=%.3f MHz, f_high=%.3f MHz", r["f_low"], r["f_high"])
        log.info(
            "    SWR per band (17m/15m/12m/10m): %s", " / ".join(f"{s:.3f}" for s in r["swrs"])
        )
        log.info("    worst SWR = %.3f", r["worst_swr"])
        results.append({"label": label, "pairing": pairing, **r})

    log.info("")
    log.info("=" * 78)
    log.info("SUMMARY — best pairing")
    log.info("=" * 78)
    results.sort(key=lambda r: r["worst_swr"])
    for r in results:
        log.info("  %-40s  worst SWR = %.3f", r["label"], r["worst_swr"])

    best = results[0]
    log.info("")
    log.info("WINNER: %s", best["label"])
    log.info(
        "  L_shunt=%.3f µH, C_A=%.0f pF, C_B=%.0f pF",
        best["l_shunt"] * 1e6,
        best["c_a"] * 1e12,
        best["c_b"] * 1e12,
    )
    log.info("  Antenna: f_low=%.3f, f_high=%.3f MHz", best["f_low"], best["f_high"])
    log.info("  Per band SWR: %s", " / ".join(f"{s:.3f}" for s in best["swrs"]))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(results))
    swrs = [r["worst_swr"] for r in results]
    ax.bar(xs, swrs, color=["C2", "C1", "C0"])
    for x, swr in zip(xs, swrs):
        ax.text(x, swr + 0.02, f"{swr:.2f}", ha="center", fontsize=11)
    ax.axhline(1.147, color="grey", ls="--", lw=0.8, label="4-cap (per-band) baseline SWR 1.15")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in results], rotation=10, fontsize=9)
    ax.set_ylabel("worst-band SWR")
    ax.set_title("L-tuner with 2 distinct C values — pairing comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(1.0, max(swrs) * 1.1 + 0.5)
    fig.tight_layout()
    fig.savefig("l_tuner_2cap.png", dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote l_tuner_2cap.png")


if __name__ == "__main__":
    main()
