#!/usr/bin/env python
"""Lossy-matcher complexity sweep on the 5-band 2-element fan dipole.

How much does worst-case SWR worsen as we use fewer matcher components?
Each topology is jointly optimized with antenna leg lengths at the
same 2 dB worst-band insertion-loss budget.

Topologies tested (source→load order):
  N=3 :  R_shunt - L_shunt - C_series                       (L-match + 1 R)
  N=4 :  R_shunt - L_shunt - C_series - L_shunt             (2 LC + 1 R)
  N=5 :  R_shunt - L_shunt - C_series - L_shunt - C_series  (2 LC pairs + 1 R)
  N=6 :  R_shunt - C_shunt - L_series - C_shunt - L_series - R_shunt
                                                            (2 LC pairs + 2 R)
  N=8 :  R_shunt - C_shunt - L_series - C_shunt - R_shunt - L_series - C_shunt - L_series
                                                            (the current optimum)

Antenna model: inverted-V (R_rad = 50 Ω); leg lengths jointly optimized.
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
LOG_PATH = os.path.join(LOG_DIR, "lossy_complexity_sweep.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sweep")


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0
LOSS_BUDGET_DB = 2.0
R_RAD = 50.0
F_LOW_BOUNDS = (14.30, 18.16)
F_HIGH_BOUNDS = (24.97, 28.47)
R_SHUNT_OFF = 1.0e6

TOPOLOGIES = [
    ("N=3", ["R_shunt", "L_shunt", "C_series"]),
    ("N=4", ["R_shunt", "L_shunt", "C_series", "L_shunt"]),
    ("N=5", ["R_shunt", "L_shunt", "C_series", "L_shunt", "C_series"]),
    ("N=6", ["R_shunt", "C_shunt", "L_series", "C_shunt", "L_series", "R_shunt"]),
    (
        "N=8",
        ["R_shunt", "C_shunt", "L_series", "C_shunt", "R_shunt", "L_series", "C_shunt", "L_series"],
    ),
]


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def section_abcd(kind: str, value: float, omegas: np.ndarray) -> np.ndarray:
    n = len(omegas)
    if kind == "L_series":
        ze = 1j * omegas * value
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "C_series":
        ze = 1.0 / (1j * omegas * value)
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "R_shunt":
        ye = np.full(n, 1.0 / value, dtype=complex)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "L_shunt":
        ye = 1.0 / (1j * omegas * value)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "C_shunt":
        ye = 1j * omegas * value
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    raise ValueError(kind)


def cascade_abcd(kinds, values, omegas):
    n = len(omegas)
    M = np.array([[np.ones(n), np.zeros(n)], [np.zeros(n), np.ones(n)]], dtype=complex)
    for kind, val in zip(kinds, values, strict=True):
        M = np.einsum("ilk,ljk->ijk", M, section_abcd(kind, val, omegas))
    return M


def z_load_of(f_low_mhz, f_high_mhz, omegas):
    el_lo = DipoleElement(leg_length_m=leg(f_low_mhz), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(f_high_mhz), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return evaluate_rational(num, den, 1j * omegas)


def evaluate(matcher_vals, kinds, f_low, f_high, omegas, r_source=Z0):
    z_load = z_load_of(f_low, f_high, omegas)
    M = cascade_abcd(kinds, matcher_vals, omegas)
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    z_in = (A * z_load + B) / (C * z_load + D)
    v_load_over_v1 = z_load / (A * z_load + B)
    v1_over_vs = z_in / (z_in + r_source)
    v_ratio = v1_over_vs * v_load_over_v1
    g_in = (z_in - r_source) / (z_in + r_source)
    abs_g = np.minimum(np.abs(g_in), 0.999)
    swr = (1 + abs_g) / (1 - abs_g)
    m_src = 4.0 * r_source * z_in.real / np.abs(z_in + r_source) ** 2
    r_l = z_load.real
    t_total = 4.0 * r_source * r_l * np.abs(v_ratio) ** 2 / np.abs(z_load) ** 2
    insertion_gain = np.where(m_src > 1e-9, t_total / m_src, 0.0)
    return {
        "swr": swr,
        "insertion_gain": np.clip(insertion_gain, 0.0, 1.0),
        "t_total": np.clip(t_total, 0.0, 1.0),
        "g_in": g_in,
        "z_in": z_in,
    }


def metrics_max(metrics):
    swr = float(metrics["swr"].max())
    loss_db = float((-10 * np.log10(np.clip(metrics["insertion_gain"], 1e-9, 1.0))).max())
    return swr, loss_db


def initial_x(kinds, omega_center, r_source=Z0):
    init = []
    for k in kinds:
        if k.startswith("L"):
            init.append(r_source / omega_center)
        elif k.startswith("C"):
            init.append(1.0 / (r_source * omega_center))
        elif k == "R_shunt":
            init.append(R_SHUNT_OFF)
        else:
            init.append(50.0)
    return np.concatenate([np.log(init), [16.0, 26.5]])


def bounds_for(kinds):
    b = []
    for k in kinds:
        if k.startswith("L"):
            b.append((np.log(1e-10), np.log(1e-2)))
        elif k.startswith("C"):
            b.append((np.log(1e-15), np.log(1e-5)))
        else:
            b.append((np.log(0.5), np.log(1e6)))
    b.append(F_LOW_BOUNDS)
    b.append(F_HIGH_BOUNDS)
    return b


def objective(x, kinds, omegas, r_source, budget, lam):
    n_k = len(kinds)
    matcher_vals = np.exp(x[:n_k])
    f_low, f_high = float(x[n_k]), float(x[n_k + 1])
    m = evaluate(matcher_vals, kinds, f_low, f_high, omegas, r_source)
    swr_max, loss_max = metrics_max(m)
    penalty = max(0.0, loss_max - budget) ** 2
    return swr_max + lam * penalty


def solve(kinds, omegas, *, label="", n_random=25, max_iter=400):
    omega_center = float(np.exp(np.mean(np.log(omegas))))
    rng = np.random.default_rng(0)
    x0 = initial_x(kinds, omega_center)
    bnds = bounds_for(kinds)
    n_k = len(kinds)

    starts = [x0.copy()]
    for _ in range(n_random):
        s = x0.copy()
        for i, k in enumerate(kinds):
            scale = 0.5 if k.startswith(("L", "C")) else 2.0
            s[i] += rng.normal(0.0, scale)
        s[n_k] += rng.normal(0.0, 1.0)
        s[n_k + 1] += rng.normal(0.0, 1.0)
        starts.append(s)
    # Explicit R-on starts to escape the lossless trap
    r_positions = [i for i, k in enumerate(kinds) if k.startswith("R")]
    for r_on in [30.0, 80.0, 200.0, 500.0]:
        for f_lo in [15.0, 16.0, 17.0]:
            for f_hi in [25.0, 26.0, 27.0]:
                s = x0.copy()
                for ri in r_positions:
                    s[ri] = np.log(r_on)
                s[n_k] = f_lo
                s[n_k + 1] = f_hi
                starts.append(s)
    log.info("  %s starting opt with %d starts", label, len(starts))

    best_x = None
    best_score = np.inf
    t0 = time.time()
    for phase, lam in enumerate([10.0, 100.0, 1000.0]):
        phase_best = None
        phase_best_score = np.inf
        for s in starts:
            s_clip = np.array([np.clip(s[i], b[0] + 0.01, b[1] - 0.01) for i, b in enumerate(bnds)])
            try:
                res = minimize(
                    objective,
                    s_clip,
                    args=(kinds, omegas, Z0, LOSS_BUDGET_DB, lam),
                    method="L-BFGS-B",
                    bounds=bnds,
                    options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-8},
                )
                v_t = np.exp(res.x[:n_k])
                f_l, f_h = float(res.x[n_k]), float(res.x[n_k + 1])
                m_t = evaluate(v_t, kinds, f_l, f_h, omegas)
                swr_t, loss_t = metrics_max(m_t)
                feas = loss_t <= LOSS_BUDGET_DB + 0.05
                score = swr_t if feas else (swr_t + 100 * (loss_t - LOSS_BUDGET_DB))
                if score < phase_best_score:
                    phase_best_score = score
                    phase_best = res.x
                if score < best_score:
                    best_score = score
                    best_x = res.x
            except Exception as e:  # noqa: BLE001
                log.debug("    %s phase%d: %s", label, phase, e)
                continue
        v_p = np.exp(phase_best[:n_k])
        m_p = evaluate(v_p, kinds, float(phase_best[n_k]), float(phase_best[n_k + 1]), omegas)
        swr_p, loss_p = metrics_max(m_p)
        log.info(
            "  %s phase %d/3 lam=%g done  score=%.3f  SWR=%.3f  loss=%.2f dB",
            label,
            phase + 1,
            lam,
            phase_best_score,
            swr_p,
            loss_p,
        )
        starts = [phase_best] + [
            phase_best + rng.normal(0.0, 0.3, size=len(phase_best))
            for _ in range(min(15, n_random))
        ]

    matcher_vals = np.exp(best_x[:n_k])
    f_low = float(best_x[n_k])
    f_high = float(best_x[n_k + 1])
    metrics = evaluate(matcher_vals, kinds, f_low, f_high, omegas)
    swr, loss = metrics_max(metrics)
    log.info(
        "  %s DONE in %.1fs  SWR=%.3f  loss=%.2f dB  f=(%.2f, %.2f)",
        label,
        time.time() - t0,
        swr,
        loss,
        f_low,
        f_high,
    )
    return matcher_vals, f_low, f_high, metrics


def main() -> None:
    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    log.info("=" * 72)
    log.info("Complexity sweep — joint length opt, %.1f dB loss budget", LOSS_BUDGET_DB)
    log.info("=" * 72)

    results = []
    for label, kinds in TOPOLOGIES:
        log.info("")
        log.info(">>> %s : %s", label, " - ".join(kinds))
        n_r = sum(1 for k in kinds if k.startswith("R"))
        n_l = sum(1 for k in kinds if k.startswith("L"))
        n_c = sum(1 for k in kinds if k.startswith("C"))
        log.info("    components: %d L, %d C, %d R (total %d)", n_l, n_c, n_r, len(kinds))
        matcher_vals, f_low, f_high, metrics = solve(kinds, omegas, label=label)
        swr, loss = metrics_max(metrics)
        per_band_swr = metrics["swr"]
        per_band_loss_db = -10 * np.log10(np.clip(metrics["insertion_gain"], 1e-9, 1.0))
        results.append(
            {
                "label": label,
                "n": len(kinds),
                "kinds": kinds,
                "values": matcher_vals,
                "f_low": f_low,
                "f_high": f_high,
                "worst_swr": swr,
                "worst_loss": loss,
                "per_band_swr": per_band_swr,
                "per_band_loss": per_band_loss_db,
            }
        )

    # ---- Summary table ----
    log.info("")
    log.info("=" * 90)
    log.info("SUMMARY — worst-case SWR vs matcher complexity (2 dB worst-loss budget)")
    log.info("=" * 90)
    log.info(
        "  %6s  %6s  %6s  %6s   %12s %12s %14s",
        "label",
        "N",
        "worst",
        "worst",
        "f_low",
        "f_high",
        "per-band SWR",
    )
    log.info("  %6s  %6s  %6s  %6s   %12s %12s", "", "", "SWR", "loss", "(MHz)", "(MHz)")
    for r in results:
        per_band_str = ", ".join(f"{s:.2f}" for s in r["per_band_swr"])
        log.info(
            "  %6s  %6d  %6.2f  %5.2f dB  %12.3f %12.3f   [%s]",
            r["label"],
            r["n"],
            r["worst_swr"],
            r["worst_loss"],
            r["f_low"],
            r["f_high"],
            per_band_str,
        )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 7))
    n_vals = [r["n"] for r in results]
    swr_vals = [r["worst_swr"] for r in results]
    ax.plot(n_vals, swr_vals, "o-", color="C0", lw=2, ms=10)
    for r in results:
        ax.annotate(
            f"  {r['worst_swr']:.2f}", xy=(r["n"], r["worst_swr"]), fontsize=10, va="center"
        )
    ax.axhline(2.0, color="grey", ls=":", lw=0.8, label="SWR = 2")
    ax.set_xlabel("matcher component count")
    ax.set_ylabel("worst-band SWR (at 2 dB loss budget)")
    ax.set_title(
        "Complexity vs SWR tradeoff for joint-length lossy matcher\n"
        f"(inverted-V antenna R_rad=50, worst loss ≤ {LOSS_BUDGET_DB} dB)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_vals)
    fig.tight_layout()
    fig.savefig("lossy_complexity_sweep.png", dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote lossy_complexity_sweep.png")
    log.info("log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
