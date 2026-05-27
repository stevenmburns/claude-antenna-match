#!/usr/bin/env python
"""Lossy matching network + co-optimized antenna lengths.

Question: with antenna leg lengths also free, can we do better on the
5-band 2-element fan dipole problem?

Constraints based on the user's hint:
- Lower element resonance f_low ∈ [14.30, 18.16] MHz (between 20m and 17m)
- Upper element resonance f_high ∈ [24.97, 28.47] MHz (between 12m and 10m)

Both lengths are free variables alongside the matcher elements (6 LC +
2 R_shunt). The two scalar Pareto questions:
  (A) min worst SWR  s.t.  insertion loss ≤ 3 dB
  (B) min worst loss s.t.  worst SWR ≤ 2.0

For comparison, also reports the lossless answer (R kept off) and the
optimum antenna lengths it picks.

Log: logs/lossy_5band_jointlen.log
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


# ---------- Logging ----------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "lossy_5band_jointlen.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("jointlen")


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0

# Bounds on the two antenna element resonance frequencies (MHz)
F_LOW_BOUNDS = (14.30, 18.16)
F_HIGH_BOUNDS = (24.97, 28.47)


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


# ---------- ABCD-matrix cascade ----------


def section_abcd(kind: str, value: float, omegas: np.ndarray) -> np.ndarray:
    n = len(omegas)
    if kind == "L_series":
        ze = 1j * omegas * value
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "C_series":
        ze = 1.0 / (1j * omegas * value)
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "R_series":
        ze = np.full(n, value, dtype=complex)
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "L_shunt":
        ye = 1.0 / (1j * omegas * value)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "C_shunt":
        ye = 1j * omegas * value
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "R_shunt":
        ye = np.full(n, 1.0 / value, dtype=complex)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    raise ValueError(f"unknown kind {kind!r}")


def cascade_abcd(kinds: list[str], values: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    n = len(omegas)
    M = np.array([[np.ones(n), np.zeros(n)], [np.zeros(n), np.ones(n)]], dtype=complex)
    for kind, val in zip(kinds, values, strict=True):
        Mi = section_abcd(kind, val, omegas)
        M = np.einsum("ilk,ljk->ijk", M, Mi)
    return M


def z_load_from_resonances(f_low_mhz: float, f_high_mhz: float, omegas: np.ndarray) -> np.ndarray:
    el_low = DipoleElement(leg_length_m=leg(f_low_mhz))
    el_high = DipoleElement(leg_length_m=leg(f_high_mhz))
    num, den = fan_dipole_impedance([el_low, el_high])
    return evaluate_rational(num, den, 1j * omegas)


def evaluate_full(
    kinds: list[str],
    matcher_values: np.ndarray,
    f_low_mhz: float,
    f_high_mhz: float,
    omegas: np.ndarray,
    r_source: float = Z0,
) -> dict:
    z_load = z_load_from_resonances(f_low_mhz, f_high_mhz, omegas)
    M = cascade_abcd(kinds, matcher_values, omegas)
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
        "z_in": z_in,
        "z_load": z_load,
    }


def metrics_summary(metrics: dict) -> tuple[float, float]:
    swr = float(metrics["swr"].max())
    loss_db = float((-10 * np.log10(np.clip(metrics["insertion_gain"], 1e-9, 1.0))).max())
    return swr, loss_db


# ---------- Variable packing ----------
#
# x = [ log(matcher_values...) , f_low_mhz , f_high_mhz ]
# Last 2 components are LINEAR (not log). bounds applied separately.

LOSSLESS_TOPOLOGY = ["C_shunt", "L_series", "C_shunt", "L_series", "C_shunt", "L_series"]
LOSSY_TOPOLOGY = [
    "R_shunt",
    "C_shunt",
    "L_series",
    "C_shunt",
    "R_shunt",
    "L_series",
    "C_shunt",
    "L_series",
]

R_SHUNT_OFF = 1.0e6
R_SERIES_OFF = 1.0e-3


def unpack(x: np.ndarray, n_matcher: int) -> tuple[np.ndarray, float, float]:
    matcher_log = x[:n_matcher]
    f_low = float(x[n_matcher])
    f_high = float(x[n_matcher + 1])
    matcher_vals = np.exp(matcher_log)
    return matcher_vals, f_low, f_high


def initial_x(
    kinds: list[str],
    omega_center: float,
    r_source: float = Z0,
    f_low_init: float = 16.0,
    f_high_init: float = 26.5,
) -> np.ndarray:
    init_lc = []
    for k in kinds:
        if k.startswith("L"):
            init_lc.append(r_source / omega_center)
        elif k.startswith("C"):
            init_lc.append(1.0 / (r_source * omega_center))
        elif k == "R_shunt":
            init_lc.append(R_SHUNT_OFF)
        else:
            init_lc.append(R_SERIES_OFF)
    return np.concatenate([np.log(init_lc), [f_low_init, f_high_init]])


def bounds_for_x(kinds: list[str]) -> list[tuple[float, float]]:
    b: list[tuple[float, float]] = []
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


# ---------- Objective ----------


def constrained_objective(
    x: np.ndarray,
    kinds: list[str],
    omegas: np.ndarray,
    r_source: float,
    mode: str,
    bound: float,
    lam: float,
) -> float:
    matcher_vals, f_low, f_high = unpack(x, len(kinds))
    metrics = evaluate_full(kinds, matcher_vals, f_low, f_high, omegas, r_source)
    swr = metrics["swr"]
    ig = metrics["insertion_gain"]
    loss_db = -10.0 * np.log10(np.maximum(ig, 1e-9))
    max_swr = float(swr.max())
    max_loss = float(loss_db.max())
    if mode == "min_swr_bounded_loss":
        penalty = max(0.0, max_loss - bound) ** 2
        return max_swr + lam * penalty
    elif mode == "min_loss_bounded_swr":
        penalty = max(0.0, max_swr - bound) ** 2
        return max_loss + lam * penalty
    else:
        raise ValueError(mode)


def solve_jointlen(
    kinds: list[str],
    omegas: np.ndarray,
    *,
    mode: str,
    bound: float,
    r_source: float = Z0,
    n_restarts: int = 25,
    max_iter: int = 500,
    rng_seed: int = 0,
    label: str = "",
) -> tuple[np.ndarray, float, float, dict]:
    """Joint optimization over matcher elements AND antenna resonance freqs."""
    rng = np.random.default_rng(rng_seed)
    omega_center = float(np.exp(np.mean(np.log(omegas))))
    x0 = initial_x(kinds, omega_center)
    bnds = bounds_for_x(kinds)

    # Build a diverse set of starts:
    #   - x0 (initial guess)
    #   - perturbations of x0
    #   - explicitly try different (f_low, f_high) starting points
    starts = [x0.copy()]
    for _ in range(n_restarts):
        s = x0.copy()
        for i, k in enumerate(kinds):
            scale = 0.5 if k.startswith(("L", "C")) else 2.0
            s[i] += rng.normal(0.0, scale)
        # Also jitter the lengths
        s[len(kinds)] += rng.normal(0.0, 1.0)
        s[len(kinds) + 1] += rng.normal(0.0, 1.0)
        starts.append(s)
    # Plus a few corner starts
    for f_lo, f_hi in [(14.3, 28.47), (18.16, 24.97), (16.0, 26.5), (15.0, 27.5), (17.0, 26.0)]:
        s = x0.copy()
        s[len(kinds)] = f_lo
        s[len(kinds) + 1] = f_hi
        starts.append(s)

    best_x = None
    best_score = np.inf
    t0 = time.time()
    penalty_phases = [10.0, 100.0, 1000.0]

    for phase_idx, lam in enumerate(penalty_phases):
        phase_best = None
        phase_best_score = np.inf
        for s in starts:
            s_clip = np.array([np.clip(s[i], b[0] + 0.01, b[1] - 0.01) for i, b in enumerate(bnds)])
            try:
                res = minimize(
                    constrained_objective,
                    s_clip,
                    args=(kinds, omegas, r_source, mode, bound, lam),
                    method="L-BFGS-B",
                    bounds=bnds,
                    options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-8},
                )
                matcher_vals, f_low, f_high = unpack(res.x, len(kinds))
                m_t = evaluate_full(kinds, matcher_vals, f_low, f_high, omegas, r_source)
                swr_t, loss_t = metrics_summary(m_t)
                if mode == "min_swr_bounded_loss":
                    feas = loss_t <= bound + 0.05
                    score = swr_t if feas else (swr_t + 100 * (loss_t - bound))
                else:
                    feas = swr_t <= bound + 0.02
                    score = loss_t if feas else (loss_t + 100 * (swr_t - bound))
                if score < phase_best_score:
                    phase_best_score = score
                    phase_best = res.x
                if score < best_score:
                    best_score = score
                    best_x = res.x
            except Exception as e:  # noqa: BLE001
                log.debug("    %s phase%d: %s", label, phase_idx, e)
                continue
        if phase_best is not None:
            matcher_vals, f_low, f_high = unpack(phase_best, len(kinds))
            m_p = evaluate_full(kinds, matcher_vals, f_low, f_high, omegas, r_source)
            swr_p, loss_p = metrics_summary(m_p)
            log.info(
                "  %s phase %d/%d lam=%g  score=%.3f  SWR=%.3f  loss=%.2f dB  "
                "f_low=%.2f f_high=%.2f",
                label,
                phase_idx + 1,
                len(penalty_phases),
                lam,
                phase_best_score,
                swr_p,
                loss_p,
                f_low,
                f_high,
            )
            starts = [phase_best] + [
                phase_best + rng.normal(0.0, 0.3, size=len(phase_best)) for _ in range(15)
            ]

    matcher_vals, f_low, f_high = unpack(best_x, len(kinds))
    metrics = evaluate_full(kinds, matcher_vals, f_low, f_high, omegas, r_source)
    swr, loss = metrics_summary(metrics)
    log.info(
        "%s DONE in %.1fs  SWR=%.3f  loss=%.3f dB  f_low=%.3f  f_high=%.3f",
        label,
        time.time() - t0,
        swr,
        loss,
        f_low,
        f_high,
    )
    return matcher_vals, f_low, f_high, metrics


# ---------- Reporting ----------


def fmt_val(kind: str, val: float) -> str:
    if kind.startswith("L"):
        return f"{val * 1e6:.3g} µH"
    if kind.startswith("C"):
        return f"{val * 1e12:.3g} pF"
    return f"{val:.3g} Ω"


def show(
    label: str,
    kinds: list[str],
    matcher_values: np.ndarray,
    f_low: float,
    f_high: float,
    metrics: dict,
) -> None:
    swr = metrics["swr"]
    loss_db = -10 * np.log10(np.clip(metrics["insertion_gain"], 1e-9, 1.0))
    log.info("%s", label)
    log.info("  antenna resonances: f_low=%.3f MHz, f_high=%.3f MHz", f_low, f_high)
    log.info("  leg lengths:        L_lo=%.3f m, L_hi=%.3f m", leg(f_low), leg(f_high))
    log.info("  matcher topology:   %s", " - ".join(kinds))
    log.info(
        "  matcher values:     %s",
        ",  ".join(f"{fmt_val(k, v)}" for k, v in zip(kinds, matcher_values)),
    )
    log.info("  %5s  %6s  %9s  %7s", "band", "SWR", "loss(dB)", "T(dB)")
    for lbl, sw, lo_db, t in zip(BAND_LABEL, swr, loss_db, metrics["t_total"]):
        t_db = 10 * np.log10(max(t, 1e-9))
        log.info("  %5s  %6.2f  %9.2f  %7.2f", lbl, sw, lo_db, t_db)
    log.info("  worst SWR = %.2f    worst loss = %.2f dB", swr.max(), loss_db.max())


def main() -> None:
    log.info("=" * 72)
    log.info("Lossy 5-band Pareto WITH joint antenna-length optimization")
    log.info("=" * 72)
    log.info("Antenna resonance bounds: f_low %s, f_high %s", F_LOW_BOUNDS, F_HIGH_BOUNDS)
    log.info("Matcher topology: %s", " - ".join(LOSSY_TOPOLOGY))

    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6

    # ---- Lossless baseline with co-optimized antenna ----
    log.info("")
    log.info(">>> Lossless baseline: matcher = 6 LC, antenna f_low/f_high free")
    matcher_v0, f_lo_0, f_hi_0, m0 = solve_jointlen(
        LOSSLESS_TOPOLOGY,
        omegas,
        mode="min_swr_bounded_loss",
        bound=0.05,
        label="lossless",  # 0.05 dB ~= effectively zero loss
    )
    show("Lossless co-opt:", LOSSLESS_TOPOLOGY, matcher_v0, f_lo_0, f_hi_0, m0)

    # ---- (A) min SWR with loss ≤ 3 dB ----
    log.info("")
    log.info(">>> (A) Minimize worst-case SWR  s.t.  insertion loss ≤ 3 dB")
    matcher_A, f_lo_A, f_hi_A, mA = solve_jointlen(
        LOSSY_TOPOLOGY,
        omegas,
        mode="min_swr_bounded_loss",
        bound=3.0,
        label="(A)",
    )
    show("Result (A):", LOSSY_TOPOLOGY, matcher_A, f_lo_A, f_hi_A, mA)

    # ---- (B) min loss with SWR ≤ 2 ----
    log.info("")
    log.info(">>> (B) Minimize worst-case insertion loss  s.t.  SWR ≤ 2.0")
    matcher_B, f_lo_B, f_hi_B, mB = solve_jointlen(
        LOSSY_TOPOLOGY,
        omegas,
        mode="min_loss_bounded_swr",
        bound=2.0,
        label="(B)",
    )
    show("Result (B):", LOSSY_TOPOLOGY, matcher_B, f_lo_B, f_hi_B, mB)

    # ---- Pareto: a few loss budgets ----
    log.info("")
    log.info(">>> Pareto sweep (loss budget → min SWR), with joint length opt")
    sweep = []
    swr_0, loss_0 = metrics_summary(m0)
    sweep.append((0.0, loss_0, swr_0, f_lo_0, f_hi_0, "lossless"))
    for budget in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        log.info("--- loss budget = %.2f dB ---", budget)
        matcher_p, f_lo, f_hi, m_p = solve_jointlen(
            LOSSY_TOPOLOGY,
            omegas,
            mode="min_swr_bounded_loss",
            bound=budget,
            n_restarts=15,
            label=f"sweep≤{budget:.1f}dB",
        )
        swr_p, loss_p = metrics_summary(m_p)
        sweep.append((budget, loss_p, swr_p, f_lo, f_hi, f"≤{budget}dB"))

    # ---- Summary tables ----
    log.info("")
    log.info("=" * 72)
    log.info("SUMMARY — joint length + lossy matcher Pareto")
    log.info("=" * 72)
    log.info(
        "  %8s  %14s  %14s  %8s  %8s", "loss ≤", "SWR achieved", "loss achieved", "f_low", "f_high"
    )
    for budget, loss_p, swr_p, f_lo, f_hi, _ in sweep:
        log.info("  %6.2f dB  %14.3f  %12.3f dB  %8.3f  %8.3f", budget, swr_p, loss_p, f_lo, f_hi)

    log.info("")
    log.info("Reference points:")
    log.info(
        "  (A) loss ≤ 3 dB:   SWR=%.3f, loss=%.3f dB, f=(%.2f, %.2f)",
        *metrics_summary(mA),
        f_lo_A,
        f_hi_A,
    )
    log.info(
        "  (B) SWR ≤ 2.0:     SWR=%.3f, loss=%.3f dB, f=(%.2f, %.2f)",
        *metrics_summary(mB),
        f_lo_B,
        f_hi_B,
    )

    # ---- Plot Pareto frontier + comparison to fixed-length runs ----
    fig, ax = plt.subplots(figsize=(10, 7))

    p_loss = [p[1] for p in sweep]
    p_swr = [p[2] for p in sweep]
    ax.plot(p_loss, p_swr, "o-", color="C0", lw=1.8, ms=8, label="joint length + lossy matcher")

    # Hard-coded comparison points from previous fixed-length runs:
    # antenna 20m+10m: lossless 5.36; (A) 3.18 @ 2.21; (B) 2.00 @ 1.89
    # antenna 17m+12m: lossless 3.77; (A) 2.94 @ 1.74; (B) 1.97 @ 2.14
    ax.scatter(
        [0.0, 2.21, 1.89],
        [5.36, 3.18, 2.00],
        s=60,
        marker="s",
        color="C3",
        alpha=0.7,
        label="fixed legs 20m+10m (prior)",
    )
    ax.scatter(
        [0.0, 1.74, 2.14],
        [3.77, 2.94, 1.97],
        s=60,
        marker="^",
        color="C2",
        alpha=0.7,
        label="fixed legs 17m+12m (prior)",
    )

    loss_A = float(-10 * np.log10(np.clip(mA["insertion_gain"], 1e-9, 1.0)).max())
    ax.scatter(
        [loss_A],
        [mA["swr"].max()],
        s=260,
        marker="*",
        color="C1",
        edgecolors="black",
        lw=1.2,
        zorder=10,
        label=f"(A) ≤3 dB → SWR {mA['swr'].max():.2f}",
    )
    loss_B = float(-10 * np.log10(np.clip(mB["insertion_gain"], 1e-9, 1.0)).max())
    ax.scatter(
        [loss_B],
        [mB["swr"].max()],
        s=260,
        marker="D",
        color="C4",
        edgecolors="black",
        lw=1.2,
        zorder=10,
        label=f"(B) SWR ≤2 → {loss_B:.2f} dB",
    )

    ax.set_xlabel("worst-band insertion loss (dB)")
    ax.set_ylabel("worst-band SWR")
    ax.set_yscale("log")
    ax.set_xlim(-0.2, max(6, max(p_loss + [loss_A, loss_B]) + 1))
    ax.set_ylim(1.0, 8)
    ax.set_title(
        "Pareto frontier with joint antenna length + lossy matcher co-optimization\n"
        "(compared to two fixed-length runs)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("lossy_jointlen.png", dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote lossy_jointlen.png")
    log.info("log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
