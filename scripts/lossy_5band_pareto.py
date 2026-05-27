#!/usr/bin/env python
"""Lossy matching network Pareto analysis on the 5-band 2-element fan dipole.

Two specific design questions:
  (A) Lowest worst-case SWR achievable, subject to insertion loss ≤ 3 dB
      at every design frequency
  (B) Lowest worst-case insertion loss, subject to worst-case SWR ≤ 2.0
      at every design frequency

Strategy: warm-start from v1 lossless optimum (so the optimizer doesn't
have to rediscover reactive matching), then add 'off' R elements and
let the optimizer choose how much to dissipate.

T(ω) is computed via an ABCD-matrix cascade so dissipation is properly
accounted for. T = (1-|Γ_in|²) · insertion_gain, with insertion_gain
= P_load/P_input ≤ 1 for lossy networks.

Logs progress to logs/lossy_5band_pareto.log.
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
from antmatch.matching import (
    MatchingProblem,
    solve_lc_ladder,
)


# ---------- Logging ----------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "lossy_5band_pareto_17_12.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("lossy")


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0


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


def evaluate(
    kinds: list[str],
    values: np.ndarray,
    z_load: np.ndarray,
    omegas: np.ndarray,
    r_source: float = Z0,
) -> dict:
    M = cascade_abcd(kinds, values, omegas)
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
    }


def metrics_summary(metrics: dict) -> tuple[float, float]:
    swr = float(metrics["swr"].max())
    loss_db = float((-10 * np.log10(np.clip(metrics["insertion_gain"], 1e-9, 1.0))).max())
    return swr, loss_db


# ---------- Lossless v1 baseline ----------


# v1 result on the 5-band 2-element fan dipole (from yarman_aksen_demo.py
# results recorded in MEMORY.md / earlier work):
#   degree 6, C_shunt-L_series x3 → SWR 3.96
# That's the best lossless topology at this complexity. Use it.
LOSSLESS_TOPOLOGY = ["C_shunt", "L_series", "C_shunt", "L_series", "C_shunt", "L_series"]


def lossless_v1_baseline(
    z_load: np.ndarray, omegas: np.ndarray, r_source: float = Z0
) -> tuple[list[str], np.ndarray, dict]:
    log.info("running v1 lossless solver on topology: %s", " - ".join(LOSSLESS_TOPOLOGY))
    problem = MatchingProblem(
        omegas_design=omegas,
        z_load_design=z_load,
        omegas_grid=omegas,
        z_load_grid=z_load,
        omegas_breakpts=omegas,
        r_source=r_source,
    )
    t0 = time.time()
    res = solve_lc_ladder(problem, LOSSLESS_TOPOLOGY, n_restarts=30, max_iter=500)
    metrics = evaluate(LOSSLESS_TOPOLOGY, res.values, z_load, omegas, r_source)
    swr, loss = metrics_summary(metrics)
    log.info(
        "v1 baseline done in %.1fs  worst SWR=%.3f  worst loss=%.3f dB", time.time() - t0, swr, loss
    )
    return LOSSLESS_TOPOLOGY, res.values, metrics


# ---------- Lossy ladder warm-start ----------

# 'off' values for R elements at warm start
R_SHUNT_OFF = 1.0e6  # 1 MΩ shunt ≈ open
R_SERIES_OFF = 1.0e-3  # 1 mΩ series ≈ short


def warm_start_with_R(
    lc_kinds: list[str], lc_values: np.ndarray, r_positions: list[tuple[int, str]]
) -> tuple[list[str], np.ndarray]:
    kinds = list(lc_kinds)
    values = list(lc_values)
    for idx, kind in sorted(r_positions, key=lambda x: -x[0]):
        kinds.insert(idx, kind)
        v0 = R_SHUNT_OFF if kind == "R_shunt" else R_SERIES_OFF
        values.insert(idx, v0)
    return kinds, np.array(values)


# ---------- Constrained optimization ----------


def constrained_objective(
    log_vals: np.ndarray,
    kinds: list[str],
    z_load: np.ndarray,
    omegas: np.ndarray,
    r_source: float,
    mode: str,
    bound: float,
    lam: float,
) -> float:
    values = np.exp(log_vals)
    metrics = evaluate(kinds, values, z_load, omegas, r_source)
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


def solve_constrained(
    kinds: list[str],
    init_values: np.ndarray,
    z_load: np.ndarray,
    omegas: np.ndarray,
    *,
    mode: str,
    bound: float,
    r_source: float = Z0,
    n_restarts: int = 20,
    max_iter: int = 400,
    rng_seed: int = 0,
    log_jitter: float = 0.5,
    label: str = "",
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(rng_seed)
    log_init = np.log(np.maximum(init_values, 1e-30))
    bounds_box: list[tuple[float, float]] = []
    for k in kinds:
        if k.startswith("L"):
            bounds_box.append((np.log(1e-10), np.log(1e-2)))
        elif k.startswith("C"):
            bounds_box.append((np.log(1e-15), np.log(1e-5)))
        else:
            bounds_box.append((np.log(0.5), np.log(1e6)))

    starts = [log_init.copy()]
    for _ in range(n_restarts):
        s = log_init.copy()
        for i, k in enumerate(kinds):
            scale = log_jitter if k.startswith(("L", "C")) else 2.0
            s[i] += rng.normal(0.0, scale)
        starts.append(s)

    best_x = None
    best_score = np.inf
    t0 = time.time()

    penalty_phases = [10.0, 100.0, 1000.0]
    for phase_idx, lam in enumerate(penalty_phases):
        phase_best = None
        phase_best_score = np.inf
        for s_idx, s in enumerate(starts):
            s_clip = np.clip(s, [b[0] + 0.5 for b in bounds_box], [b[1] - 0.5 for b in bounds_box])
            try:
                res = minimize(
                    constrained_objective,
                    s_clip,
                    args=(kinds, z_load, omegas, r_source, mode, bound, lam),
                    method="L-BFGS-B",
                    bounds=bounds_box,
                    options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-8},
                )
                v_test = np.exp(res.x)
                m_test = evaluate(kinds, v_test, z_load, omegas, r_source)
                swr_t, loss_t = metrics_summary(m_test)
                if mode == "min_swr_bounded_loss":
                    feasible = loss_t <= bound + 0.05
                    score = swr_t if feasible else (swr_t + 100 * (loss_t - bound))
                else:
                    feasible = swr_t <= bound + 0.02
                    score = loss_t if feasible else (loss_t + 100 * (swr_t - bound))
                if score < phase_best_score:
                    phase_best_score = score
                    phase_best = res.x
                if score < best_score:
                    best_score = score
                    best_x = res.x
            except Exception as e:  # noqa: BLE001
                log.debug("    %s phase%d start%d: %s", label, phase_idx, s_idx, e)
                continue

        # End-of-phase log
        if phase_best is not None:
            v_p = np.exp(phase_best)
            m_p = evaluate(kinds, v_p, z_load, omegas, r_source)
            swr_p, loss_p = metrics_summary(m_p)
            log.info(
                "  %s phase %d/%d lam=%g done  best score=%.3f  SWR=%.3f  loss=%.2f dB",
                label,
                phase_idx + 1,
                len(penalty_phases),
                lam,
                phase_best_score,
                swr_p,
                loss_p,
            )
            # Refine starts around current phase-best for next phase
            starts = [phase_best] + [
                phase_best + rng.normal(0.0, 0.3, size=len(kinds))
                for _ in range(min(15, n_restarts))
            ]

    values = np.exp(best_x)
    metrics = evaluate(kinds, values, z_load, omegas, r_source)
    swr, loss = metrics_summary(metrics)
    log.info(
        "%s DONE in %.1fs  worst SWR=%.3f  worst loss=%.3f dB", label, time.time() - t0, swr, loss
    )
    return values, metrics


# ---------- Reporting ----------


def fmt_val(kind: str, val: float) -> str:
    if kind.startswith("L"):
        return f"{val * 1e6:.3g} µH"
    if kind.startswith("C"):
        return f"{val * 1e12:.3g} pF"
    return f"{val:.3g} Ω"


def show(label: str, kinds: list[str], values: np.ndarray, metrics: dict) -> None:
    swr = metrics["swr"]
    ig = metrics["insertion_gain"]
    loss_db = -10 * np.log10(np.clip(ig, 1e-9, 1.0))
    log.info("%s", label)
    log.info("  topology: %s", " - ".join(kinds))
    log.info("  values:   %s", ",  ".join(f"{fmt_val(k, v)}" for k, v in zip(kinds, values)))
    log.info("  %5s  %6s  %9s  %7s", "band", "SWR", "loss(dB)", "T(dB)")
    for lbl, sw, lo_db, t in zip(BAND_LABEL, swr, loss_db, metrics["t_total"]):
        t_db = 10 * np.log10(max(t, 1e-9))
        log.info("  %5s  %6.2f  %9.2f  %7.2f", lbl, sw, lo_db, t_db)
    log.info("  worst SWR = %.2f    worst loss = %.2f dB", swr.max(), loss_db.max())


def main() -> None:
    log.info("=" * 70)
    log.info("Lossy 5-band Pareto analysis on 2-element fan dipole")
    log.info("=" * 70)

    # Antenna resonances moved INWARD to 17m and 12m (was 20m/10m).
    # Narrower bandwidth ratio (1.37:1 vs 1.99:1) → much smaller midband
    # impedance swing, so a simpler lossy matcher should do better.
    el_lower = DipoleElement(leg_length_m=leg(18.1575))
    el_upper = DipoleElement(leg_length_m=leg(24.970))
    num, den = fan_dipole_impedance([el_lower, el_upper])
    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    z_load = evaluate_rational(num, den, 1j * omegas)

    log.info("antenna: 2 elements at 18.16 / 24.97 MHz (17m/12m resonances, fixed)")
    for lbl, f, z in zip(BAND_LABEL, F_DESIGN_MHZ, z_load):
        g = (z - Z0) / (z + Z0)
        s = (1 + abs(g)) / (1 - abs(g))
        log.info("  %s f=%.3f MHz  Z=%+.2f%+.2fj  bare SWR=%.2f", lbl, f, z.real, z.imag, s)

    # ----- Stage 0: lossless v1 baseline -----
    log.info("")
    log.info(">>> STAGE 0: lossless v1 LC baseline")
    lc_kinds, lc_values, lc_metrics = lossless_v1_baseline(z_load, omegas)
    show("v1 lossless baseline:", lc_kinds, lc_values, lc_metrics)

    # Add 2 shunt R elements (initially off): one at input, one in middle
    kinds, init_vals = warm_start_with_R(lc_kinds, lc_values, [(0, "R_shunt"), (3, "R_shunt")])
    m0 = evaluate(kinds, init_vals, z_load, omegas)
    s0, l0 = metrics_summary(m0)
    log.info("")
    log.info("lossy topology (R values start at 1 MΩ = effectively off):")
    log.info("  %s", " - ".join(kinds))
    log.info("warm start sanity (R off): worst SWR=%.2f, worst loss=%.3f dB", s0, l0)

    # ----- (A) min SWR with loss ≤ 3 dB -----
    log.info("")
    log.info(">>> (A) Minimize worst-case SWR  s.t.  insertion loss ≤ 3 dB")
    values_A, metrics_A = solve_constrained(
        kinds,
        init_vals,
        z_load,
        omegas,
        mode="min_swr_bounded_loss",
        bound=3.0,
        label="(A)",
    )
    show("Result (A):", kinds, values_A, metrics_A)

    # ----- (B) min loss with SWR ≤ 2 -----
    log.info("")
    log.info(">>> (B) Minimize worst-case insertion loss  s.t.  SWR ≤ 2.0")
    values_B, metrics_B = solve_constrained(
        kinds,
        init_vals,
        z_load,
        omegas,
        mode="min_loss_bounded_swr",
        bound=2.0,
        label="(B)",
    )
    show("Result (B):", kinds, values_B, metrics_B)

    # ----- Pareto sweeps -----
    log.info("")
    log.info(">>> Pareto sweep — fix loss budget, find min achievable worst SWR")
    loss_budgets = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    pareto_loss = [(0.0, 0.0, lc_metrics["swr"].max(), lc_kinds, lc_values, lc_metrics)]
    for budget in loss_budgets:
        log.info("--- loss budget = %.2f dB ---", budget)
        v_p, m_p = solve_constrained(
            kinds,
            init_vals,
            z_load,
            omegas,
            mode="min_swr_bounded_loss",
            bound=budget,
            n_restarts=15,
            label=f"sweep_loss≤{budget:.1f}",
        )
        swr_p, loss_p = metrics_summary(m_p)
        pareto_loss.append((budget, loss_p, swr_p, kinds, v_p, m_p))

    log.info("")
    log.info(">>> Pareto sweep — fix SWR ceiling, find min achievable worst loss")
    swr_ceilings = [3.0, 2.5, 2.0, 1.75, 1.5, 1.3]
    pareto_swr = []
    for ceiling in swr_ceilings:
        log.info("--- SWR ceiling = %.2f ---", ceiling)
        v_p, m_p = solve_constrained(
            kinds,
            init_vals,
            z_load,
            omegas,
            mode="min_loss_bounded_swr",
            bound=ceiling,
            n_restarts=15,
            label=f"sweep_SWR≤{ceiling:.2f}",
        )
        swr_p, loss_p = metrics_summary(m_p)
        pareto_swr.append((ceiling, loss_p, swr_p, v_p, m_p))

    # ----- Print summary tables -----
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY — Pareto sweep (loss budget → min SWR)")
    log.info("=" * 70)
    log.info("  %8s   %14s   %14s", "loss ≤", "achieved SWR", "achieved loss")
    for budget, loss_p, swr_p, *_ in pareto_loss:
        log.info("  %6.2f dB   %14.3f   %12.2f dB", budget, swr_p, loss_p)

    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY — Pareto sweep (SWR ceiling → min loss)")
    log.info("=" * 70)
    log.info("  %6s   %14s   %14s", "SWR ≤", "achieved loss", "achieved SWR")
    for ceiling, loss_p, swr_p, *_ in pareto_swr:
        log.info("  %6.2f   %12.2f dB   %14.3f", ceiling, loss_p, swr_p)

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(10, 7))
    p_x = [p[1] for p in pareto_loss]
    p_y = [p[2] for p in pareto_loss]
    ax.plot(p_x, p_y, "o-", color="C0", lw=1.8, ms=8, label="min SWR for loss budget")

    p2_x = [p[1] for p in pareto_swr]
    p2_y = [p[2] for p in pareto_swr]
    ax.plot(
        p2_x, p2_y, "s--", color="C3", lw=1.6, ms=8, alpha=0.8, label="min loss for SWR ceiling"
    )

    ax.axhline(
        lc_metrics["swr"].max(),
        color="grey",
        ls=":",
        lw=1,
        label=f"v1 lossless 6-LC ladder (SWR {lc_metrics['swr'].max():.2f})",
    )

    loss_A_val = float(-10 * np.log10(np.clip(metrics_A["insertion_gain"], 1e-9, 1.0)).max())
    ax.scatter(
        [loss_A_val],
        [metrics_A["swr"].max()],
        s=260,
        marker="*",
        color="C1",
        edgecolors="black",
        lw=1.2,
        zorder=10,
        label=f"(A) ≤3 dB loss → SWR {metrics_A['swr'].max():.2f}",
    )
    loss_B_val = float(-10 * np.log10(np.clip(metrics_B["insertion_gain"], 1e-9, 1.0)).max())
    ax.scatter(
        [loss_B_val],
        [metrics_B["swr"].max()],
        s=260,
        marker="D",
        color="C4",
        edgecolors="black",
        lw=1.2,
        zorder=10,
        label=f"(B) SWR ≤2 → {loss_B_val:.2f} dB",
    )

    ax.set_xlabel("worst-band insertion loss (dB)")
    ax.set_ylabel("worst-band SWR")
    ax.set_yscale("log")
    ax.set_xlim(-0.2, max(8, max(p_x + p2_x + [loss_A_val, loss_B_val]) + 1))
    ax.set_ylim(1.0, 6)
    ax.set_title(
        "Pareto: worst SWR vs worst insertion loss\n"
        "2-element fan dipole (fixed legs at 18.16 / 24.97 MHz — 17m/12m), 5-band design"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("lossy_pareto_17_12.png", dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote lossy_pareto_17_12.png")
    log.info("log file: %s", LOG_PATH)


if __name__ == "__main__":
    main()
