#!/usr/bin/env python
"""Same N=8 topology, but with both R_shunt values FIXED at 500 Ω.

Compares to the unconstrained R result (which chose 437 Ω and 523 Ω)
to see how much SWR we lose by using a single off-the-shelf resistor
value for both positions.

Joint length opt, 2 dB worst-loss budget, R_rad=50 (inverted-V).
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
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
LOG_PATH = os.path.join(LOG_DIR, "lossy_fixed_R500.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fix500")


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0
LOSS_BUDGET_DB = 2.0
R_RAD = 50.0
R_FIXED = 500.0
F_LOW_BOUNDS = (14.30, 18.16)
F_HIGH_BOUNDS = (24.97, 28.47)

KINDS = [
    "R_shunt",
    "C_shunt",
    "L_series",
    "C_shunt",
    "R_shunt",
    "L_series",
    "C_shunt",
    "L_series",
]
R_POSITIONS = [i for i, k in enumerate(KINDS) if k.startswith("R")]
FREE_POSITIONS = [i for i in range(len(KINDS)) if i not in R_POSITIONS]


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def section_abcd(kind, val, omegas):
    n = len(omegas)
    if kind == "L_series":
        ze = 1j * omegas * val
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "C_series":
        ze = 1.0 / (1j * omegas * val)
        return np.array([[np.ones(n), ze], [np.zeros(n), np.ones(n)]])
    if kind == "L_shunt":
        ye = 1.0 / (1j * omegas * val)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "C_shunt":
        ye = 1j * omegas * val
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    if kind == "R_shunt":
        ye = np.full(n, 1.0 / val, dtype=complex)
        return np.array([[np.ones(n), np.zeros(n)], [ye, np.ones(n)]])
    raise ValueError(kind)


def cascade_abcd(values, omegas):
    n = len(omegas)
    M = np.array([[np.ones(n), np.zeros(n)], [np.zeros(n), np.ones(n)]], dtype=complex)
    for kind, val in zip(KINDS, values, strict=True):
        M = np.einsum("ilk,ljk->ijk", M, section_abcd(kind, val, omegas))
    return M


def z_load_of(f_low, f_high, omegas):
    el_lo = DipoleElement(leg_length_m=leg(f_low), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(f_high), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return evaluate_rational(num, den, 1j * omegas)


def evaluate_full(values, f_low, f_high, omegas):
    z_load = z_load_of(f_low, f_high, omegas)
    M = cascade_abcd(values, omegas)
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    z_in = (A * z_load + B) / (C * z_load + D)
    v_load_over_v1 = z_load / (A * z_load + B)
    v1_over_vs = z_in / (z_in + Z0)
    v_ratio = v1_over_vs * v_load_over_v1
    g_in = (z_in - Z0) / (z_in + Z0)
    abs_g = np.minimum(np.abs(g_in), 0.999)
    swr = (1 + abs_g) / (1 - abs_g)
    m_src = 4.0 * Z0 * z_in.real / np.abs(z_in + Z0) ** 2
    r_l = z_load.real
    t_total = 4.0 * Z0 * r_l * np.abs(v_ratio) ** 2 / np.abs(z_load) ** 2
    insertion_gain = np.where(m_src > 1e-9, t_total / m_src, 0.0)
    return {
        "swr": swr,
        "insertion_gain": np.clip(insertion_gain, 0.0, 1.0),
        "t_total": np.clip(t_total, 0.0, 1.0),
    }


def metrics_max(m):
    return (
        float(m["swr"].max()),
        float((-10 * np.log10(np.clip(m["insertion_gain"], 1e-9, 1.0))).max()),
    )


def pack(x_free):
    """x_free has [log(L,C) for free positions, f_low, f_high].
    Returns full values array (R values fixed at R_FIXED)."""
    values = np.zeros(len(KINDS))
    for j, pos in enumerate(FREE_POSITIONS):
        values[pos] = np.exp(x_free[j])
    for pos in R_POSITIONS:
        values[pos] = R_FIXED
    f_low = float(x_free[-2])
    f_high = float(x_free[-1])
    return values, f_low, f_high


def objective(x_free, omegas, budget, lam):
    values, f_low, f_high = pack(x_free)
    m = evaluate_full(values, f_low, f_high, omegas)
    swr_max, loss_max = metrics_max(m)
    return swr_max + lam * max(0.0, loss_max - budget) ** 2


def solve():
    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    omega_center = float(np.exp(np.mean(np.log(omegas))))
    rng = np.random.default_rng(0)

    # initial: physics-based start for L/C, midpoints for lengths
    init_lc = []
    for pos in FREE_POSITIONS:
        k = KINDS[pos]
        if k.startswith("L"):
            init_lc.append(Z0 / omega_center)
        else:
            init_lc.append(1.0 / (Z0 * omega_center))
    x0 = np.concatenate([np.log(init_lc), [16.0, 26.5]])

    bnds = []
    for pos in FREE_POSITIONS:
        k = KINDS[pos]
        if k.startswith("L"):
            bnds.append((np.log(1e-10), np.log(1e-2)))
        else:
            bnds.append((np.log(1e-15), np.log(1e-5)))
    bnds.append(F_LOW_BOUNDS)
    bnds.append(F_HIGH_BOUNDS)

    # Multistart
    starts = [x0.copy()]
    for _ in range(40):
        s = x0.copy()
        for j in range(len(FREE_POSITIONS)):
            s[j] += rng.normal(0.0, 0.5)
        s[-2] += rng.normal(0.0, 1.0)
        s[-1] += rng.normal(0.0, 1.0)
        starts.append(s)
    for f_lo in [15.0, 16.0, 17.0]:
        for f_hi in [25.0, 26.0, 27.0]:
            s = x0.copy()
            s[-2] = f_lo
            s[-1] = f_hi
            starts.append(s)
    log.info("multistart: %d starts (R values pinned at %.0f Ω)", len(starts), R_FIXED)

    best_x = None
    best_score = np.inf
    t0 = time.time()
    for phase, lam in enumerate([10.0, 100.0, 1000.0]):
        phase_best = None
        phase_best_score = np.inf
        for s in starts:
            s_clip = np.array([np.clip(s[j], b[0] + 0.01, b[1] - 0.01) for j, b in enumerate(bnds)])
            try:
                res = minimize(
                    objective,
                    s_clip,
                    args=(omegas, LOSS_BUDGET_DB, lam),
                    method="L-BFGS-B",
                    bounds=bnds,
                    options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-8},
                )
                values, f_l, f_h = pack(res.x)
                m_t = evaluate_full(values, f_l, f_h, omegas)
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
                log.debug("phase%d: %s", phase, e)
                continue
        v_p, f_lp, f_hp = pack(phase_best)
        m_p = evaluate_full(v_p, f_lp, f_hp, omegas)
        sp, lp = metrics_max(m_p)
        log.info(
            "phase %d/3 lam=%g  SWR=%.3f  loss=%.2f dB  f=(%.2f, %.2f)",
            phase + 1,
            lam,
            sp,
            lp,
            f_lp,
            f_hp,
        )
        starts = [phase_best] + [
            phase_best + rng.normal(0.0, 0.3, size=len(phase_best)) for _ in range(15)
        ]

    values, f_low, f_high = pack(best_x)
    log.info("DONE in %.1fs", time.time() - t0)
    return values, f_low, f_high


def main():
    log.info("Fixed R=%.0f Ω, joint length opt, %.1f dB loss budget", R_FIXED, LOSS_BUDGET_DB)
    log.info("topology: %s", " - ".join(KINDS))
    values, f_low, f_high = solve()
    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    m = evaluate_full(values, f_low, f_high, omegas)
    swr_max, loss_max = metrics_max(m)
    log.info("")
    log.info("=== Result ===")
    log.info("f_low=%.3f MHz, f_high=%.3f MHz", f_low, f_high)
    log.info("worst SWR=%.3f, worst loss=%.3f dB", swr_max, loss_max)
    log.info("values:")
    for k, v in zip(KINDS, values):
        if k.startswith("L"):
            log.info("  %s = %.3g µH", k, v * 1e6)
        elif k.startswith("C"):
            log.info("  %s = %.3g pF", k, v * 1e12)
        else:
            log.info("  %s = %.0f Ω (PINNED)", k, v)
    log.info("per-band:")
    losses = -10 * np.log10(np.clip(m["insertion_gain"], 1e-9, 1.0))
    log.info("  %5s  %6s  %9s", "band", "SWR", "loss(dB)")
    for lbl, sw, lo_db in zip(BAND_LABEL, m["swr"], losses):
        log.info("  %5s  %6.2f  %9.2f", lbl, sw, lo_db)
    log.info("")
    log.info("Compare to free-R optimum: SWR=2.50 @ R=437/523 Ω")
    log.info("       penalty for R=500 fixed: ΔSWR = %.2f", swr_max - 2.50)


if __name__ == "__main__":
    main()
