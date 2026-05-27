#!/usr/bin/env python
"""Loss-budget sweep for the N=5 matcher (R + L + C + L + C).

The complexity sweep showed N=5 is the "sweet spot" — 5 components
(one resistor + two inductors + two capacitors) recover SWR 2.59 at
2 dB loss, within 0.08 SWR units of the best N=8 result. This script
asks: at the N=5 topology, what's the Pareto curve over loss budget?

For each budget B, solve:
  minimize worst-band SWR  s.t.  worst-band loss ≤ B
jointly over the 5 matcher values AND the 2 antenna leg lengths.

Reports both the Pareto curve and per-band breakdown at each point.
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
LOG_PATH = os.path.join(LOG_DIR, "n5_loss_sweep.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("n5sweep")


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0
R_RAD = 50.0
F_LOW_BOUNDS = (14.30, 18.16)
F_HIGH_BOUNDS = (24.97, 28.47)
R_SHUNT_OFF = 1.0e6

KINDS = ["R_shunt", "L_shunt", "C_series", "L_shunt", "C_series"]
LOSS_BUDGETS = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


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


def evaluate(matcher_vals, f_low, f_high, omegas):
    z_load = z_load_of(f_low, f_high, omegas)
    M = cascade_abcd(matcher_vals, omegas)
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
        "g_in": g_in,
    }


def metrics_max(m):
    return (
        float(m["swr"].max()),
        float((-10 * np.log10(np.clip(m["insertion_gain"], 1e-9, 1.0))).max()),
    )


def initial_x(omega_center):
    init = []
    for k in KINDS:
        if k.startswith("L"):
            init.append(Z0 / omega_center)
        elif k.startswith("C"):
            init.append(1.0 / (Z0 * omega_center))
        elif k == "R_shunt":
            init.append(R_SHUNT_OFF)
        else:
            init.append(50.0)
    return np.concatenate([np.log(init), [16.0, 26.5]])


def bounds_for():
    b = []
    for k in KINDS:
        if k.startswith("L"):
            b.append((np.log(1e-10), np.log(1e-2)))
        elif k.startswith("C"):
            b.append((np.log(1e-15), np.log(1e-5)))
        else:
            b.append((np.log(0.5), np.log(1e6)))
    b.append(F_LOW_BOUNDS)
    b.append(F_HIGH_BOUNDS)
    return b


def objective(x, omegas, budget, lam):
    n_k = len(KINDS)
    values = np.exp(x[:n_k])
    f_low = float(x[n_k])
    f_high = float(x[n_k + 1])
    m = evaluate(values, f_low, f_high, omegas)
    swr_max, loss_max = metrics_max(m)
    return swr_max + lam * max(0.0, loss_max - budget) ** 2


def solve(budget, omegas, label=""):
    omega_center = float(np.exp(np.mean(np.log(omegas))))
    rng = np.random.default_rng(0)
    x0 = initial_x(omega_center)
    bnds = bounds_for()
    n_k = len(KINDS)

    starts = [x0.copy()]
    for _ in range(20):
        s = x0.copy()
        for i, k in enumerate(KINDS):
            scale = 0.5 if k.startswith(("L", "C")) else 2.0
            s[i] += rng.normal(0.0, scale)
        s[n_k] += rng.normal(0.0, 1.0)
        s[n_k + 1] += rng.normal(0.0, 1.0)
        starts.append(s)
    r_positions = [i for i, k in enumerate(KINDS) if k.startswith("R")]
    for r_on in [30.0, 80.0, 200.0, 500.0]:
        for f_lo in [15.0, 16.0, 17.0]:
            for f_hi in [25.0, 26.0, 27.0]:
                s = x0.copy()
                for ri in r_positions:
                    s[ri] = np.log(r_on)
                s[n_k] = f_lo
                s[n_k + 1] = f_hi
                starts.append(s)

    log.info("  %s [budget=%.2f dB] %d starts", label, budget, len(starts))
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
                    args=(omegas, budget, lam),
                    method="L-BFGS-B",
                    bounds=bnds,
                    options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8},
                )
                v_t = np.exp(res.x[:n_k])
                f_l = float(res.x[n_k])
                f_h = float(res.x[n_k + 1])
                m_t = evaluate(v_t, f_l, f_h, omegas)
                swr_t, loss_t = metrics_max(m_t)
                feas = loss_t <= budget + 0.05
                score = swr_t if feas else (swr_t + 100 * (loss_t - budget))
                if score < phase_best_score:
                    phase_best_score = score
                    phase_best = res.x
                if score < best_score:
                    best_score = score
                    best_x = res.x
            except Exception:  # noqa: BLE001
                continue
        if phase_best is not None:
            starts = [phase_best] + [
                phase_best + rng.normal(0.0, 0.3, size=len(phase_best)) for _ in range(10)
            ]

    matcher_vals = np.exp(best_x[:n_k])
    f_low = float(best_x[n_k])
    f_high = float(best_x[n_k + 1])
    metrics = evaluate(matcher_vals, f_low, f_high, omegas)
    swr, loss = metrics_max(metrics)
    log.info(
        "  %s [budget=%.2f] DONE in %.1fs  SWR=%.3f  loss=%.2f dB  f=(%.2f, %.2f)",
        label,
        budget,
        time.time() - t0,
        swr,
        loss,
        f_low,
        f_high,
    )
    return matcher_vals, f_low, f_high, metrics


def main():
    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    log.info("=" * 72)
    log.info("N=5 matcher loss-budget sweep")
    log.info("topology: %s", " - ".join(KINDS))
    log.info("antenna: inverted-V R_rad=%.0f, lengths jointly optimized", R_RAD)
    log.info("=" * 72)

    results = []
    for budget in LOSS_BUDGETS:
        matcher_vals, f_low, f_high, m = solve(budget, omegas, label=f"B{budget:.1f}")
        swr, loss = metrics_max(m)
        results.append(
            {
                "budget": budget,
                "matcher_vals": matcher_vals,
                "f_low": f_low,
                "f_high": f_high,
                "swr": swr,
                "loss": loss,
                "per_band_swr": m["swr"],
                "per_band_loss": -10 * np.log10(np.clip(m["insertion_gain"], 1e-9, 1.0)),
            }
        )

    log.info("")
    log.info("=" * 80)
    log.info("SUMMARY — N=5 matcher Pareto frontier")
    log.info("=" * 80)
    log.info(
        "  %8s   %10s  %10s   %8s  %8s", "budget", "worst SWR", "worst loss", "f_low", "f_high"
    )
    for r in results:
        log.info(
            "  %6.2f dB   %10.3f  %8.2f dB   %8.3f  %8.3f",
            r["budget"],
            r["swr"],
            r["loss"],
            r["f_low"],
            r["f_high"],
        )

    log.info("")
    log.info("Per-band SWR at each budget:")
    log.info("  %8s   %6s %6s %6s %6s %6s", "budget", *BAND_LABEL)
    for r in results:
        log.info("  %6.2f dB   %6.2f %6.2f %6.2f %6.2f %6.2f", r["budget"], *r["per_band_swr"])

    log.info("")
    log.info("Per-band loss(dB) at each budget:")
    log.info("  %8s   %6s %6s %6s %6s %6s", "budget", *BAND_LABEL)
    for r in results:
        log.info("  %6.2f dB   %6.2f %6.2f %6.2f %6.2f %6.2f", r["budget"], *r["per_band_loss"])

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    achieved_loss = [r["loss"] for r in results]
    achieved_swr = [r["swr"] for r in results]
    axes[0].plot(achieved_loss, achieved_swr, "o-", color="C0", lw=2, ms=10)
    for r in results:
        axes[0].annotate(
            f"  ≤{r['budget']:.1f} dB", xy=(r["loss"], r["swr"]), fontsize=9, va="center"
        )
    axes[0].axhline(2.0, color="grey", ls=":", lw=0.8)
    axes[0].set_xlabel("worst-band insertion loss (dB)")
    axes[0].set_ylabel("worst-band SWR")
    axes[0].set_yscale("log")
    axes[0].set_title("N=5 (R+L+C+L+C) Pareto frontier\njoint length opt, inverted-V R_rad=50")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].set_ylim(1, 6)

    # per-band SWR per budget heatmap-like display
    budgets = [r["budget"] for r in results]
    per_band_swr_matrix = np.array([r["per_band_swr"] for r in results])
    im = axes[1].imshow(
        per_band_swr_matrix.T,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=1.0,
        vmax=3.5,
        extent=[budgets[0], budgets[-1], 4.5, -0.5],
    )
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(BAND_LABEL)
    axes[1].set_xlabel("loss budget (dB)")
    axes[1].set_title("Per-band SWR as a function of loss budget")
    plt.colorbar(im, ax=axes[1], label="SWR")
    for i, r in enumerate(results):
        for j in range(5):
            axes[1].text(
                r["budget"],
                j,
                f"{r['per_band_swr'][j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if r["per_band_swr"][j] > 2.5 else "black",
            )
    fig.tight_layout()
    fig.savefig("n5_loss_sweep.png", dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote n5_loss_sweep.png")
    log.info("log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
