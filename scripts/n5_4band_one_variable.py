#!/usr/bin/env python
"""4-band N=5 matcher — what if ONE component is per-band tunable?

For each of the 5 positions (R, L1, C1, L2, C2) in turn:
  - Treat that component as variable (4 separate values, one per band)
  - Keep the other 4 components fixed across all bands
  - Jointly optimize antenna leg lengths
  - Minimize worst-case SWR across the 4 bands at 2 dB loss budget

Total variables: 4 fixed component log-values + 4 per-band variable
log-values + 2 antenna lengths = 10.

Report which variable-position yields the best SWR improvement vs the
all-fixed baseline (SWR 1.90 at 2 dB loss).
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
LOG_PATH = os.path.join(LOG_DIR, "n5_4band_one_variable.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("onevar")


F_DESIGN_MHZ = np.array([18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["17m", "15m", "12m", "10m"]
Z0 = 50.0
R_RAD = 50.0
F_LOW_BOUNDS = (17.0, 22.0)
F_HIGH_BOUNDS = (24.0, 29.0)
R_SHUNT_OFF = 1.0e6
LOSS_BUDGET_DB = 2.0

KINDS = ["R_shunt", "L_shunt", "C_series", "L_shunt", "C_series"]
N_K = len(KINDS)
N_BANDS = len(F_DESIGN_MHZ)


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
    if kind == "R_shunt":
        return np.array([[1, 0], [1.0 / val, 1]], dtype=complex)
    raise ValueError(kind)


def cascade_at(values: np.ndarray, omega: float) -> np.ndarray:
    M = np.eye(2, dtype=complex)
    for k, v in zip(KINDS, values, strict=True):
        M = M @ section_abcd_at(k, v, omega)
    return M


def z_load_at(omega: float, f_low_mhz: float, f_high_mhz: float) -> complex:
    el_lo = DipoleElement(leg_length_m=leg(f_low_mhz), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(f_high_mhz), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return complex(evaluate_rational(num, den, np.array([1j * omega]))[0])


def evaluate_per_band(
    shared_log_vals: np.ndarray,
    var_log_vals: np.ndarray,
    var_pos: int,
    f_low: float,
    f_high: float,
    r_source: float = Z0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (swr_per_band, loss_db_per_band) for the 4 design freqs.

    shared_log_vals: log values for the 4 NON-variable components, in the
        order of the non-var positions in KINDS
    var_log_vals: log values for the variable position, one per band
    """
    swrs = np.zeros(N_BANDS)
    losses_db = np.zeros(N_BANDS)
    # Build per-band full values
    nonvar_positions = [i for i in range(N_K) if i != var_pos]
    for b, f_mhz in enumerate(F_DESIGN_MHZ):
        omega = 2 * np.pi * f_mhz * 1e6
        full_vals = np.empty(N_K)
        for j, pos in enumerate(nonvar_positions):
            full_vals[pos] = np.exp(shared_log_vals[j])
        full_vals[var_pos] = np.exp(var_log_vals[b])
        z_l = z_load_at(omega, f_low, f_high)
        M = cascade_at(full_vals, omega)
        z_in = (M[0, 0] * z_l + M[0, 1]) / (M[1, 0] * z_l + M[1, 1])
        v_load_over_v1 = z_l / (M[0, 0] * z_l + M[0, 1])
        v1_over_vs = z_in / (z_in + r_source)
        v_ratio = v1_over_vs * v_load_over_v1
        g_in = (z_in - r_source) / (z_in + r_source)
        abs_g = min(abs(g_in), 0.999)
        swrs[b] = (1 + abs_g) / (1 - abs_g)
        m_src = 4.0 * r_source * z_in.real / abs(z_in + r_source) ** 2
        t_total = 4.0 * r_source * z_l.real * abs(v_ratio) ** 2 / abs(z_l) ** 2
        ig = t_total / m_src if m_src > 1e-9 else 0.0
        ig = max(min(ig, 1.0), 1e-9)
        losses_db[b] = -10.0 * np.log10(ig)
    return swrs, losses_db


def bounds_for_var(var_pos: int) -> tuple[list, list, list]:
    """Returns (shared_bounds, var_bounds_per_band, antenna_bounds).

    Each is a list of (low, high) log-tuples in the order they'll appear
    in the optimization vector x.
    """
    shared_b = []
    nonvar = [i for i in range(N_K) if i != var_pos]
    for pos in nonvar:
        k = KINDS[pos]
        if k.startswith("L"):
            shared_b.append((np.log(1e-10), np.log(1e-2)))
        elif k.startswith("C"):
            shared_b.append((np.log(1e-15), np.log(1e-5)))
        else:
            shared_b.append((np.log(0.5), np.log(1e6)))
    var_k = KINDS[var_pos]
    var_b_each = (
        (np.log(1e-10), np.log(1e-2))
        if var_k.startswith("L")
        else (
            (np.log(1e-15), np.log(1e-5)) if var_k.startswith("C") else (np.log(0.5), np.log(1e6))
        )
    )
    var_b = [var_b_each] * N_BANDS
    ant_b = [F_LOW_BOUNDS, F_HIGH_BOUNDS]
    return shared_b, var_b, ant_b


def initial_x(var_pos: int, omega_center: float) -> np.ndarray:
    nonvar = [i for i in range(N_K) if i != var_pos]
    shared = []
    for pos in nonvar:
        k = KINDS[pos]
        if k.startswith("L"):
            shared.append(Z0 / omega_center)
        elif k.startswith("C"):
            shared.append(1.0 / (Z0 * omega_center))
        elif k == "R_shunt":
            shared.append(R_SHUNT_OFF)
        else:
            shared.append(50.0)
    var_k = KINDS[var_pos]
    if var_k.startswith("L"):
        var_init = [Z0 / omega_center] * N_BANDS
    elif var_k.startswith("C"):
        var_init = [1.0 / (Z0 * omega_center)] * N_BANDS
    elif var_k == "R_shunt":
        var_init = [200.0] * N_BANDS
    else:
        var_init = [50.0] * N_BANDS
    return np.concatenate([np.log(shared), np.log(var_init), [19.5, 26.5]])


def unpack(x: np.ndarray, var_pos: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_shared = N_K - 1
    shared = x[:n_shared]
    variable = x[n_shared : n_shared + N_BANDS]
    f_low = float(x[n_shared + N_BANDS])
    f_high = float(x[n_shared + N_BANDS + 1])
    return shared, variable, f_low, f_high


def objective(x: np.ndarray, var_pos: int, lam: float) -> float:
    shared, variable, f_low, f_high = unpack(x, var_pos)
    swrs, losses = evaluate_per_band(shared, variable, var_pos, f_low, f_high)
    max_swr = float(swrs.max())
    max_loss = float(losses.max())
    penalty = max(0.0, max_loss - LOSS_BUDGET_DB) ** 2
    return max_swr + lam * penalty


def solve_for_var_position(var_pos: int) -> dict:
    log.info("")
    log.info(">>> variable position %d : %s", var_pos, KINDS[var_pos])
    omega_center = float(np.exp(np.mean(np.log(2 * np.pi * F_DESIGN_MHZ * 1e6))))
    rng = np.random.default_rng(var_pos)  # different seed per position

    shared_b, var_b, ant_b = bounds_for_var(var_pos)
    bnds = shared_b + var_b + ant_b
    x0 = initial_x(var_pos, omega_center)

    starts = [x0.copy()]
    for _ in range(25):
        s = x0.copy()
        s[: N_K - 1] += rng.normal(0.0, 0.5, size=N_K - 1)
        s[N_K - 1 : N_K - 1 + N_BANDS] += rng.normal(0.0, 0.5, size=N_BANDS)
        s[-2] += rng.normal(0.0, 1.0)
        s[-1] += rng.normal(0.0, 1.0)
        starts.append(s)
    # R-on starts (for R_shunt positions)
    if KINDS[var_pos] == "R_shunt":
        for r_on in [50.0, 150.0, 500.0]:
            for f_lo in [18.0, 20.0]:
                s = x0.copy()
                s[N_K - 1 : N_K - 1 + N_BANDS] = np.log(r_on)
                s[-2] = f_lo
                s[-1] = 26.5
                starts.append(s)
    else:
        # For other variable positions, R is shared — set it ON
        r_pos_in_shared = [
            j
            for j, p in enumerate([i for i in range(N_K) if i != var_pos])
            if KINDS[[i for i in range(N_K) if i != var_pos][j]] == "R_shunt"
        ]
        for r_on in [50.0, 150.0, 500.0]:
            for f_lo in [18.0, 20.0, 21.0]:
                s = x0.copy()
                for j in r_pos_in_shared:
                    s[j] = np.log(r_on)
                s[-2] = f_lo
                s[-1] = 26.5
                starts.append(s)

    log.info("  %d starts", len(starts))
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
                    args=(var_pos, lam),
                    method="L-BFGS-B",
                    bounds=bnds,
                    options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8},
                )
                shared_t, variable_t, f_l, f_h = unpack(res.x, var_pos)
                swrs_t, losses_t = evaluate_per_band(shared_t, variable_t, var_pos, f_l, f_h)
                swr_max_t = float(swrs_t.max())
                loss_max_t = float(losses_t.max())
                feas = loss_max_t <= LOSS_BUDGET_DB + 0.05
                score = swr_max_t if feas else (swr_max_t + 100 * (loss_max_t - LOSS_BUDGET_DB))
                if score < phase_best_score:
                    phase_best_score = score
                    phase_best = res.x
                if score < best_score:
                    best_score = score
                    best_x = res.x
            except Exception:  # noqa: BLE001
                continue
        if phase_best is not None:
            shared_p, variable_p, f_lp, f_hp = unpack(phase_best, var_pos)
            swrs_p, losses_p = evaluate_per_band(shared_p, variable_p, var_pos, f_lp, f_hp)
            log.info(
                "  phase %d/3 lam=%g  SWR=%.3f  loss=%.2f dB  f=(%.2f, %.2f)",
                phase + 1,
                lam,
                float(swrs_p.max()),
                float(losses_p.max()),
                f_lp,
                f_hp,
            )
            starts = [phase_best] + [
                phase_best + rng.normal(0.0, 0.3, size=len(phase_best)) for _ in range(10)
            ]

    shared, variable, f_low, f_high = unpack(best_x, var_pos)
    swrs, losses = evaluate_per_band(shared, variable, var_pos, f_low, f_high)
    log.info(
        "  var=%s DONE in %.1fs  worst SWR=%.3f  worst loss=%.2f dB",
        KINDS[var_pos],
        time.time() - t0,
        float(swrs.max()),
        float(losses.max()),
    )
    return {
        "var_pos": var_pos,
        "var_kind": KINDS[var_pos],
        "shared_log_vals": shared,
        "variable_log_vals": variable,
        "f_low": f_low,
        "f_high": f_high,
        "swrs": swrs,
        "losses": losses,
        "worst_swr": float(swrs.max()),
        "worst_loss": float(losses.max()),
    }


def fmt(kind: str, val: float) -> str:
    if kind.startswith("L"):
        return f"{val * 1e6:.3g} µH"
    if kind.startswith("C"):
        return f"{val * 1e12:.3g} pF"
    return f"{int(val)} Ω"


def main():
    log.info("=" * 72)
    log.info("4-band N=5 matcher — one component variable per band (2 dB budget)")
    log.info("=" * 72)
    log.info("Baseline (all-fixed) reference: worst SWR = 1.90 at 2.0 dB loss")
    log.info("")

    results = []
    for pos in range(N_K):
        r = solve_for_var_position(pos)
        results.append(r)

    log.info("")
    log.info("=" * 80)
    log.info("SUMMARY — which variable-position helps most?")
    log.info("=" * 80)
    log.info(
        "  %6s  %12s  %10s  %10s   per-band SWR / variable values",
        "pos",
        "kind",
        "worst SWR",
        "worst loss",
    )
    for r in results:
        kind = r["var_kind"]
        var_vals = np.exp(r["variable_log_vals"])
        var_str = " / ".join(fmt(kind, v) for v in var_vals)
        swr_str = " / ".join(f"{s:.2f}" for s in r["swrs"])
        log.info(
            "  %6d  %12s  %10.3f  %8.2f dB    SWRs: %s",
            r["var_pos"],
            kind,
            r["worst_swr"],
            r["worst_loss"],
            swr_str,
        )
        log.info("                                                       vals: %s", var_str)
        log.info(
            "    shared: "
            + ", ".join(
                fmt(KINDS[p], np.exp(r["shared_log_vals"][j]))
                for j, p in enumerate([i for i in range(N_K) if i != r["var_pos"]])
            )
            + "  ;  f=(%.2f, %.2f)" % (r["f_low"], r["f_high"])
        )

    # Pick the winner and report cleanly
    best = min(results, key=lambda r: r["worst_swr"])
    log.info("")
    log.info("WINNER: variable position %d (%s)", best["var_pos"], best["var_kind"])
    log.info(
        "  worst SWR = %.3f  (vs 1.90 all-fixed baseline; improvement %.2f)",
        best["worst_swr"],
        1.90 - best["worst_swr"],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(N_K)
    swrs_per_pos = [r["worst_swr"] for r in results]
    ax.bar(xs, swrs_per_pos, color=["C2" if r is best else "C0" for r in results])
    ax.axhline(1.90, color="grey", ls="--", lw=0.8, label="all-fixed (SWR 1.90)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}: {KINDS[i]}" for i in range(N_K)], rotation=15)
    ax.set_ylabel("worst-band SWR")
    ax.set_title(
        "4-band N=5 matcher: per-band variation of one component\n"
        "(2 dB loss budget, joint length opt)"
    )
    for x, swr in zip(xs, swrs_per_pos):
        ax.text(x, swr + 0.02, f"{swr:.2f}", ha="center", fontsize=10)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(1.0, 2.2)
    fig.tight_layout()
    fig.savefig("n5_4band_one_variable.png", dpi=140, bbox_inches="tight")
    log.info("wrote n5_4band_one_variable.png")
    log.info("log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
