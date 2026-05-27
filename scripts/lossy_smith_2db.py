#!/usr/bin/env python
"""Smith chart for the joint-length + lossy matcher optimum at ~2 dB worst-case loss.

Uses the corrected metrics_summary (the prior script reported the MIN
per-band loss as 'worst loss'; this script reports the MAX as it should).

Single targeted optimization: minimize worst-band SWR subject to
worst-band insertion loss ≤ 2 dB, with antenna leg lengths AND matcher
elements all free.

Plots:
  - SWR sweep (linear-freq scan over the HF band) for the resulting matcher
  - Smith chart of Γ_in(jω) with the 5 design freqs marked
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
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
LOG_PATH = os.path.join(LOG_DIR, "lossy_smith_2db_invV.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("smith2db")


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0
F_LOW_BOUNDS = (14.30, 18.16)
F_HIGH_BOUNDS = (24.97, 28.47)
LOSS_BUDGET_DB = 2.0

# Inverted-V geometry: R_rad ≈ 50 Ω at typical apex angle (~120°)
# instead of ~65 Ω for a flat dipole. This natively matches a 50 Ω
# feedline at each element's resonance, freeing the matcher to work
# on the harder midband swing.
R_RAD = 50.0

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


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


# ---------- ABCD cascade ----------


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


def cascade_abcd(kinds: list[str], values: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    n = len(omegas)
    M = np.array([[np.ones(n), np.zeros(n)], [np.zeros(n), np.ones(n)]], dtype=complex)
    for kind, val in zip(kinds, values, strict=True):
        M = np.einsum("ilk,ljk->ijk", M, section_abcd(kind, val, omegas))
    return M


def z_load_of(f_low_mhz: float, f_high_mhz: float, omegas: np.ndarray) -> np.ndarray:
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
        "g_in": g_in,
        "z_in": z_in,
        "z_load": z_load,
        "insertion_gain": np.clip(insertion_gain, 0.0, 1.0),
        "t_total": np.clip(t_total, 0.0, 1.0),
    }


def metrics(metrics_d):
    """Returns (worst_swr, worst_loss_db) — corrected max-loss bug."""
    swr = float(metrics_d["swr"].max())
    loss_db = float((-10 * np.log10(np.clip(metrics_d["insertion_gain"], 1e-9, 1.0))).max())
    return swr, loss_db


# ---------- Optimization ----------


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
    n = len(kinds)
    matcher_vals = np.exp(x[:n])
    f_low = float(x[n])
    f_high = float(x[n + 1])
    m = evaluate(matcher_vals, kinds, f_low, f_high, omegas, r_source)
    swr_max, loss_max = metrics(m)
    penalty = max(0.0, loss_max - budget) ** 2
    return swr_max + lam * penalty


def solve():
    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    omega_center = float(np.exp(np.mean(np.log(omegas))))
    rng = np.random.default_rng(0)
    x0 = initial_x(LOSSY_TOPOLOGY, omega_center)
    bnds = bounds_for(LOSSY_TOPOLOGY)
    n_k = len(LOSSY_TOPOLOGY)

    # Build starts: x0, plus random jitters, plus EXPLICIT "R turned on" starts
    starts = [x0.copy()]
    for _ in range(40):
        s = x0.copy()
        for i, k in enumerate(LOSSY_TOPOLOGY):
            scale = 0.5 if k.startswith(("L", "C")) else 2.0
            s[i] += rng.normal(0.0, scale)
        s[n_k] += rng.normal(0.0, 1.0)
        s[n_k + 1] += rng.normal(0.0, 1.0)
        starts.append(s)
    # Explicit "R on" starts to escape the lossless trap
    for r_on in [30.0, 80.0, 200.0, 500.0]:
        for f_lo in [15.0, 16.0, 17.0, 18.0]:
            for f_hi in [25.0, 26.0, 27.0]:
                s = x0.copy()
                # Set both R_shunt vars to r_on
                r_positions = [i for i, k in enumerate(LOSSY_TOPOLOGY) if k == "R_shunt"]
                for ri in r_positions:
                    s[ri] = np.log(r_on)
                s[n_k] = f_lo
                s[n_k + 1] = f_hi
                starts.append(s)
    log.info("multistart: %d starts", len(starts))

    best_x = None
    best_score = np.inf
    t0 = time.time()
    for phase, lam in enumerate([10.0, 100.0, 1000.0]):
        phase_best = None
        phase_best_score = np.inf
        for s_idx, s in enumerate(starts):
            s_clip = np.array([np.clip(s[i], b[0] + 0.01, b[1] - 0.01) for i, b in enumerate(bnds)])
            try:
                res = minimize(
                    objective,
                    s_clip,
                    args=(LOSSY_TOPOLOGY, omegas, Z0, LOSS_BUDGET_DB, lam),
                    method="L-BFGS-B",
                    bounds=bnds,
                    options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-8},
                )
                matcher_vals = np.exp(res.x[:n_k])
                f_low = float(res.x[n_k])
                f_high = float(res.x[n_k + 1])
                m_t = evaluate(matcher_vals, LOSSY_TOPOLOGY, f_low, f_high, omegas)
                swr_t, loss_t = metrics(m_t)
                feasible = loss_t <= LOSS_BUDGET_DB + 0.05
                score = swr_t if feasible else (swr_t + 100 * (loss_t - LOSS_BUDGET_DB))
                if score < phase_best_score:
                    phase_best_score = score
                    phase_best = res.x
                if score < best_score:
                    best_score = score
                    best_x = res.x
            except Exception as e:  # noqa: BLE001
                log.debug("phase%d start%d: %s", phase, s_idx, e)
                continue
        m_p = evaluate(
            np.exp(phase_best[:n_k]),
            LOSSY_TOPOLOGY,
            float(phase_best[n_k]),
            float(phase_best[n_k + 1]),
            omegas,
        )
        swr_p, loss_p = metrics(m_p)
        log.info(
            "phase %d/3 lam=%g done  score=%.3f  SWR=%.3f  loss=%.2f dB  f=(%.2f, %.2f)",
            phase + 1,
            lam,
            phase_best_score,
            swr_p,
            loss_p,
            float(phase_best[n_k]),
            float(phase_best[n_k + 1]),
        )
        starts = [phase_best] + [
            phase_best + rng.normal(0.0, 0.3, size=len(phase_best)) for _ in range(15)
        ]

    matcher_vals = np.exp(best_x[:n_k])
    f_low = float(best_x[n_k])
    f_high = float(best_x[n_k + 1])
    log.info("DONE in %.1fs", time.time() - t0)
    return matcher_vals, f_low, f_high


def main():
    log.info("Solving: min worst SWR  s.t.  worst loss ≤ %.1f dB  (joint length)", LOSS_BUDGET_DB)
    log.info("topology: %s", " - ".join(LOSSY_TOPOLOGY))
    matcher_vals, f_low, f_high = solve()
    omegas_d = 2 * np.pi * F_DESIGN_MHZ * 1e6
    m_d = evaluate(matcher_vals, LOSSY_TOPOLOGY, f_low, f_high, omegas_d)
    swr_d, loss_d = metrics(m_d)
    log.info("")
    log.info("=== Optimum at budget %.1f dB ===", LOSS_BUDGET_DB)
    log.info(
        "f_low=%.3f MHz, f_high=%.3f MHz  (legs %.3f m, %.3f m)",
        f_low,
        f_high,
        leg(f_low),
        leg(f_high),
    )
    log.info("worst SWR = %.3f    worst loss = %.3f dB", swr_d, loss_d)
    log.info("matcher values:")
    for k, v in zip(LOSSY_TOPOLOGY, matcher_vals):
        unit = "µH" if k.startswith("L") else ("pF" if k.startswith("C") else "Ω")
        scale = 1e6 if k.startswith("L") else (1e12 if k.startswith("C") else 1.0)
        log.info("  %s = %.3g %s", k, v * scale, unit)
    log.info("per-band:")
    log.info("  %5s  %6s  %9s  %7s", "band", "SWR", "loss(dB)", "T(dB)")
    losses = -10 * np.log10(np.clip(m_d["insertion_gain"], 1e-9, 1.0))
    for lbl, sw, lo_db, t in zip(BAND_LABEL, m_d["swr"], losses, m_d["t_total"]):
        log.info("  %5s  %6.2f  %9.2f  %7.2f", lbl, sw, lo_db, 10 * np.log10(max(t, 1e-9)))

    # ---- Sweep + Smith chart ----
    f_sweep = np.linspace(10e6, 32e6, 3000)
    w_sweep = 2 * np.pi * f_sweep
    m_sweep = evaluate(matcher_vals, LOSSY_TOPOLOGY, f_low, f_high, w_sweep)
    g_sweep = m_sweep["g_in"]
    swr_sweep = m_sweep["swr"]
    losses_sweep = -10 * np.log10(np.clip(m_sweep["insertion_gain"], 1e-9, 1.0))

    fig = plt.figure(figsize=(15, 7))
    ax_l = fig.add_subplot(1, 2, 1)
    ax_sm = fig.add_subplot(1, 2, 2)

    # --- SWR + loss panel ---
    ax_l.semilogy(f_sweep / 1e6, swr_sweep, color="C3", lw=1.6, label="SWR")
    ax_l.axhline(2.0, color="grey", ls=":", lw=0.7)
    ax_l.set_ylabel("SWR  (log)", color="C3")
    ax_l.tick_params(axis="y", labelcolor="C3")
    ax_l.set_ylim(1, 30)
    for f, lbl in zip(F_DESIGN_MHZ, BAND_LABEL):
        ax_l.axvline(f, color="grey", ls=":", lw=0.5, alpha=0.6)
        ax_l.text(
            f,
            0.97,
            lbl,
            transform=ax_l.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="grey",
        )
    ax_l.scatter(
        F_DESIGN_MHZ,
        m_d["swr"],
        color="C1",
        s=70,
        zorder=5,
        edgecolors="black",
        label="design freqs",
    )
    ax_l.set_xlabel("frequency (MHz)")
    ax_l.legend(loc="upper left", fontsize=9)
    ax_l.grid(True, which="both", alpha=0.3)

    # Overlay insertion loss on right axis
    ax_l2 = ax_l.twinx()
    ax_l2.plot(f_sweep / 1e6, losses_sweep, color="C0", lw=1.3, alpha=0.7, label="loss (dB)")
    ax_l2.scatter(F_DESIGN_MHZ, losses, color="C0", s=50, zorder=5, edgecolors="black")
    ax_l2.axhline(
        LOSS_BUDGET_DB, color="C0", ls="--", lw=0.7, alpha=0.6, label=f"{LOSS_BUDGET_DB} dB budget"
    )
    ax_l2.set_ylabel("insertion loss (dB)", color="C0")
    ax_l2.tick_params(axis="y", labelcolor="C0")
    ax_l2.set_ylim(0, 8)
    ax_l2.legend(loc="upper right", fontsize=9)

    ax_l.set_title(
        f"SWR + loss vs frequency\n"
        f"f_low={f_low:.2f} MHz, f_high={f_high:.2f} MHz  "
        f"worst SWR={swr_d:.2f}  worst loss={loss_d:.2f} dB"
    )

    # --- Smith chart ---
    s_sweep = g_sweep.reshape(-1, 1, 1)
    net = rf.Network(
        frequency=rf.Frequency.from_f(f_sweep / 1e6, unit="MHz"),
        s=s_sweep,
        z0=Z0,
        name="lossy 2dB",
    )
    net.plot_s_smith(ax=ax_sm, draw_labels=False, show_legend=False, color="C3", lw=1.5)

    g_design = m_d["g_in"]
    net_d = rf.Network(
        frequency=rf.Frequency.from_f(F_DESIGN_MHZ, unit="MHz"),
        s=g_design.reshape(-1, 1, 1),
        z0=Z0,
    )
    net_d.plot_s_smith(
        ax=ax_sm,
        draw_labels=False,
        show_legend=False,
        marker="o",
        markersize=11,
        linestyle="",
        color="C1",
    )
    for f, g, lbl in zip(F_DESIGN_MHZ, g_design, BAND_LABEL):
        ax_sm.annotate(
            f"{lbl}",
            xy=(g.real, g.imag),
            xytext=(11, 11),
            textcoords="offset points",
            fontsize=12,
            color="C1",
            fontweight="bold",
        )

    # SWR=2 circle for reference
    swr2_radius = 1.0 / 3.0  # |Γ| = 1/3 → SWR 2
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_sm.plot(swr2_radius * np.cos(theta), swr2_radius * np.sin(theta), "k--", lw=0.6, alpha=0.5)

    ax_sm.set_title(
        "Smith chart — coupled+lossy optimum (~2 dB loss budget)\n"
        "red trace = Γ_in(jω) sweep 10-32 MHz, dashed circle = SWR 2"
    )
    fig.tight_layout()
    out = "lossy_smith_2db_invV.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
