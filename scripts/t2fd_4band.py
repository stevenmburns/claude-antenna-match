#!/usr/bin/env python
"""T2FD (Tilted Terminated Folded Dipole) sized for the 4 bands and compared
to the L-tuner from yesterday's work.

Model: the T2FD as a uniform transmission line of characteristic impedance
Z_0, length L_m, terminated in R_term at the far end, fed at the near end
through a balun_ratio:1 balun. The TL model is the standard first-cut
engineering approximation — it captures the impedance smoothing but does
not by itself say how much power is radiated vs dissipated in R_term.

For efficiency: a lossless TL passes ALL forward power to R_term, so the
"raw TL model" predicts 100% loss. Real T2FDs radiate because the line
IS a radiating structure with an effective attenuation constant α (the
radiation constant). We model this with a single tunable α (Np/m) and
compute:
  - power into termination R_term (heat)
  - power lost in the line (interpreted as RADIATED — the useful part)

Fixed Z_0 ∈ {300, 450, 600} Ω (typical for folded dipole structures);
optimize L, R_term, balun_ratio over a few discrete choices.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "t2fd_4band.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("t2fd")


F_DESIGN_MHZ = np.array([18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["17m", "15m", "12m", "10m"]
Z0_COAX = 50.0
C_LIGHT = 299_792_458.0
VF = 0.95  # velocity factor for wire-in-air

# ---- Free parameters ----
# L_m: total antenna length
# R_term: terminating resistor (Ω)
# alpha_per_m: radiation attenuation per meter (Np/m). Typical: 0.02-0.1
#   for HF-resonant dipole-like structures. Higher = more radiation per
#   unit length = more "efficient" antenna.
# Z_0_LINE: characteristic impedance of the folded structure
# BALUN_RATIO: impedance step-down from antenna terminals to coax


def gamma_in_lossy_line(
    omega: float, Z_0: float, length_m: float, R_term: float, alpha_per_m: float, balun_ratio: float
):
    """Return (Z_in_coax, P_term/P_in_line, P_radiated/P_in_line).

    Treats the T2FD as a uniform lossy TL terminated in R_term. The
    'loss' α represents radiation, NOT ohmic dissipation: power
    'lost' to α is the power radiated. Power 'lost' to R_term is
    actual heat.
    """
    beta = omega * VF / C_LIGHT
    gamma_line = alpha_per_m + 1j * beta
    gl = gamma_line * length_m
    # tanh of complex
    tanh_gl = np.tanh(gl)
    Z_in_terminals = Z_0 * (R_term + Z_0 * tanh_gl) / (Z_0 + R_term * tanh_gl)
    Z_in_coax = Z_in_terminals / balun_ratio

    # Power partition: for a lossy TL, the input power splits between
    # what's radiated (lost in α) and what reaches the termination.
    # For a matched line (R_term = Z_0), this is easy:
    #   P_term/P_in = exp(-2 α L)
    #   P_radiated/P_in = 1 - exp(-2 α L)
    # For a mismatched line we'd need to integrate forward and reflected
    # waves. Use the matched-line approximation; it's accurate when
    # |R_term - Z_0|/|R_term + Z_0| is small.
    p_term_over_p_in = np.exp(-2 * alpha_per_m * length_m)
    p_rad_over_p_in = 1.0 - p_term_over_p_in
    return Z_in_coax, p_term_over_p_in, p_rad_over_p_in


def swr(z_in: complex) -> float:
    g = (z_in - Z0_COAX) / (z_in + Z0_COAX)
    abs_g = min(abs(g), 0.9999)
    return (1 + abs_g) / (1 - abs_g)


def evaluate(omega, Z_0, length_m, R_term, alpha, balun):
    z_in, p_term, p_rad = gamma_in_lossy_line(omega, Z_0, length_m, R_term, alpha, balun)
    s = swr(z_in)
    g_in = (z_in - Z0_COAX) / (z_in + Z0_COAX)
    # Match factor: 1 - |Γ_in|² (fraction of source-available power into antenna)
    m_src = 1 - abs(g_in) ** 2
    # Efficiency: of the power that enters, fraction radiated
    eff_into_line = float(p_rad)
    # End-to-end "system efficiency" = m_src × eff_into_line
    sys_eff = m_src * eff_into_line
    return s, m_src, eff_into_line, sys_eff


def objective(x, Z_0, alpha, balun, freq_w):
    log_L, log_R = x
    L_m = np.exp(log_L)
    R_term = np.exp(log_R)
    swrs = []
    for w in freq_w:
        s, _, _, _ = evaluate(w, Z_0, L_m, R_term, alpha, balun)
        swrs.append(s)
    return float(max(swrs))


def solve(Z_0, alpha, balun):
    freq_w = 2 * np.pi * F_DESIGN_MHZ * 1e6
    # Initial: L = 1/2 wavelength at center freq, R_term ≈ Z_0
    omega_center = 2 * np.pi * 22e6
    lambda_center = 2 * np.pi * C_LIGHT * VF / omega_center
    x0 = np.array([np.log(lambda_center / 2), np.log(Z_0)])
    bnds = [
        (np.log(2.0), np.log(20.0)),  # length 2-20 m
        (np.log(50.0), np.log(2000.0)),  # R_term 50-2000 Ω
    ]
    rng = np.random.default_rng(42)
    starts = [x0]
    for _ in range(30):
        s = x0 + rng.normal(0, [0.5, 0.7])
        starts.append(s)

    best_x = None
    best_score = np.inf
    for s in starts:
        s_clip = np.array([np.clip(s[i], b[0] + 0.01, b[1] - 0.01) for i, b in enumerate(bnds)])
        try:
            res = minimize(
                objective,
                s_clip,
                args=(Z_0, alpha, balun, freq_w),
                method="L-BFGS-B",
                bounds=bnds,
                options={"maxiter": 300, "ftol": 1e-11},
            )
            if res.fun < best_score:
                best_score = float(res.fun)
                best_x = res.x
        except Exception:
            continue
    L_m = np.exp(best_x[0])
    R_term = np.exp(best_x[1])
    return L_m, R_term, best_score


def main():
    log.info("=" * 78)
    log.info("T2FD 4-band design comparison vs L-tuner")
    log.info("=" * 78)
    log.info("Design freqs: %s MHz", F_DESIGN_MHZ.tolist())

    # Sweep over a few reasonable parameter combinations
    Z0_choices = [300.0, 450.0, 600.0]
    alpha_choices = [0.02, 0.05, 0.10]  # Np/m  (rough radiation losses)
    balun_choices = [4.0, 9.0]

    all_results = []
    for Z_0 in Z0_choices:
        for alpha in alpha_choices:
            for balun in balun_choices:
                L_m, R_term, worst_swr = solve(Z_0, alpha, balun)
                freq_w = 2 * np.pi * F_DESIGN_MHZ * 1e6
                per_band = []
                for w in freq_w:
                    s, m_src, eff_line, sys_eff = evaluate(w, Z_0, L_m, R_term, alpha, balun)
                    per_band.append(
                        {"swr": s, "m_src": m_src, "eff_line": eff_line, "sys_eff": sys_eff}
                    )
                all_results.append(
                    {
                        "Z_0": Z_0,
                        "alpha": alpha,
                        "balun": balun,
                        "L_m": L_m,
                        "R_term": R_term,
                        "worst_swr": worst_swr,
                        "per_band": per_band,
                    }
                )

    # Sort by worst SWR
    all_results.sort(key=lambda r: r["worst_swr"])
    log.info("")
    log.info(
        "%-8s %-10s %-7s %-7s %-9s %-9s %-9s %-9s",
        "Z_0",
        "α (Np/m)",
        "balun",
        "L(m)",
        "R(Ω)",
        "worst SWR",
        "rad eff",
        "sys eff",
    )
    log.info("-" * 80)
    for r in all_results[:9]:  # top 9
        # Average system efficiency across 4 bands
        avg_eff = np.mean([b["sys_eff"] for b in r["per_band"]])
        eff_db = -10 * np.log10(avg_eff) if avg_eff > 0 else 999
        log.info(
            "%-8.0f %-10.3f %-7.0f %-7.2f %-9.0f %-9.2f %-9.1f%% %-9.1f%%  (avg loss %.2f dB)",
            r["Z_0"],
            r["alpha"],
            r["balun"],
            r["L_m"],
            r["R_term"],
            r["worst_swr"],
            100 * np.mean([b["eff_line"] for b in r["per_band"]]),
            100 * avg_eff,
            eff_db,
        )

    log.info("")
    log.info("Best T2FD (lowest worst-SWR) detailed per-band:")
    best = all_results[0]
    log.info(
        "  Z_0=%.0f Ω, α=%.3f Np/m, balun=%.0f:1, L=%.2f m, R_term=%.0f Ω",
        best["Z_0"],
        best["alpha"],
        best["balun"],
        best["L_m"],
        best["R_term"],
    )
    log.info(
        "  %-5s  %-7s  %-9s  %-12s  %-12s  %-12s",
        "band",
        "SWR",
        "1-|Γ|²",
        "rad eff",
        "sys eff (linear)",
        "sys loss (dB)",
    )
    for lbl, b in zip(BAND_LABEL, best["per_band"]):
        log.info(
            "  %-5s  %-7.2f  %-9.3f  %-12.1f%%  %-12.1f%%  %-12.2f",
            lbl,
            b["swr"],
            b["m_src"],
            100 * b["eff_line"],
            100 * b["sys_eff"],
            -10 * np.log10(b["sys_eff"]) if b["sys_eff"] > 0 else 999,
        )

    log.info("")
    log.info("COMPARISON to L-tuner (yesterday's work):")
    log.info("  L-tuner (2-cap): worst SWR 1.32, 0 dB loss, 4 bands.")
    log.info("  L-tuner (4-cap): worst SWR 1.15, 0 dB loss, 4 bands.")
    log.info(
        "  Best T2FD here:  worst SWR %.2f, avg loss %.2f dB.",
        best["worst_swr"],
        -10 * np.log10(np.mean([b["sys_eff"] for b in best["per_band"]])),
    )

    # Plot: for each of a few representative parameter combos, plot the
    # SWR vs frequency across the design range
    fig, ax = plt.subplots(figsize=(11, 6))
    f_sweep = np.linspace(15, 32, 400)
    w_sweep = 2 * np.pi * f_sweep * 1e6

    # Pick 3 interesting points: lowest-SWR, highest-eff, and a middle case
    plotted = [all_results[0]]  # best SWR
    plotted.append(
        max(all_results, key=lambda r: np.mean([b["sys_eff"] for b in r["per_band"]]))
    )  # best eff
    if len(all_results) > 6:
        plotted.append(all_results[len(all_results) // 2])  # mid-table

    for i, r in enumerate(plotted):
        swrs = []
        for w in w_sweep:
            s, _, _, _ = evaluate(w, r["Z_0"], r["L_m"], r["R_term"], r["alpha"], r["balun"])
            swrs.append(s)
        label = (
            f"Z₀={r['Z_0']:.0f}, α={r['alpha']:.2f}, balun={r['balun']:.0f}:1, "
            f"L={r['L_m']:.1f}m, R={r['R_term']:.0f}Ω"
        )
        ax.semilogy(f_sweep, swrs, lw=1.8, label=label)
    for f in F_DESIGN_MHZ:
        ax.axvline(f, color="grey", ls=":", lw=0.5, alpha=0.6)
    ax.axhline(2.0, color="grey", ls="--", lw=0.7)
    ax.axhline(1.32, color="C2", ls="-.", lw=1.0, label="L-tuner (2-cap) ref SWR 1.32")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("SWR (log)")
    ax.set_ylim(1, 10)
    ax.set_title("T2FD SWR vs frequency for several parameter choices vs L-tuner")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("t2fd_4band.png", dpi=140, bbox_inches="tight")
    log.info("")
    log.info("wrote t2fd_4band.png")


if __name__ == "__main__":
    main()
