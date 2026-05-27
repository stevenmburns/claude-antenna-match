#!/usr/bin/env python
"""Lossy matching network analysis on the 5-band 2-element fan dipole.

The Bode-Fano bound (v2) says any LOSSLESS matcher hits a worst-case
SWR floor of ~3.7 across the 5 amateur HF bands for the 2-element fan
dipole. Lossy matchers can exceed that bound *in SWR terms* by
dissipating some signal. This script quantifies the tradeoff.

Three configurations:
  (a) Lossless LC ladder — the v1 baseline (Bode-Fano-constrained)
  (b) Lossless LC ladder + a shunt input resistor — a one-knob lossy add
  (c) Joint-optimized LCR ladder — best SWR vs insertion-loss tradeoff

Reports per band:
  - SWR at the source (what the radio sees)
  - Insertion loss in dB (signal lost to heat in the matcher)
  - Total transducer gain T in dB (= source match * insertion gain)
"""

from __future__ import annotations

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


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


# ---------- ABCD-matrix cascade with resistive elements ----------
#
# Each section is one of:
#   L_series, C_series, R_series  -> series impedance Z_e
#   L_shunt,  C_shunt,  R_shunt   -> shunt admittance Y_e = 1/Z_e
# Source-to-load order; cascade is M_total = M_1 · M_2 · ... · M_n.


SECTION_KINDS = (
    "L_series",
    "C_series",
    "R_series",
    "L_shunt",
    "C_shunt",
    "R_shunt",
)


def section_abcd(kind: str, value: float, omegas: np.ndarray) -> np.ndarray:
    """Return a 2x2 ABCD array, shape (2, 2, n_omega), at each ω."""
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
    """Return ABCD of the cascade, shape (2, 2, n_omega).

    `kinds` is source-to-load order; the cascade matrix is the product.
    """
    n = len(omegas)
    M = np.array([[np.ones(n), np.zeros(n)], [np.zeros(n), np.ones(n)]], dtype=complex)
    for kind, val in zip(kinds, values, strict=True):
        Mi = section_abcd(kind, val, omegas)
        # Per-ω matrix multiply: M_new[i,j,k] = sum_l M[i,l,k] * Mi[l,j,k]
        M_new = np.einsum("ilk,ljk->ijk", M, Mi)
        M = M_new
    return M


def z_in_and_voltage_ratio(
    M: np.ndarray, z_load: np.ndarray, r_source: float
) -> tuple[np.ndarray, np.ndarray]:
    """From ABCD M and load Z_L, return (Z_in, V_load/V_source).

    Derivation: with port 2 terminated in Z_L we have V2 = Z_L·I2.
    [V1; I1] = M · [V2; I2] = M · [Z_L · I2; I2]
    so V1 = (A·Z_L + B)·I2 and I1 = (C·Z_L + D)·I2.
    Z_in = V1/I1 = (A·Z_L + B) / (C·Z_L + D).
    With Thévenin source V_s, R_s: V1 = V_s · Z_in / (Z_in + R_s),
    so V_load = V2 = (V1) / ((A·Z_L + B) / Z_L) = V1 · Z_L / (A·Z_L + B).
    V_load / V_s = (Z_in / (Z_in + R_s)) · (Z_L / (A·Z_L + B)).
    """
    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]
    z_in = (A * z_load + B) / (C * z_load + D)
    v_load_over_v1 = z_load / (A * z_load + B)
    v1_over_vs = z_in / (z_in + r_source)
    return z_in, v1_over_vs * v_load_over_v1


def metrics_from_zin_and_vratio(
    z_in: np.ndarray, v_ratio: np.ndarray, z_load: np.ndarray, r_source: float
) -> dict[str, np.ndarray]:
    """SWR, insertion gain, total transducer gain T = P_load/P_avail.

    P_avail = |V_s|² / (8·R_s)  (peak-amplitude convention)
    P_load  = (1/2)·|V_load|²·Re(1/Z_L*) = (1/2)·|V_load|²·R_L/|Z_L|²
    T = 4·R_s·R_L·|V_load/V_s|² / |Z_L|²

    Source-match factor (= 1 - |Γ_in|² with Γ_in measured against R_s):
    M_src = 4·R_s·Re(Z_in) / |Z_in + R_s|²

    Insertion gain = T / M_src (the proportion of input power that
    reaches the load — always ≤ 1, equals 1 for lossless N).
    """
    g_in = (z_in - r_source) / (z_in + r_source)
    swr = (1 + np.abs(g_in)) / (1 - np.abs(g_in))
    m_src = 4.0 * r_source * z_in.real / np.abs(z_in + r_source) ** 2
    r_l = z_load.real
    t_total = 4.0 * r_source * r_l * np.abs(v_ratio) ** 2 / np.abs(z_load) ** 2
    insertion_gain = np.where(m_src > 1e-12, t_total / m_src, 0.0)
    return {
        "swr": swr,
        "m_src": m_src,
        "insertion_gain": insertion_gain,
        "t_total": t_total,
    }


# ---------- optimizer ----------


def _objective(
    log_vals: np.ndarray,
    kinds: list[str],
    z_load: np.ndarray,
    omegas: np.ndarray,
    r_source: float,
    weights: np.ndarray,
    alpha: float,
    mode: str,
) -> float:
    """Max-min worst-case objective. `mode` selects what to maximize.

    mode == "t_total": classical transducer gain (lossless: same as
        lossy doing the optimal thing, since loss only hurts t_total).
    mode == "m_src":   pure source-match (what the SWR meter sees).
        With this mode the optimizer is allowed to dissipate power
        to flatten SWR. Compare to mode="t_total" to see the tradeoff.
    """
    values = np.exp(log_vals)
    M = cascade_abcd(kinds, values, omegas)
    z_in, v_ratio = z_in_and_voltage_ratio(M, z_load, r_source)
    metrics = metrics_from_zin_and_vratio(z_in, v_ratio, z_load, r_source)
    t = metrics[mode]
    t = np.clip(t, 1e-12, 1.0)
    return (1.0 / alpha) * np.log(np.sum(weights * np.exp(-alpha * t)))


def solve_ladder(
    kinds: list[str],
    z_load: np.ndarray,
    omegas: np.ndarray,
    r_source: float = 50.0,
    mode: str = "t_total",
    n_restarts: int = 30,
    max_iter: int = 500,
    alpha: float = 50.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, dict]:
    omega_center = float(np.exp(np.mean(np.log(omegas))))
    rng = np.random.default_rng(rng_seed)

    def init(kind):
        if kind.startswith("L"):
            return r_source / omega_center
        if kind.startswith("C"):
            return 1.0 / (r_source * omega_center)
        # R: choose a reasonable mid-decade resistor as starting point
        return r_source * 2.0

    init_vec = np.array([init(k) for k in kinds])
    weights = np.ones_like(omegas)

    starts = [np.log(init_vec)]
    for _ in range(n_restarts):
        starts.append(np.log(init_vec) + rng.normal(0.0, 0.7, size=len(kinds)))

    best_x, best_obj = None, np.inf
    for s in starts:
        try:
            res = minimize(
                _objective,
                s,
                args=(kinds, z_load, omegas, r_source, weights, alpha, mode),
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9},
            )
            if res.fun < best_obj:
                best_obj = float(res.fun)
                best_x = res.x
        except Exception:
            continue

    values = np.exp(best_x)
    M = cascade_abcd(kinds, values, omegas)
    z_in, v_ratio = z_in_and_voltage_ratio(M, z_load, r_source)
    metrics = metrics_from_zin_and_vratio(z_in, v_ratio, z_load, r_source)
    return values, metrics


# ---------- demo ----------


def fmt_swr(s: float) -> str:
    return f"{s:.2f}" if s < 50 else f"{s:.0f}"


def main() -> None:
    el20 = DipoleElement(leg_length_m=leg(14.3))
    el10 = DipoleElement(leg_length_m=leg(28.47))
    num, den = fan_dipole_impedance([el20, el10])
    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    z_load = evaluate_rational(num, den, 1j * omegas)

    print("2-element fan dipole at 5 design freqs:")
    print(f"  {'band':>4}  {'f (MHz)':>8}  {'Z_load':>22}  {'bare SWR':>8}")
    for lbl, f, z in zip(BAND_LABEL, F_DESIGN_MHZ, z_load):
        g = (z - Z0) / (z + Z0)
        swr = (1 + abs(g)) / (1 - abs(g))
        print(f"  {lbl:>4}  {f:>8.3f}  {z.real:>+9.2f}{z.imag:>+9.2f}j  {fmt_swr(swr):>8}")
    print()

    configs = [
        # label,                                 kinds,                                          mode
        (
            "(a) Lossless LC, optimize T (lossless v1 baseline)",
            ["L_shunt", "C_series", "L_shunt", "C_series", "L_shunt", "C_series"],
            "t_total",
        ),
        (
            "(b) LC + shunt R at input, optimize SWR",
            ["R_shunt", "L_shunt", "C_series", "L_shunt", "C_series", "L_shunt", "C_series"],
            "m_src",
        ),
        (
            "(c) LCR mixed, optimize SWR (best of both worlds?)",
            [
                "R_shunt",
                "L_shunt",
                "C_series",
                "R_series",
                "L_shunt",
                "C_series",
                "L_shunt",
                "C_series",
            ],
            "m_src",
        ),
        ("(d) Pure-resistive π pad — sanity check", ["R_shunt", "R_series", "R_shunt"], "m_src"),
    ]

    results = {}
    for label, kinds, mode in configs:
        values, metrics = solve_ladder(kinds, z_load, omegas, mode=mode)
        results[label] = (kinds, values, metrics)

        print("=" * 88)
        print(label)
        print("=" * 88)
        print(f"  topology (src→load): {' - '.join(kinds)}")
        print("  values: " + ", ".join(_fmt_value(k, v) for k, v in zip(kinds, values)))
        print(f"  {'band':>4}  {'SWR':>6}  {'ins.loss':>10}  {'T total':>10}")
        for lbl, swr, ig, t in zip(
            BAND_LABEL, metrics["swr"], metrics["insertion_gain"], metrics["t_total"]
        ):
            il_db = -10 * np.log10(max(ig, 1e-9))
            t_db = 10 * np.log10(max(t, 1e-9))
            print(f"  {lbl:>4}  {fmt_swr(swr):>6}  {il_db:>7.2f} dB  {t_db:>7.2f} dB")
        print(
            f"  worst SWR: {metrics['swr'].max():.2f}    "
            f"worst T: {10 * np.log10(max(metrics['t_total'].min(), 1e-9)):.2f} dB"
        )
        print()

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    x_pos = np.arange(5)
    width = 0.18
    for i, (label, (kinds, values, metrics)) in enumerate(results.items()):
        short = label.split(",")[0]  # e.g. "(a) Lossless LC"
        axes[0].bar(x_pos + (i - 1.5) * width, metrics["swr"], width=width, label=short)
        axes[1].bar(
            x_pos + (i - 1.5) * width,
            -10 * np.log10(np.maximum(metrics["t_total"], 1e-9)),
            width=width,
            label=short,
        )
    axes[0].axhline(1.0, color="black", lw=0.4)
    axes[0].axhline(2.0, color="grey", ls=":", lw=0.7)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(BAND_LABEL)
    axes[0].set_ylabel("SWR")
    axes[0].set_yscale("log")
    axes[0].set_title("SWR at the source — what the radio sees")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, axis="y", which="both", alpha=0.3)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(BAND_LABEL)
    axes[1].set_ylabel("Total loss (dB)")
    axes[1].set_title("Power lost (signal vs heat) — what you trade away")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("lossy_5band.png", dpi=140, bbox_inches="tight")
    print("wrote lossy_5band.png")


def _fmt_value(kind: str, val: float) -> str:
    if kind.startswith("L"):
        return f"{kind}={val * 1e6:.3g} µH"
    if kind.startswith("C"):
        return f"{kind}={val * 1e12:.3g} pF"
    return f"{kind}={val:.3g} Ω"


if __name__ == "__main__":
    main()
