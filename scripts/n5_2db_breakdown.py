#!/usr/bin/env python
"""Per-component stress breakdown for the N=5 matcher at 2 dB loss budget.

Reads the optimum component values from logs/n5_loss_sweep.log (the
2.0 dB budget row), then walks the cascade at each design frequency
to report:
  - resistor: power dissipated (W)
  - capacitor: voltage across it (V peak)
  - inductor: current through it (A peak)

Assumes 100 W AVAILABLE from a 50 Ω Thévenin source.
"""

from __future__ import annotations

import re
import sys

import numpy as np

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
P_AVAIL_W = 100.0
R_RAD = 50.0
KINDS = ["R_shunt", "L_shunt", "C_series", "L_shunt", "C_series"]
LOG_PATH = "logs/n5_loss_sweep.log"


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def parse_2db_result(log_path: str):
    """Find the B2.0 result line and the values listed beneath."""
    text = open(log_path).read()
    # Look for: "B2.0 [budget=2.00] DONE ... SWR=... loss=... f=(LOW, HIGH)"
    m = re.search(
        r"B2\.0 \[budget=2\.0\d*\] DONE in [\d.]+s\s+SWR=([\d.]+)\s+loss=([\d.]+) dB\s+f=\(([\d.]+),\s*([\d.]+)\)",
        text,
    )
    if not m:
        raise RuntimeError(
            "could not find B2.0 result line in log; has the sweep reached 2.0 dB yet?"
        )
    swr, loss, f_low, f_high = (
        float(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
        float(m.group(4)),
    )
    # Note: the sweep log doesn't print matcher values directly. We re-solve.
    return swr, loss, f_low, f_high


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


def z_load_at(omega: float, f_low: float, f_high: float) -> complex:
    el_lo = DipoleElement(leg_length_m=leg(f_low), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(f_high), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return complex(evaluate_rational(num, den, np.array([1j * omega]))[0])


def analyse(
    values: np.ndarray, f_low: float, f_high: float, omega: float, p_avail_W: float = P_AVAIL_W
) -> dict:
    z_l = z_load_at(omega, f_low, f_high)
    M = cascade_at(values, omega)
    z_in = (M[0, 0] * z_l + M[0, 1]) / (M[1, 0] * z_l + M[1, 1])

    v_s_mag = np.sqrt(8.0 * Z0 * p_avail_W)
    v_s = complex(v_s_mag)
    i_in = v_s / (Z0 + z_in)
    v_in = v_s - Z0 * i_in
    p_into_net = 0.5 * (v_in * np.conj(i_in)).real

    v, i = v_in, i_in
    per = []
    for kind, val in zip(KINDS, values, strict=True):
        if kind == "L_series":
            ze = 1j * omega * val
            v_new, i_new = v - ze * i, i
            per.append(
                {
                    "kind": kind,
                    "val": val,
                    "I_peak": float(np.abs(i)),
                    "V_peak": float(np.abs(ze * i)),
                    "P": 0.0,
                }
            )
        elif kind == "C_series":
            ze = 1.0 / (1j * omega * val)
            v_new, i_new = v - ze * i, i
            per.append(
                {
                    "kind": kind,
                    "val": val,
                    "I_peak": float(np.abs(i)),
                    "V_peak": float(np.abs(ze * i)),
                    "P": 0.0,
                }
            )
        elif kind == "L_shunt":
            ye = 1.0 / (1j * omega * val)
            v_new, i_new = v, i - ye * v
            per.append(
                {
                    "kind": kind,
                    "val": val,
                    "V_peak": float(np.abs(v)),
                    "I_peak": float(np.abs(ye * v)),
                    "P": 0.0,
                }
            )
        elif kind == "C_shunt":
            ye = 1j * omega * val
            v_new, i_new = v, i - ye * v
            per.append(
                {
                    "kind": kind,
                    "val": val,
                    "V_peak": float(np.abs(v)),
                    "I_peak": float(np.abs(ye * v)),
                    "P": 0.0,
                }
            )
        elif kind == "R_shunt":
            v_new, i_new = v, i - v / val
            per.append(
                {
                    "kind": kind,
                    "val": val,
                    "V_peak": float(np.abs(v)),
                    "I_peak": float(np.abs(v / val)),
                    "P": 0.5 * np.abs(v) ** 2 / val,
                }
            )
        v, i = v_new, i_new
    p_load = 0.5 * (v * np.conj(i)).real
    return {
        "p_into_net": float(p_into_net),
        "p_load": float(p_load),
        "per_section": per,
        "z_in": z_in,
        "z_load": z_l,
    }


def fmt_val(kind: str, val: float) -> str:
    if kind.startswith("L"):
        return f"{val * 1e6:.3g} µH"
    if kind.startswith("C"):
        return f"{val * 1e12:.3g} pF"
    return f"{int(val)} Ω"


def main():
    # Solve for the 2 dB optimum (we have to re-solve because the log
    # doesn't store matcher values per budget).
    print("Re-solving N=5 at 2 dB loss budget for component values...")
    sys.path.insert(0, "scripts")
    from n5_loss_sweep import solve

    omegas = 2 * np.pi * F_DESIGN_MHZ * 1e6
    matcher_vals, f_low, f_high, m_design = solve(2.0, omegas, label="(rebuild)")

    swr_max = float(m_design["swr"].max())
    loss_max = float((-10 * np.log10(np.clip(m_design["insertion_gain"], 1e-9, 1.0))).max())
    print()
    print("=== N=5 matcher at 2 dB worst loss ===")
    print(f"Antenna: f_low={f_low:.3f} MHz, f_high={f_high:.3f} MHz")
    print(f"Worst SWR={swr_max:.3f}, worst loss={loss_max:.3f} dB")
    print(f"Topology: {' - '.join(KINDS)}")
    print(f"Values:   {', '.join(fmt_val(k, v) for k, v in zip(KINDS, matcher_vals))}")
    print()

    # Analyse at each design freq
    band_data = {}
    for lbl, f_mhz in zip(BAND_LABEL, F_DESIGN_MHZ):
        omega = 2 * np.pi * f_mhz * 1e6
        band_data[lbl] = analyse(matcher_vals, f_low, f_high, omega)

    # Power balance
    print("POWER per band (W) at 100 W available source power")
    print("-" * 70)
    r_idx = [i for i, k in enumerate(KINDS) if k.startswith("R")]
    header = f"{'band':>5}  {'P→net':>7}"
    for i in r_idx:
        header += f"  {'P[R' + str(i + 1) + '=' + str(int(matcher_vals[i])) + 'Ω]':>16}"
    header += f"  {'P_load':>7}  {'P_ref':>7}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = band_data[lbl]
        row = f"  {lbl:>3}  {a['p_into_net']:>7.2f}"
        for i in r_idx:
            row += f"  {a['per_section'][i]['P']:>16.2f}"
        p_ref = P_AVAIL_W - a["p_into_net"]
        row += f"  {a['p_load']:>7.2f}  {p_ref:>7.2f}"
        print(row)

    print()
    print("CAPACITOR voltage (V peak)")
    print("-" * 70)
    cap_idx = [i for i, k in enumerate(KINDS) if k.startswith("C")]
    header = f"{'band':>5}"
    for i in cap_idx:
        header += f"  {KINDS[i] + '[' + str(i + 1) + ']=' + fmt_val(KINDS[i], matcher_vals[i]):>20}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = band_data[lbl]
        row = f"  {lbl:>3}"
        for i in cap_idx:
            row += f"  {a['per_section'][i]['V_peak']:>20.2f}"
        print(row)
    print()
    print("  WORST-CASE voltages (for cap rating):")
    for i in cap_idx:
        v_max = max(band_data[lbl]["per_section"][i]["V_peak"] for lbl in BAND_LABEL)
        print(
            f"    {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], matcher_vals[i])}):  "
            f"max V = {v_max:.1f} V peak  →  ≥{int(v_max * 1.5)} V WV"
        )

    print()
    print("INDUCTOR current (A peak)")
    print("-" * 70)
    ind_idx = [i for i, k in enumerate(KINDS) if k.startswith("L")]
    header = f"{'band':>5}"
    for i in ind_idx:
        header += f"  {KINDS[i] + '[' + str(i + 1) + ']=' + fmt_val(KINDS[i], matcher_vals[i]):>20}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = band_data[lbl]
        row = f"  {lbl:>3}"
        for i in ind_idx:
            row += f"  {a['per_section'][i]['I_peak']:>20.3f}"
        print(row)
    print()
    print("  WORST-CASE currents (for inductor sizing):")
    for i in ind_idx:
        i_max = max(band_data[lbl]["per_section"][i]["I_peak"] for lbl in BAND_LABEL)
        print(
            f"    {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], matcher_vals[i])}):  "
            f"max I = {i_max:.2f} A peak ({i_max / np.sqrt(2):.2f} A RMS)"
        )

    print()
    print("RESISTOR power/voltage")
    print("-" * 70)
    for i in r_idx:
        p_max = max(band_data[lbl]["per_section"][i]["P"] for lbl in BAND_LABEL)
        v_max = max(band_data[lbl]["per_section"][i]["V_peak"] for lbl in BAND_LABEL)
        print(
            f"  R_shunt pos {i + 1} ({fmt_val(KINDS[i], matcher_vals[i])}):  "
            f"max P = {p_max:.2f} W,  max V = {v_max:.0f} V peak  "
            f"→ rate ≥{int(p_max * 2)} W non-inductive"
        )


if __name__ == "__main__":
    main()
