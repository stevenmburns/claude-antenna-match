#!/usr/bin/env python
"""Per-component stress breakdown for the 4-band N=5 matcher at 2 dB loss.

Uses values directly from the 4-band loss-sweep log (budget=2.0 dB row):
  f_low=19.144 MHz, f_high=26.933 MHz
  R=153 Ω, L=0.673 µH, C=82.7 pF, L=0.55 µH, C=263 pF
  worst SWR = 1.90

Reports per band:
  - power dissipated in the resistor (W)
  - voltage across each capacitor (V peak)
  - current through each inductor (A peak)

Assumes 100 W AVAILABLE from a 50 Ω source.
"""

from __future__ import annotations

import numpy as np

from antmatch.antenna import (
    C_LIGHT,
    VF_BARE_WIRE,
    DipoleElement,
    evaluate_rational,
    fan_dipole_impedance,
)


# 4 BANDS — no 20m
F_DESIGN_MHZ = np.array([18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["17m", "15m", "12m", "10m"]
Z0 = 50.0
P_AVAIL_W = 100.0
R_RAD = 50.0

# ---- 4-band 2 dB optimum (from logs/n5_loss_sweep_4band.log) ----
F_LOW_MHZ = 19.144
F_HIGH_MHZ = 26.933

KINDS = ["R_shunt", "L_shunt", "C_series", "L_shunt", "C_series"]
VALUES_SI = np.array(
    [
        153.0,  # R_shunt (Ω)
        0.673e-6,  # L_shunt (H)
        82.7e-12,  # C_series (F)
        0.55e-6,  # L_shunt (H)
        263.0e-12,  # C_series (F)
    ]
)


def leg(f_mhz: float) -> float:
    return VF_BARE_WIRE * C_LIGHT / (4.0 * f_mhz * 1e6)


def z_load_at(omega: float) -> complex:
    el_lo = DipoleElement(leg_length_m=leg(F_LOW_MHZ), r_rad=R_RAD)
    el_hi = DipoleElement(leg_length_m=leg(F_HIGH_MHZ), r_rad=R_RAD)
    num, den = fan_dipole_impedance([el_lo, el_hi])
    return complex(evaluate_rational(num, den, np.array([1j * omega]))[0])


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


def cascade_at(omega: float) -> np.ndarray:
    M = np.eye(2, dtype=complex)
    for k, v in zip(KINDS, VALUES_SI, strict=True):
        M = M @ section_abcd_at(k, v, omega)
    return M


def analyse(omega: float, p_avail_W: float = P_AVAIL_W) -> dict:
    z_l = z_load_at(omega)
    M = cascade_at(omega)
    z_in = (M[0, 0] * z_l + M[0, 1]) / (M[1, 0] * z_l + M[1, 1])

    v_s_mag = np.sqrt(8.0 * Z0 * p_avail_W)
    v_s = complex(v_s_mag)
    i_in = v_s / (Z0 + z_in)
    v_in = v_s - Z0 * i_in
    p_into_net = 0.5 * (v_in * np.conj(i_in)).real

    v, i = v_in, i_in
    per = []
    for kind, val in zip(KINDS, VALUES_SI, strict=True):
        if kind == "L_series":
            ze = 1j * omega * val
            v_new, i_new = v - ze * i, i
            per.append({"I_peak": float(np.abs(i)), "V_peak": float(np.abs(ze * i)), "P": 0.0})
        elif kind == "C_series":
            ze = 1.0 / (1j * omega * val)
            v_new, i_new = v - ze * i, i
            per.append({"I_peak": float(np.abs(i)), "V_peak": float(np.abs(ze * i)), "P": 0.0})
        elif kind == "L_shunt":
            ye = 1.0 / (1j * omega * val)
            v_new, i_new = v, i - ye * v
            per.append({"V_peak": float(np.abs(v)), "I_peak": float(np.abs(ye * v)), "P": 0.0})
        elif kind == "C_shunt":
            ye = 1j * omega * val
            v_new, i_new = v, i - ye * v
            per.append({"V_peak": float(np.abs(v)), "I_peak": float(np.abs(ye * v)), "P": 0.0})
        elif kind == "R_shunt":
            v_new, i_new = v, i - v / val
            per.append(
                {
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


def main() -> None:
    print("=== 4-BAND N=5 matcher at 2 dB worst loss ===")
    print(f"Antenna: f_low={F_LOW_MHZ:.3f} MHz, f_high={F_HIGH_MHZ:.3f} MHz")
    print(f"         legs = {leg(F_LOW_MHZ):.3f} m and {leg(F_HIGH_MHZ):.3f} m")
    print(f"Topology: {' - '.join(KINDS)}")
    print(f"Values:   {', '.join(fmt_val(k, v) for k, v in zip(KINDS, VALUES_SI))}")
    print(f"Design freqs: {F_DESIGN_MHZ.tolist()} MHz (no 20m)")
    print()

    band_data = {}
    for lbl, f_mhz in zip(BAND_LABEL, F_DESIGN_MHZ):
        omega = 2 * np.pi * f_mhz * 1e6
        band_data[lbl] = analyse(omega)

    # Power balance
    print("POWER per band (W) at 100 W available source power")
    print("-" * 60)
    r_idx = [i for i, k in enumerate(KINDS) if k.startswith("R")]
    header = f"{'band':>5}  {'P→net':>7}"
    for i in r_idx:
        header += f"  {'P[R=' + str(int(VALUES_SI[i])) + 'Ω]':>13}"
    header += f"  {'P_load':>7}  {'P_ref':>7}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = band_data[lbl]
        row = f"  {lbl:>3}  {a['p_into_net']:>7.2f}"
        for i in r_idx:
            row += f"  {a['per_section'][i]['P']:>13.2f}"
        p_ref = P_AVAIL_W - a["p_into_net"]
        row += f"  {a['p_load']:>7.2f}  {p_ref:>7.2f}"
        print(row)

    print()
    print("CAPACITOR voltage (V peak)")
    print("-" * 60)
    cap_idx = [i for i, k in enumerate(KINDS) if k.startswith("C")]
    header = f"{'band':>5}"
    for i in cap_idx:
        header += f"  {KINDS[i] + '[' + str(i + 1) + ']=' + fmt_val(KINDS[i], VALUES_SI[i]):>18}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = band_data[lbl]
        row = f"  {lbl:>3}"
        for i in cap_idx:
            row += f"  {a['per_section'][i]['V_peak']:>18.2f}"
        print(row)
    print()
    print("  WORST-CASE voltages:")
    for i in cap_idx:
        v_max = max(band_data[lbl]["per_section"][i]["V_peak"] for lbl in BAND_LABEL)
        print(
            f"    {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], VALUES_SI[i])}):  "
            f"max V = {v_max:.1f} V peak  →  ≥{int(v_max * 1.5)} V WV"
        )

    print()
    print("INDUCTOR current (A peak)")
    print("-" * 60)
    ind_idx = [i for i, k in enumerate(KINDS) if k.startswith("L")]
    header = f"{'band':>5}"
    for i in ind_idx:
        header += f"  {KINDS[i] + '[' + str(i + 1) + ']=' + fmt_val(KINDS[i], VALUES_SI[i]):>18}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = band_data[lbl]
        row = f"  {lbl:>3}"
        for i in ind_idx:
            row += f"  {a['per_section'][i]['I_peak']:>18.3f}"
        print(row)
    print()
    print("  WORST-CASE currents:")
    for i in ind_idx:
        i_max = max(band_data[lbl]["per_section"][i]["I_peak"] for lbl in BAND_LABEL)
        print(
            f"    {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], VALUES_SI[i])}):  "
            f"max I = {i_max:.2f} A peak ({i_max / np.sqrt(2):.2f} A RMS)"
        )

    print()
    print("RESISTOR")
    print("-" * 60)
    for i in r_idx:
        p_max = max(band_data[lbl]["per_section"][i]["P"] for lbl in BAND_LABEL)
        v_max = max(band_data[lbl]["per_section"][i]["V_peak"] for lbl in BAND_LABEL)
        print(
            f"  {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], VALUES_SI[i])}):  "
            f"max P = {p_max:.2f} W,  max V = {v_max:.0f} V peak  "
            f"→ rate ≥{int(p_max * 2)} W non-inductive"
        )


if __name__ == "__main__":
    main()
