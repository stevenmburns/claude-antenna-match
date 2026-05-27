#!/usr/bin/env python
"""Per-component stress breakdown for the R_rad=50 (inverted-V) lossy optimum.

Walks the matching network forward at each design frequency, tracking
voltage and current at every node. Reports for each component the
quantity that bounds its rating:

  - resistors: power dissipated (W)
  - capacitors: voltage across them (V peak)
  - inductors: current through them (A peak)

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


F_DESIGN_MHZ = np.array([14.300, 18.1575, 21.383, 24.970, 28.470])
BAND_LABEL = ["20m", "17m", "15m", "12m", "10m"]
Z0 = 50.0
P_AVAIL_W = 100.0
R_RAD = 50.0  # inverted-V

# ---- R_rad=50 (inverted-V) joint-length 2 dB optimum ----
F_LOW_MHZ = 16.320
F_HIGH_MHZ = 24.970

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
VALUES_SI = np.array(
    [
        437.0,  # R_shunt
        113.0e-12,  # C_shunt
        0.465e-6,  # L_series
        160.0e-12,  # C_shunt
        523.0,  # R_shunt
        0.865e-6,  # L_series
        79.6e-12,  # C_shunt
        0.438e-6,  # L_series
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
        ze = 1j * omega * val
        return np.array([[1, ze], [0, 1]], dtype=complex)
    if kind == "C_series":
        ze = 1.0 / (1j * omega * val)
        return np.array([[1, ze], [0, 1]], dtype=complex)
    if kind == "R_series":
        return np.array([[1, val], [0, 1]], dtype=complex)
    if kind == "L_shunt":
        ye = 1.0 / (1j * omega * val)
        return np.array([[1, 0], [ye, 1]], dtype=complex)
    if kind == "C_shunt":
        ye = 1j * omega * val
        return np.array([[1, 0], [ye, 1]], dtype=complex)
    if kind == "R_shunt":
        return np.array([[1, 0], [1.0 / val, 1]], dtype=complex)
    raise ValueError(kind)


def cascade_at(omega: float) -> np.ndarray:
    M = np.eye(2, dtype=complex)
    for k, v in zip(KINDS, VALUES_SI, strict=True):
        M = M @ section_abcd_at(k, v, omega)
    return M


def analyse(omega: float, p_avail_W: float = P_AVAIL_W) -> dict:
    """Returns per-section component stresses at this ω.

    For each section, records the relevant peak-amplitude quantity:
      - resistor (series): power, current through
      - resistor (shunt):  power, voltage across
      - inductor:          current through it (peak), |V| across (peak)
      - capacitor:         voltage across it (peak), |I| through (peak)
    """
    z_l = z_load_at(omega)
    M = cascade_at(omega)
    z_in = (M[0, 0] * z_l + M[0, 1]) / (M[1, 0] * z_l + M[1, 1])

    # |V_s|^2 / (8 R_s) = P_avail
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
            stress = {
                "kind": kind,
                "val": val,
                "I_through_peak": float(np.abs(i)),
                "V_across_peak": float(np.abs(ze * i)),
                "P_diss": 0.0,
            }
        elif kind == "C_series":
            ze = 1.0 / (1j * omega * val)
            v_new, i_new = v - ze * i, i
            stress = {
                "kind": kind,
                "val": val,
                "I_through_peak": float(np.abs(i)),
                "V_across_peak": float(np.abs(ze * i)),
                "P_diss": 0.0,
            }
        elif kind == "R_series":
            v_new, i_new = v - val * i, i
            stress = {
                "kind": kind,
                "val": val,
                "I_through_peak": float(np.abs(i)),
                "V_across_peak": float(np.abs(val * i)),
                "P_diss": 0.5 * np.abs(i) ** 2 * val,
            }
        elif kind == "L_shunt":
            ye = 1.0 / (1j * omega * val)
            v_new, i_new = v, i - ye * v
            stress = {
                "kind": kind,
                "val": val,
                "V_across_peak": float(np.abs(v)),
                "I_through_peak": float(np.abs(ye * v)),
                "P_diss": 0.0,
            }
        elif kind == "C_shunt":
            ye = 1j * omega * val
            v_new, i_new = v, i - ye * v
            stress = {
                "kind": kind,
                "val": val,
                "V_across_peak": float(np.abs(v)),
                "I_through_peak": float(np.abs(ye * v)),
                "P_diss": 0.0,
            }
        elif kind == "R_shunt":
            v_new, i_new = v, i - v / val
            stress = {
                "kind": kind,
                "val": val,
                "V_across_peak": float(np.abs(v)),
                "I_through_peak": float(np.abs(v / val)),
                "P_diss": 0.5 * np.abs(v) ** 2 / val,
            }
        else:
            raise ValueError(kind)
        per.append(stress)
        v, i = v_new, i_new

    v_load, i_load = v, i
    p_load = 0.5 * (v_load * np.conj(i_load)).real
    return {
        "z_in": z_in,
        "z_load": z_l,
        "p_into_net": float(p_into_net),
        "p_load": float(p_load),
        "per_section": per,
    }


def fmt_val(kind: str, val: float) -> str:
    if kind.startswith("L"):
        return f"{val * 1e6:.3g} µH"
    if kind.startswith("C"):
        return f"{val * 1e12:.3g} pF"
    return f"{int(val)} Ω"


def main() -> None:
    print(
        f"R_rad = {R_RAD:.0f} Ω (inverted-V), antenna f_low={F_LOW_MHZ} MHz, "
        f"f_high={F_HIGH_MHZ} MHz"
    )
    print(f"Topology: {' - '.join(f'{k}[{fmt_val(k, v)}]' for k, v in zip(KINDS, VALUES_SI))}")
    print(f"Available source power: {P_AVAIL_W} W")
    print()

    # Pre-compute all band data
    all_data = {}
    for lbl, f_mhz in zip(BAND_LABEL, F_DESIGN_MHZ):
        omega = 2 * np.pi * f_mhz * 1e6
        all_data[lbl] = analyse(omega)

    # ---- 1) Power balance ----
    print("=" * 78)
    print("POWER BALANCE per band (W)")
    print("=" * 78)
    r_positions = [i for i, k in enumerate(KINDS) if k.startswith("R")]
    header = f"{'band':>5}  {'P→net':>7}"
    for i in r_positions:
        header += f"  {'P[R' + str(i + 1) + ']':>9}"
    header += f"  {'P_load':>7}  {'P_ref*':>7}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = all_data[lbl]
        row = f"  {lbl:>3}  {a['p_into_net']:>7.2f}"
        for i in r_positions:
            row += f"  {a['per_section'][i]['P_diss']:>9.2f}"
        p_ref = P_AVAIL_W - a["p_into_net"]
        row += f"  {a['p_load']:>7.2f}  {p_ref:>7.2f}"
        print(row)
    print()
    print("  *P_ref = P_avail - P_into_net (the source-side reflection)")

    # ---- 2) Capacitor voltages (peak) ----
    print()
    print("=" * 78)
    print("CAPACITOR PEAK VOLTAGE per band (V peak)")
    print("=" * 78)
    cap_idx = [i for i, k in enumerate(KINDS) if k.startswith("C")]
    header = f"{'band':>5}"
    for i in cap_idx:
        header += f"  {KINDS[i] + '[' + str(i + 1) + ']=' + fmt_val(KINDS[i], VALUES_SI[i]):>16}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = all_data[lbl]
        row = f"  {lbl:>3}"
        for i in cap_idx:
            row += f"  {a['per_section'][i]['V_across_peak']:>16.2f}"
        print(row)
    print()
    cap_max = {
        i: max(all_data[lbl]["per_section"][i]["V_across_peak"] for lbl in BAND_LABEL)
        for i in cap_idx
    }
    print("  WORST-CASE per cap:")
    for i in cap_idx:
        print(
            f"    {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], VALUES_SI[i])}):  "
            f"max V = {cap_max[i]:.1f} V peak  →  rate ≥{int(cap_max[i] * 1.5)} V WV"
        )

    # ---- 3) Inductor currents (peak) ----
    print()
    print("=" * 78)
    print("INDUCTOR PEAK CURRENT per band (A peak)")
    print("=" * 78)
    ind_idx = [i for i, k in enumerate(KINDS) if k.startswith("L")]
    header = f"{'band':>5}"
    for i in ind_idx:
        header += f"  {KINDS[i] + '[' + str(i + 1) + ']=' + fmt_val(KINDS[i], VALUES_SI[i]):>16}"
    print(header)
    print("-" * len(header))
    for lbl in BAND_LABEL:
        a = all_data[lbl]
        row = f"  {lbl:>3}"
        for i in ind_idx:
            row += f"  {a['per_section'][i]['I_through_peak']:>16.3f}"
        print(row)
    print()
    ind_max = {
        i: max(all_data[lbl]["per_section"][i]["I_through_peak"] for lbl in BAND_LABEL)
        for i in ind_idx
    }
    print("  WORST-CASE per inductor:")
    for i in ind_idx:
        i_max = ind_max[i]
        # RMS = peak/sqrt(2), continuous; resistor I²R loss in finite-Q L is one
        # concern, core saturation is another
        print(
            f"    {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], VALUES_SI[i])}):  "
            f"max I = {i_max:.2f} A peak ({i_max / np.sqrt(2):.2f} A RMS)  "
            f"→ rate ≥{int(i_max * 1.3)} A peak / "
            f"saturation flux density check needed"
        )

    # ---- 4) Resistor sizing reminder ----
    print()
    print("=" * 78)
    print("RESISTOR SIZING (worst-case per band, for component selection)")
    print("=" * 78)
    for i in r_positions:
        p_max = max(all_data[lbl]["per_section"][i]["P_diss"] for lbl in BAND_LABEL)
        v_max = max(all_data[lbl]["per_section"][i]["V_across_peak"] for lbl in BAND_LABEL)
        print(
            f"  {KINDS[i]} pos {i + 1} ({fmt_val(KINDS[i], VALUES_SI[i])}):  "
            f"max P = {p_max:.2f} W   max V = {v_max:.0f} V peak   "
            f"→ pick ≥{int(p_max * 2)} W non-inductive"
        )


if __name__ == "__main__":
    main()
