# 2026-05-26 — Lossy matching network analysis for the 5-band fan dipole

## TL;DR

- Extended v1 LC-ladder to include **resistive sections** (R_shunt) via ABCD-matrix
  cascade, with proper accounting of dissipation (P_load < P_into_network for lossy
  networks). Loss-bounded constrained optimization with quadratic penalty + multistart
  L-BFGS-B.
- **Two-element fan dipole with joint length optimization + lossy matcher** delivers
  the Pareto curve: **0 dB loss → SWR 3.2; 2 dB loss → SWR 2.6; 5 dB loss → SWR 1.9**
  (all worst-case across the 5 amateur HF bands).
- **Complexity sweep**: N=4 components already gets within 0.13 SWR units of N=8.
  N=5 (R_shunt + L_shunt + C_series + L_shunt + C_series) is the engineering sweet
  spot.
- **Caught and fixed an important bug** in `metrics_summary` — the worst-loss
  calculation was inverted (`(-10·log).max()` was parsed as `-10·(log.max())`,
  reporting the *smallest* loss as worst). All prior loss-bounded results from
  before this writeup were silently wrong; results in this doc use the fix.
- **Off-the-shelf 500 Ω fixed resistors cost only 0.02 SWR** vs the free-R optimum
  (437 / 523 Ω), confirming the R values aren't sensitive parameters.

## Context

Yesterday's work (2026-05-25) ended with v3 Darlington synthesis postponed and a
deep dive into the user's actual physical 5-band fan dipole. Today shifted to a
different practical question: *for the fixed 2-element fan dipole problem, can
lossy matching networks beat the Bode-Fano SWR floor — and at what cost in
delivered power?*

## What was added

### `src/antmatch/matching.py` extensions (NOT committed yet)
The lossy analysis is currently in standalone scripts (`scripts/lossy_*.py`)
rather than promoted into the package. If we want to merge the lossy framework
into the main `matching.py`, the diffs are:
- Add `R_series` and `R_shunt` to `SECTION_KINDS`
- Add `cascade_abcd` returning the full 2×2 ABCD matrix
- Add `evaluate_lossy(...)` returning SWR, insertion gain, and total T separately
- Add constrained-optimization solver via penalty method

### Scripts (all in `scripts/`)
- `lossy_5band_pareto.py` — first attempt; has the metric-bug, kept for history
- `lossy_5band_jointlen.py` — adds joint antenna length optimization (also bug)
- `lossy_smith_2db.py` — Smith chart at 2 dB budget (R_rad=65, free R) — bug fix applied here
- `lossy_complexity_sweep.py` — N ∈ {3,4,5,6,8} comparison at 2 dB budget
- `lossy_fixed_R500.py` — tests off-the-shelf 500 Ω resistor penalty
- `n5_loss_sweep.py` — Pareto curve for N=5 across loss budgets 0.3-5 dB
- `n5_2db_breakdown.py` — per-component stress at the N=5 / 2 dB optimum
- `bare_antenna_swr.py` — SWR vs frequency of the bare antenna at three different
  resonance configurations
- `resistor_power_breakdown.py`, `component_stress_breakdown.py` — earlier breakdowns

Plots (in repo root, `.gitignore`d): `lossy_pareto*.png`, `lossy_jointlen.png`,
`lossy_smith_2db*.png`, `lossy_complexity_sweep.png`, `n5_loss_sweep.png`,
`bare_antenna_swr.png`.

## The bug

`metrics_summary` originally was:
```python
loss_db = float(-10 * np.log10(np.clip(insertion_gain, 1e-9, 1.0)).max())
```
Python's parsing makes this `-10 * (np.log10(...).max())`, which takes the MAX of
the log values (closest to zero, since log(gain<1) is negative), then negates.
The result is the **smallest** loss across bands — the opposite of what was
intended. Fix:
```python
loss_db = float((-10 * np.log10(np.clip(insertion_gain, 1e-9, 1.0))).max())
```
The bug masked the real loss: at the N=8 2-dB optimum, per-band losses were
1.4-2.0 dB by the new metric, but 5-8 dB on most bands when computed correctly.
**All results in this status doc post-date the fix.**

## Headline results — N=5 matcher Pareto curve

Joint antenna-length optimization, inverted-V antenna (R_rad = 50 Ω), topology
R_shunt + L_shunt + C_series + L_shunt + C_series:

| loss budget | worst SWR | f_low MHz | f_high MHz |
|---|---|---|---|
| 0.3 dB | 3.17 | 17.04 | 25.47 |
| 1.0 dB | 2.89 | 16.86 | 25.77 |
| **2.0 dB** | **2.59** | 17.08 | 25.87 |
| 3.0 dB | 2.38 | 17.34 | 25.87 |
| 5.0 dB | 1.93 | 17.88 | 26.01 |

Curve is **almost exactly linear in dB**: ≈ 0.25 SWR units per dB of loss.

### N=5 2-dB component values
- R_shunt = **340 Ω**
- L_shunt = 1.12 µH
- C_series = 76.4 pF
- L_shunt = 0.896 µH
- C_series = 256 pF

### Component stress at 2 dB / 100 W
- R (340 Ω): max **30 W** dissipation, 143 V peak — needs ≥60 W non-inductive
- C₃ (76.4 pF series): max **421 V peak on 20m** — needs ≥600 V WV (silver mica or
  vacuum variable). This is the most stressed component.
- C₅ (256 pF series): 93 V peak max — 150 V WV mica is fine
- L₂ (1.12 µH shunt): 1.1 A peak max — modest
- L₄ (0.896 µH shunt): **5.0 A peak on 20m** — needs #16 AWG or larger; Q ≥100 to
  avoid eating into the loss budget

## Complexity vs SWR (at 2 dB budget)

| N | topology | worst SWR | Δ vs N=8 |
|---|---|---|---|
| 3 | R+L+C | 5.50 | +3.00 |
| **4** | R+L+C+L | **2.63** | +0.13 |
| **5** | R+L+C+L+C | **2.59** | +0.09 |
| 6 | R+C+L+C+L+R | 2.51 | +0.01 |
| 8 | R+C+L+C+R+L+C+L | 2.51 | 0 |

**Knee is at N=4**, with N=5 a small refinement. Past N=6 there's no benefit. N=3
hits a wall (SWR 5.50) because there isn't enough reactive freedom to flatten 5
bands.

**Trading parts for stress:** N=5 puts more strain on each component than N=8.
Single resistor takes 30 W (vs 23 W spread across two). C peak voltage is 421 V
(vs 332 V at N=8). Inductor peak current is 5 A (vs 3.2 A at N=8). The Pareto
isn't free — fewer parts means each part works harder.

## Bare-antenna SWR (no matcher) at the 3 configurations

| antenna f_low / f_high | 20m | 17m | 15m | 12m | 10m | **worst** |
|---|---|---|---|---|---|---|
| 14.30 / 28.47 (original, what you built) | 1.07 | 19.2 | 23.2 | 8.0 | 1.07 | 23.2 |
| 18.16 / 24.97 (17m + 12m corners) | 22.3 | 1.16 | 5.6 | 1.16 | 8.7 | 22.3 |
| **17.0 / 25.8 (N=5 optimum)** | 13.5 | 2.96 | 9.1 | 1.67 | 5.9 | **13.5** |

Moving the antenna resonances inward to ~17 / ~26 MHz drops the worst-case bare
SWR from 23 to 13.5 — the matcher then compresses that 13.5:1 down to 2.6:1
with 2 dB of dissipation.

## What "100 W input" means in practice (N=5 / 2 dB / 17.08 / 25.87 MHz)

Per band, with 100 W available from a 50 Ω source:

| band | P→net | P_resistor | P_load | P_reflected |
|---|---|---|---|---|
| 20m | 80.4 W | 4.8 W | 75.6 W | 19.6 W |
| 17m | 80.3 W | **29.8 W** | 50.6 W | 19.7 W |
| 15m | 80.3 W | 11.0 W | 69.3 W | 19.7 W |
| 12m | 81.2 W | **30.1 W** | 51.1 W | 18.8 W |
| 10m | 80.3 W | 11.0 W | 69.3 W | 19.7 W |

Worst-band power-to-load is **51 W (50.6% efficiency on 17m and 12m)**, best is
75.6 W (76% on 20m). About 5-30 W heats the resistor, about 18-20 W reflects.

## Open questions / next directions

1. **Finite-Q L and C** — current optimizer assumes lossless reactives. Real
   inductors at HF have Q ≈ 100-200; capacitors are typically very high Q. Adding
   Q=100 inductor loss would add maybe 0.5 dB to the worst-band loss, eating into
   the budget. Worth re-running the Pareto with realistic Q.
2. **4-band variant** — if the user can drop 20m, does the 4-band Pareto curve
   look better? Smaller bandwidth ratio (1.57:1 vs 1.99:1) should help.
3. **Higher loss budgets for SWR ≤ 1.5** — extrapolating the linear-in-dB Pareto,
   getting SWR ≤ 1.5 might need 8-10 dB loss, which is impractical. Worth
   verifying that the extrapolation holds and confirming there's no hidden corner.
4. **Mutual coupling**: today's analysis ignores it entirely. With the realistic
   pentagon-spacer 5-element antenna (yesterday's work), the lossy matcher could
   be applied to the coupled `Z_a(jω)` to see if its 2 dB loss budget can recover
   the 15m hybridization band. Requires rational-fitting the numerical Z_L.
5. **Merge into matching.py** — the lossy framework currently lives in scripts.
   Should be promoted to package code with `SECTION_KINDS` extension and a
   `solve_lossy_ladder` function in `matching.py`.

## Engineering recommendation

For the 2-element fan dipole problem with 5-band coverage:

- **Antenna**: inverted-V geometry (R_rad ≈ 50 Ω), legs trimmed for resonances at
  17.0 MHz and 25.8 MHz (legs ≈ 4.16 m and 2.75 m).
- **Matcher**: R_shunt + L_shunt + C_series + L_shunt + C_series. Pick a loss
  budget based on station capabilities:
  - 1.5 dB → SWR 2.7 (modest heat, modest match)
  - 2.0 dB → SWR 2.6 (sweet spot)
  - 3.0 dB → SWR 2.4 (good match, more heat)
- **Components** (at 2 dB / 100 W): 500 Ω 60 W non-inductive R; 800 V silver
  mica for series C₃; 150 V mica for series C₅; air-core ≥ #16 AWG inductors.
  Total parts count: 5.

This is dramatically simpler than a typical multi-band auto-tuner and works
across all 5 amateur HF bands simultaneously without retuning.

## Artifacts list

Currently uncommitted on branch `fan-dipole-coupling-study`:
- `src/antmatch/matching.py` — unchanged today (lossy logic still in scripts)
- `scripts/lossy_5band_pareto.py`, `scripts/lossy_5band_jointlen.py`,
  `scripts/lossy_smith_2db.py`, `scripts/lossy_complexity_sweep.py`,
  `scripts/lossy_fixed_R500.py`, `scripts/n5_loss_sweep.py`,
  `scripts/n5_2db_breakdown.py`, `scripts/resistor_power_breakdown.py`,
  `scripts/component_stress_breakdown.py`, `scripts/bare_antenna_swr.py`
- `docs/2026-05-26-lossy-matcher-pareto.md` — this writeup
- Log files in `logs/` (kept for re-analysis)
