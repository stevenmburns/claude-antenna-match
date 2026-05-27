# 2026-05-26 (afternoon) — L-tuner discovery for 4-band fan dipole

Companion to `2026-05-26-lossy-matcher-pareto.md` (morning work). The
morning landed on an 8-element lossy matcher with 2 dB worst-band loss
giving SWR 2.5 across 5 bands. This afternoon's work answers two
follow-up questions and finds a **dramatically simpler design**.

## TL;DR

- **Dropping the 20m band** transforms the problem. The same N=5 matcher
  achieves SWR ≤ 2 across 4 bands (17m/15m/12m/10m) with only 1.5 dB loss
  (vs 5 dB for 5-band). Component stresses drop ~2×.
- **Making one component per-band-tunable** is dramatic: with **L₃ (middle
  shunt inductor) switched per band**, the matcher needs **no resistor at
  all** — SWR 1.01 across 4 bands, **lossless**.
- **Topology reduction**: dropping the resistor and trying smaller LC-only
  topologies with one per-band variable shows that an **N=2 L-tuner**
  (1 fixed shunt L + 1 per-band-switched series C) achieves SWR 1.15 across
  4 bands, lossless.
- **Two-cap version**: pairing the bands cleverly (17m+12m on one cap,
  15m+10m on another) lets the L-tuner work with **only 2 fixed cap values**
  and a 2-position band switch — worst SWR 1.32. This is the simplest
  practical 4-band matchbox we found.
- **The pairing matters enormously.** Alternating (17m+12m | 15m+10m) →
  1.32; low/high (17m+15m | 12m+10m) → 2.64; outer/inner → 2.40. The
  alternating pairing exploits the L-tuner's "harmonic" matching points.

## Background

After the morning's 5-band 2 dB lossy result (SWR 2.5, lots of heat), the
user asked: *can we do better by dropping 20m?* The answer turned into a
cascade of simplifications that ended with a minimum-viable 2-cap L-tuner.

## Step 1 — 4-band loss-budget sweep (N=5 matcher)

Same R+L+C+L+C topology, drop 20m, joint length opt. Pareto frontier:

| budget | 4-band worst SWR | 5-band worst SWR (prior) |
|---|---|---|
| 0.3 dB | **2.21** | 3.17 |
| 1.0 dB | **2.08** | 2.89 |
| **1.5 dB** | **1.99** | 2.73 |
| 2.0 dB | **1.90** | 2.59 |
| 3.0 dB | **1.72** | 2.38 |
| 5.0 dB | **1.46** | 1.93 |

4-band needs only **1.5 dB to hit SWR ≤ 2**, vs 5 dB for the 5-band case
— a **3.5 dB power-budget improvement** just from dropping 20m. Optimizer
puts the antenna resonances at ~19-21 MHz and ~26-27 MHz (interior of the
4-band range, near the geometric center).

At 2 dB, every component is about half as stressed as in the 5-band case:
peak cap voltage 233 V (vs 421), peak inductor current 2.6 A (vs 5.0).

## Step 2 — what if ONE component is per-band tunable?

Keeping the N=5 R+L+C+L+C topology at the 2 dB loss budget, try each of
the 5 positions as per-band-tunable (4 distinct values for each band).
The other 4 components stay shared across bands. Antenna lengths jointly
optimized.

| variable position | worst SWR | worst loss | comment |
|---|---|---|---|
| 0 (R₀ shunt) | 1.87 | 2.00 dB | barely better than fixed-R baseline |
| 1 (L₁ shunt) | 1.012 | 1.74 dB | near-perfect |
| 2 (C₂ series) | 1.007 | 1.53 dB | tied lowest SWR |
| **3 (L₃ shunt)** | **1.008** | **0.00 dB** ← | **lossless! optimizer turned R off** |
| 4 (C₄ series) | 1.054 | 2.00 dB | uses full budget |

**Position 3 (middle shunt inductor variable per band) → lossless perfect
match.** The optimizer set the shared R to 990 kΩ (effectively open). The
remaining matcher is purely reactive: per-band L₃ tuning provides enough
freedom that no dissipation is needed.

L₃ values per band: 0.45 µH (17m), 0.49 µH (15m), 0.24 µH (12m), 2.41 µH
(10m). The 10m value is anomalously large — likely a different operating
mode (near-open) than the others.

## Step 3 — remove the R and shrink the topology

If one variable inductor can carry the entire match, do we need 5
components? Topology sweep over N=4, N=3, N=2 (all lossless LC, one
component per-band variable):

| N | topology | best variable | worst SWR |
|---|---|---|---|
| 4 | πLπL (Csh-Lse-Csh-Lse) | L_series | **1.012** |
| 4 | LπLπ (Lsh-Cse-Lsh-Cse) | C_series | 1.023 |
| **3** | LπL (Lsh-Cse-Lsh) | C_series | **1.021** ← |
| 3 | πLπ (Cse-Lsh-Cse) | C_series | 1.122 |
| 3 | πCπ (Csh-Lse-Csh) | L_series | 1.361 |
| 3 | LCL (Lse-Csh-Lse) | L_series | 1.048 |
| **2** | C_series + L_shunt | **C_series** | **1.147** ← |
| 2 | L_series + C_shunt | C_shunt | 1.418 |
| 2 | C_shunt + L_series | L_series | 2.222 |
| 2 | L_shunt + C_series | C_series | 2.743 |

**N=3 LπL gets SWR 1.02 — same as N=4** with one fewer component.
**N=2 L-match (C_series + L_shunt) gets SWR 1.15** with only 2 components.
The other 3 L-network orientations are much worse — only the
*series-C-then-shunt-L* orientation works because the antenna is mostly
inductive at the design freqs (positive imag), so it needs the series
capacitor to cancel reactance before the shunt L sets the impedance level.

## Step 4 — L-tuner with only 2 distinct C values

The N=2 L-tuner needs 4 distinct C values (one per band). Can we use just
2? Group bands in pairs:

| pairing | worst SWR |
|---|---|
| **alternating: 17m+12m on C_A, 15m+10m on C_B** | **1.32** ← |
| outer/inner: 17m+10m on C_A, 15m+12m on C_B | 2.40 |
| low/high: 17m+15m on C_A, 12m+10m on C_B | 2.64 |

**Alternating pairing wins by 2×.** The other two pairings are
qualitatively worse. The L-tuner's matching impedance vs frequency curve
crosses 50 Ω at two frequencies that are roughly a factor of √2 apart;
the alternating pairing puts band 1+3 (1.37×) on one cap and band 2+4
(1.33×) on the other, both within the L-tuner's natural double-crossing
range.

### The 2-cap L-tuner final design

- **L_shunt = 0.634 µH** (1 fixed inductor)
- **C_A = 64 pF** (switch position A: 17m + 12m)
- **C_B = 310 pF** (switch position B: 15m + 10m)
- **Antenna: f_low = 21.26 MHz, f_high = 28.56 MHz** (legs 3.35 m, 2.50 m)
- **Per-band SWR: 1.32 / 1.32 / 1.32 / 1.32** (perfectly equalized)
- **No resistor; no loss; no variable cap; no autotuner**

A DPDT band switch (or a 4-pole switch since coax + ground may need to be
switched too) is the only mechanical part beyond the fixed L and 2 fixed
caps. Cost-wise this is a $10 box for an antenna match — the actual cost
of building one is dominated by the chassis and connectors, not the
matching network.

## Engineering progression — what we learned

| design | components | loss | worst SWR | bands | notes |
|---|---|---|---|---|---|
| Lossy 8-element (morning) | 8 (with 2 R) | 2 dB | 2.50 | 5 | original target |
| Lossy 5-element (N=5, 5-band) | 5 (with 1 R) | 2 dB | 2.59 | 5 | sweet-spot complexity |
| Lossy 5-element (4-band) | 5 (with 1 R) | 2 dB | 1.90 | 4 | dropping 20m helps a lot |
| Lossy 5-element (4-band) | 5 (with 1 R) | 1.5 dB | 1.99 | 4 | sub-2 SWR for ~30% power |
| **Lossless N=3 (4-band, per-band tuned)** | 3 | 0 | 1.02 | 4 | one switched cap, three fixed parts |
| **Lossless N=2 L-tuner (4-band, per-band C)** | 2 | 0 | 1.15 | 4 | one switched cap, one fixed L |
| **Lossless N=2 L-tuner (4-band, 2-cap switch)** | 2 | 0 | 1.32 | 4 | one band switch, two fixed caps, one fixed L |

## Caveats

- **All analysis assumes lossless inductors and capacitors.** Real Q=100
  inductors at HF would add ~0.5 dB of loss; this is small enough not to
  ruin any of these designs, but the actual SWR may shift slightly.
- **Per-band C switching requires that the operator changes the switch
  position when changing bands.** Not an autotuner; not "set and
  forget".
- **The 4-band designs DO NOT cover 20m.** If 20m is required, fall back
  to the 5-band lossy designs from the morning.
- **Mutual coupling is still ignored.** A real fan dipole has inter-element
  coupling (see 2026-05-25 work). The "joint length optimization" assumes
  independent elements.
- **Optimizer may not always find the global optimum** at low loss budgets
  (multi-start coverage issue, well-documented in the morning doc). The
  monotonicity of the L-tuner Pareto curve is good, but spot-checks at low
  loss showed occasional local minima.

## Open questions / next directions

1. **Mutual coupling integration**: use yesterday's coupled `Z_a(jω)`
   model and re-run the L-tuner experiment to see how badly coupling
   degrades the 1.32 SWR result. This is the cleanest test of "is the
   L-tuner real for the actual antenna?".
2. **Finite-Q components**: redo the L-tuner optimization with
   inductor Q = 100 and capacitor Q = 1000 (typical mica). Expect
   SWR to shift slightly (within ±0.1) and a few hundred mW of loss
   to appear.
3. **20m via separate switch**: if the operator also wants 20m, design
   a 3-position switch (A, B, "20m") where the third position adds an
   extra fixed shunt component to extend coverage. Quick to test.
4. **Promote to package code**: today's lossy framework lives in
   scripts. Add `R_shunt`/`R_series` to `SECTION_KINDS` and the ABCD
   cascade machinery to `src/antmatch/matching.py`. Add the joint-length
   solver to `solve_lc_ladder` as an option.

## Engineering recommendation

For a 4-band (17m/15m/12m/10m) fan dipole with inverted-V geometry:

**Build this:**
- 2-element fan dipole, legs trimmed for resonances at **21.3 MHz**
  (about 3.35 m per leg) and **28.6 MHz** (about 2.50 m per leg)
- Matcher box: 0.634 µH air-core inductor in shunt to ground at the
  output side, and a DPDT band switch between two fixed silver-mica
  capacitors (64 pF for the "17m+12m" position, 310 pF for the
  "15m+10m" position) in series with the input
- Achieves SWR ≤ 1.32 on every amateur band from 17m through 10m
  with zero matching-network loss

This is dramatically simpler and more efficient than the morning's
8-element 2-dB-loss design, at the cost of requiring a band-switch flip
between bands. Trade made: zero auto-magic, zero loss, two physical caps,
and operator presses a switch.

## Bug fixed today

`metrics_summary` was originally `loss_db = float(-10 * np.log10(...).max())`
which Python parses as `-10 * (log.max())` — returning the *smallest* per-band
loss, not the worst. Fixed in both `lossy_5band_pareto.py` and
`lossy_5band_jointlen.py`. All results in this doc are post-fix.

## Files added today

Code (none in `src/`; all in `scripts/`):
- `bare_antenna_swr.py` — bare antenna SWR plots at 3 antenna configs
- `component_stress_breakdown.py` — per-component analysis for N=8 inverted-V
- `lossy_5band_demo.py` — first lossy demo (pre-bug-fix; kept for history)
- `lossy_5band_jointlen.py` — 5-band joint length opt
- `lossy_5band_pareto.py` — 5-band Pareto sweep
- `lossy_complexity_sweep.py` — N=3,4,5,6,8 comparison at 2 dB
- `lossy_fixed_R500.py` — off-the-shelf R penalty check
- `lossy_smith_2db.py` — Smith chart at 2 dB
- `n5_2db_breakdown.py` — N=5 5-band component analysis
- `n5_4band_2db_breakdown.py` — N=5 4-band component analysis
- `n5_4band_one_variable.py` — single-variable-per-band sweep
- `n5_4band_smith.py` — 4-band Smith chart
- `n5_loss_sweep.py` — N=5 5-band loss-budget sweep
- `n5_loss_sweep_4band.py` — N=5 4-band loss-budget sweep
- `resistor_power_breakdown.py` — per-resistor analysis
- `topology_reduction_sweep.py` — topology size sweep with one variable
- `l_tuner_smith.py` — Smith chart for L-tuner per-band design
- `l_tuner_2cap.py` — 2-cap L-tuner pairing analysis

Docs:
- `2026-05-26-lossy-matcher-pareto.md` (morning) — lossy framework + N=5 results
- `2026-05-26-l-tuner-discovery.md` (this file) — afternoon simplification
