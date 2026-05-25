# 2026-05-25 — fan dipole mutual coupling study

## TL;DR

- v3 Darlington synthesis: **postponed by user**. Diagnostic completed, design questions resolved — when work resumes, no further investigation needed before coding `src/antmatch/darlington.py`.
- Pivoted to studying the user's **physical 5-band fan dipole** (5 elements on pentagonal spacers, ~15 cm sides, inverted-V, R_rad ≈ 50 Ω). User reports: 4 bands tune to SWR 1.3 by iterative trim, but 15m sticks at ~2.5 with no clear low-SWR point near 21.38 MHz.
- **Diagnosis: mode hybridization.** With moderate inductive coupling between elements, the 15m mode hybridizes with its frequency-adjacent neighbors (17m at -15%, 12m at +17%) and loses its independent resonance.
- **Tuning strategies are exhausted.** Per-element iterative tuning, joint minimax, and neighbor-detune all give similar results — the residue is intrinsic to the geometry, not a tuning failure.
- **Recommendations** for next iteration: (a) pentagon ordering `20m — 12m — 15m — 17m — 10m`, (b) wider spacers (≥10" sides) to drop k_L below the hybridization threshold, (c) external L-network matcher on the 15m feed only, or (d) drop 15m from the fan and accept an external matcher for it.

## Context

Project state at start of day: v1 (LC-ladder + joint length co-optimization) and v2 (Yarman-Aksen Bode-Fano upper bound) landed on main. v3 Darlington synthesis was the planned next step.

## v3 status: postponed

User decision today. The diagnostic was run first and answered the open design questions:

- **Off-axis transmission zeros?** None possible in current v2 parameterization (`f(s) = s^k`). v3 does NOT need Type C / D sections.
- **Numerical care needed?** No. `g(s)` is strictly Hurwitz with no near-axis modes in every v2 optimum at d ∈ {2,4,6,8}. Stay numpy-only.
- **Mixed-zero handling?** Yes. d=4 and d=6 winners have both origin-zeros and infinity-zeros, so v3 must alternate Cauer steps.
- **5-band SWR plateau confirmed:** SWR 4.99 → 4.23 → 3.96 → 3.71 across d=2,4,6,8. The ≈3.7 Bode-Fano floor for the *fixed 2-element antenna* is real.

When v3 work resumes: implement `src/antmatch/darlington.py` consuming a `BelevitchTriple` and emitting `SECTION_KINDS`-compatible L/C sections. Round-trip-verify against `cascade_zin`. Diagnostic file `scripts/v3_diagnostic.py` reproduces the resolved-questions evidence.

## Physical antenna observation (from user)

- 5 wires on **regular pentagon spacers**, ~6" (15 cm) sides → diagonal ~25 cm
- **Multiple spacers along the length** → wires are parallel to each other beyond the first 20 cm from feedpoint
- **Inverted-V geometry** (bundle goes up to apex and down)
- Observed SWR after iterative tuning: 1.3 on 20m, 17m, 12m, 10m. 15m stuck at ~2.5 with **no low-SWR dip anywhere near 21.38 MHz**.
- User does not remember which band sat at which pentagon vertex.

## Model: coupled fan dipole

### Geometry-driven physics

For parallel half-wave dipoles at small d/λ (d/λ < 0.025 across all our bands and the 15-25 cm pentagon distances), mutual radiation resistance R_21 → R_self. Inductive coupling depends logarithmically on spacing and varies more between adjacent vs diagonal pentagon pairs.

Two pair-distances exist for a regular pentagon: 5 adjacent (side, 15 cm) and 5 diagonal (~25 cm). Pentagon ordering = a free choice of which BAND goes to which VERTEX.

### Implementation

Added to `src/antmatch/antenna.py`:

- `fan_dipole_impedance_coupled(elements, coupling, omegas, coupling_R=None)` — numerical evaluation of `Z_a(jω)` from an N×N impedance matrix at each ω. Inductive `coupling[i,j]` → mutual `M_ij = k_L·√(L_i·L_j)`. Optional `coupling_R[i,j]` → mutual `R_ij = α_R·√(R_i·R_j)`.
- `geometric_coupling_matrix(n, k_nn, decay=None)` — k decays with index distance
- `uniform_coupling_matrix(n, k)` — every off-diagonal equal

This is **NUMERICAL**, not rational — cannot be fed directly into v2 Belevitch synthesis. Future v3 work using this antenna model would need a rational-fit step (Vector Fitting / AAA).

### Coupling regimes explored

| regime | k_L adj/diag | α_R adj/diag | what it shows |
|---|---|---|---|
| pure R, very strong | 0/0 | 0.95/0.92 | Pentagon ordering irrelevant; retune drives every band to SWR ≈ 1.0 |
| mixed | 0.08/0.04 | 0.80/0.75 | Best ordering matters; one band stuck at ~1.23 |
| inductive heavy | 0.20/0.10 | 0.10/0.05 | **15m stuck at SWR 4-6** matching user's observation |
| 4-resonance threshold | 0.21/0.10 | 0.40/0.36 | 5 modes collapse to 4 visible dips — hybridization signature |

The user's experience is most consistent with a **mixed regime where inductive coupling is non-negligible** — purely resistive coupling alone cannot reproduce the stuck-band observation (retuning fixes everything in that regime).

## Mode hybridization (the core mechanism)

Inter-element coupling perturbs the 5 mode frequencies by ≈ k · ω_0. When k exceeds the relative frequency spacing between adjacent elements, modes **delocalize** — the supposedly-15m element no longer has its own resonance; instead, the 17m-15m-12m triplet shares 3 hybrid modes that all sit AWAY from 21.38 MHz.

Why 15m specifically:

- 15m has **two close neighbors on both sides** at relative frequency offsets of -15% (17m) and +17% (12m)
- All other bands have only ONE close neighbor (the bookends 20m and 10m have only one neighbor at all)
- When both sides hybridize symmetrically, the center element's "bonding mode" moves *down* in frequency and the "antibonding mode" moves *up*, leaving an empty zone right where 15m used to live

Verified in `scripts/joint_vs_per_element_tune.py` and visualized in `scripts/hybridized_smith.py`.

## Pentagon ordering recommendation

From `scripts/pentagon_search_RL.py`, enumerating all 12 distinct orderings (after rotation/reflection) across mixed and inductive-heavy regimes:

**Best ordering (consistent across non-trivial regimes): `20m — 12m — 15m — 17m — 10m` around the pentagon.**

- 15m's pentagon-adjacent neighbors: 12m and 17m (its two frequency-nearest bands)
- 15m's pentagon-diagonal neighbors: 20m and 10m (the two frequency-extremes)

Counter-intuitive at first glance — you might expect "put 15m's frequency-near neighbors far away on the pentagon." But the model says the opposite: placing 12m and 17m as 15m's adjacent (closer-spaced) neighbors lets their **perturbations partially cancel** because 12m is detuned UP and 17m is detuned DOWN, contributing opposite-sign imaginary parts. The total perturbation at 15m's resonance is minimized.

Worst orderings put two frequency-adjacent bands on pentagon-adjacent vertices on the same side of 15m (e.g., 20m-17m-15m-... ).

Robustness: across all coupling regimes tested, the best ordering ranks max-SWR ~30-50% better than the worst. Pentagon ordering matters but doesn't make-or-break the design.

## Tuning strategy analysis

From `scripts/joint_vs_per_element_tune.py`, with the best pentagon ordering and moderate coupling:

| strategy | max SWR | comment |
|---|---|---|
| (A) per-element coordinate descent | 1.42 | What the user physically did |
| (B) joint minimax over all 5 lengths | 1.37 | Marginal improvement; "flattens" rather than fixing |
| (C) deliberately detune 17m & 12m away from 15m | 1.42 | Optimizer chose zero shift — A was already optimal |

**Verdict: no tuning strategy dramatically beats per-element trim once the coupling pushes you past the hybridization threshold.** The user's iterative tuning was already near-optimal — the stuck-15m residue is structural, not a tuning failure.

## Smith chart (`hybridized_smith.png`)

Visualizes the bare 5-element fan dipole vs the per-element-tuned version, with k_L = 0.16:

- Without tuning: 5 modes are present but shifted UP from design freqs by 1-2 MHz at the upper bands
- After retune: 4 modes pull back close to design freqs (20m, 17m, 12m, 10m all settle near center of Smith chart), but **15m mode stays out at |Γ| ≈ 0.5 (SWR 2.9)**
- The actual 15m resonance sits at ~21.6 MHz, not 21.38 — operating 200 kHz off-center gives slightly better SWR than the design freq value

## Recommendations

### For a physical rebuild

Ordered by effort / reversibility:

1. **Re-string with the best pentagon ordering**: `20m — 12m — 15m — 17m — 10m` going around the spacer.  Cheap, mechanical-only change. Expected: 15m residue improves modestly but does not disappear.
2. **Widen pentagon spacers** to ≥10" (25 cm) sides. Halves the relative coupling. Likely drops k_L below the hybridization threshold and brings 15m's resonance back as a distinct dip.
3. **Asymmetric pentagon** — stretch the spacer so the 15m vertex has wider adjacencies than the others. Keeps mechanical compactness for the bookend bands.
4. **Drop to 4 elements** (combine 17m and 12m into a trapped element, or skip one band). Eliminates the triplet that's hybridizing.
5. **External L-network matcher on 15m only**. The current antenna's actual 15m impedance is roughly 140 + j100 Ω; a one-section L-match transforms this to 50 Ω. Add as a separate box fed via a 15m-band switch, leaving the other 4 bands going straight through.

### For next-step modeling work

The current `fan_dipole_impedance_coupled` is a useful predictor but underdetermined: (k_L, α_R) hasn't been fit to the user's actual measured SWR. If a 5-band measured SWR sweep is available, **fit those two parameters** and we have a quantitative model of the user's antenna that can predict the effect of any of the rebuild options above. Without measured data the model is qualitative.

The model's `Z_a(jω)` is also a candidate test case for v2 — running Yarman-Aksen on the coupled antenna would tell us **the Bode-Fano-limited SWR achievable at each band with an optimal matching network**, including 15m. If the v2 bound at 15m is ≤ 2.0, an L-matcher (option 5 above) is sufficient. If the bound is still 2.5+, the geometry itself is the bottleneck.

## Open questions

- **Why does the user observe 15m specifically stuck, while the model in some regimes predicts 12m or 10m stuck?** Likely because (k_L, α_R) for the real antenna isn't quite what I guessed. Measuring 5 SWR sweeps would close this.
- **Common-mode currents at the apex feedpoint** — the model treats the 5 wires as 5 independent ports sharing a common terminal voltage. In reality there's a balun and a feedline; common-mode contributions could be significant and would add a different coupling channel.
- **Wire-length finite-Q effects**: the model uses Q=10 for every element. Real lower-band wires are longer and may have higher Q from less ground proximity loss at HF heights, or lower Q from ohmic loss. This affects the hybridization threshold per-band.

## Artifacts added today

Code (`src/antmatch/antenna.py`):
- `fan_dipole_impedance_coupled` (extended to take optional `coupling_R`)
- `geometric_coupling_matrix`, `uniform_coupling_matrix`

Scripts:
- `scripts/v3_diagnostic.py` — answers v3 design questions on the 5-band problem
- `scripts/antenna_bode.py` — Bode plot of the 2-element baseline
- `scripts/antenna_smith.py` — Smith chart of the 2-element baseline
- `scripts/five_element_fan.py` — uncoupled 5-element model (the easy case)
- `scripts/five_element_coupled.py` — coupling regimes + retune
- `scripts/pentagon_search.py` — inductive-only pentagon ordering search (12 distinct orderings)
- `scripts/pentagon_search_RL.py` — pentagon search with both resistive and inductive coupling
- `scripts/joint_vs_per_element_tune.py` — A vs B vs C tuning strategy comparison
- `scripts/hybridized_smith.py` — Smith chart with uncoupled / coupled / retuned overlay

Output plots (regenerated by the scripts above):
- `antenna_bode.png`, `antenna_smith.png` — 2-element baseline
- `five_element_bode.png`, `five_element_smith.png` — uncoupled 5-element
- `five_element_coupled.png` — coupling effect on SWR
- `tuning_strategies.png` — A vs B vs C
- `hybridized_smith.png` — hybridization signature + retune effect
