# antmatch

Broadband matching network design with co-optimizable antenna geometry.

Tools for matching a configurable (e.g. fan-dipole) wire antenna to a real
source resistance across multiple discrete design frequencies. Provides a
simple analytical antenna model, an LC-ladder transducer-gain optimizer,
and a joint optimizer that varies dipole leg lengths and matching-network
element values together.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest                                                 # ~10 s, 18 tests
python scripts/joint_dipole_match_demo.py              # ~1-2 min
python scripts/yarman_aksen_demo.py                    # ~3-4 min
```

## What's in the box

- `antmatch.antenna` — series-RLC model of a half-wave dipole; parallel
  combination for a fan dipole. Returns Z_a(s) as ascending-power
  numerator/denominator coefficients.
- `antmatch.matching.solve_lc_ladder` — fixed-topology LC-ladder optimizer.
  Multi-start L-BFGS-B over log-element-values; max-min transducer gain.
- `antmatch.matching.solve_joint_lengths_and_ladder` — co-optimize dipole
  leg lengths *and* matching ladder. Passing `ladder_kinds=[]` runs a
  length-only optimization (isolates the value of picking the right
  lengths from the value of adding L/C).
- `antmatch.yarman_aksen.solve_yarman_aksen` — proper Carlin/Yarman-Aksen
  real-frequency technique using Belevitch polynomial parameterization of
  a lossless 2-port. Realizability enforced by construction (the (h, f, g)
  triple is bounded-real), so the worst-case transducer gain it returns
  is a true Bode-Fano upper bound for any lossless matching network of
  the chosen complexity. Reports T values only — synthesizing component
  values from the optimum (h, f, g) is a separate Darlington/Brune step
  (not implemented).
- `antmatch.matching.solve_carlin_upper_bound` — earlier scaffold,
  labeled-incomplete (does not enforce realizability). Superseded by
  `antmatch.yarman_aksen`; kept for the historical exposition.

## Antenna model caveats

Crude on purpose — exists to give the matching optimizer a realistic
*shape* of Z_a(s) to work against. Does not model:

- mutual coupling between fan elements (real impact: 1-3 % resonance
  shift at typical fan spacings)
- height-above-ground effects on radiation resistance
- wire-diameter-dependent Q
- harmonic resonances of each element

For final wire dimensions, cross-check with NEC.

## Relationship to claude-driving-point-synthesis

This repo grew out of the matching-network design discussion in the
sibling `claude-driving-point-synthesis` (`dpsynth`) repository. The two
are independent today: this repo only needs `numpy` and `scipy`. A
future connection would be:

- vector-fit a measured antenna Z(jω) → rational PR Z_a(s)
- design a Hurwitz Γ(s) via a proper Carlin/Yarman-Aksen pipeline
- form Z₁(s) = R_s·(1+Γ)/(1−Γ)
- hand Z₁(s) to `dpsynth.bott_duffin_synthesis` for the lossless
  matching-network LC structure

That synthesis hand-off is not implemented here.

## License

MIT. © 2026 Steven Burns.
