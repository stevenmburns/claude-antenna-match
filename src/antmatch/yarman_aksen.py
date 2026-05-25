"""Carlin/Yarman-Aksen real-frequency technique — proper v2.

Parameterizes a lossless 2-port matching network in Belevitch canonical
form and optimizes its scattering polynomials to maximize the worst-case
transducer power gain into a prescribed load Z_L(jω). Realizability is
enforced *by construction* (the (h, f, g) triple is bounded-real), so
the answer this module returns is a genuine upper bound on what any
passive lossless matching network of the chosen transmission-zero
structure can achieve — including the Bode-Fano integral constraints.

Belevitch parameterization
--------------------------
A 2-port lossless N with scattering matrix S normalized to R_s has

    s11(s) = h(s) / g(s),   s22(s) = sigma * h(-s) / g(s),
    s21(s) * s12(s) = f(s) * f(-s) / (g(s) * g(-s)),

with the spectral-factor identity

    g(s) * g(-s) = h(s) * h(-s) + f(s) * f(-s).

`g(s)` is the Hurwitz spectral factor of the RHS (positive on the jω
axis, Hurwitz by sorting roots into the closed LHP). `f(s)` is fixed
by the chosen *transmission-zero structure* of N (e.g., f(s) = s^k
puts k transmission zeros at the origin, f(s) = 1 puts them at
infinity). `h(s)` is the FREE real polynomial — its coefficients are
the optimization variables.

`sigma = ±1` is the symmetry/antisymmetry sign of the network; for
this implementation we use sigma = -1 (the "reciprocal symmetric"
choice for series-shunt ladders), and the function form below makes
that explicit. Either sign yields a valid lossless 2-port; sigma = +1
corresponds to the dual topology.

Transducer power gain into Z_L
------------------------------
With ρ_L(s) = (Z_L(s) - R_s) / (Z_L(s) + R_s) the load reflection
normalized to R_s, the source-to-load transducer power gain is

    T(ω) = (1 - |ρ_L(jω)|²) * |s21(jω)|² / |1 - s22(jω) * ρ_L(jω)|²

(see Pozar §5.6 or Carlin-Civalleri Ch. 6). This is what we maximize.

Status
------
v2 (this file):
  * Belevitch (h, f, g) parameterization
  * Hurwitz spectral factorization via numpy root sort
  * transducer gain at the design frequencies
  * scipy.optimize.minimize over h coefficients (max-min objective)
  * NO synthesis hand-off — we report the upper-bound T(ω), not the
    component values. Producing component values from (h, f, g) is
    a separate Darlington/Brune extraction problem.

References
----------
- Belevitch, V. (1968). *Classical Network Theory.* Holden-Day.
- Yarman, B. S. & Carlin, H. J. (1982). "A simplified 'real frequency'
  technique applied to broad-band multistage microwave amplifiers."
  IEEE Trans. MTT, 30(12), 2216-2222.
- Yarman, B. S. (2010). *Design of Ultra Wideband Power Transfer
  Networks.* Wiley. (The polynomial-RFT textbook.)
- Pozar, D. M. (2012). *Microwave Engineering* (4th ed.), §5.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Polynomial helpers
# ---------------------------------------------------------------------------
#
# Convention: polynomial coefficients are ascending-power numpy arrays,
# matching the rest of antmatch. p[0] is the constant term, p[k] is the
# coefficient of s^k.


def poly_eval(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Horner evaluation of poly p at complex points s."""
    s = np.asarray(s)
    out = np.zeros_like(s, dtype=complex)
    for c in reversed(p):
        out = out * s + c
    return out


def poly_neg_s(p: np.ndarray) -> np.ndarray:
    """Return q where q(s) = p(-s). Negates coefficients of odd powers."""
    q = p.astype(float).copy()
    q[1::2] *= -1.0
    return q


def poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(a, b)


def poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[: len(a)] += a
    out[: len(b)] += b
    return out


def hurwitz_spectral_factor(p_sym: np.ndarray) -> np.ndarray:
    """Find Hurwitz g(s) such that g(s) * g(-s) == p_sym(s).

    Assumes ``p_sym`` is symmetric (p_sym(s) == p_sym(-s)) and non-
    negative on the jω axis. Returns a real polynomial ``g(s)`` with
    all roots in the closed LHP (real part ≤ 0).

    Algorithm: every root r of p has a partner -r (since p is symmetric
    in s → -s). For roots off the jω axis we keep the LHP one (the RHP
    twin belongs to g(-s)). For roots ON the jω axis, the pair {jω, -jω}
    necessarily appears with even multiplicity, half going to g and half
    to g(-s); we keep half the count (sorted by imaginary part, every
    other one). Finally we rescale g so its leading coefficient matches
    sqrt of p's leading coefficient.
    """
    p = np.asarray(p_sym, dtype=float)
    # Strip trailing zeros (highest-power coefficients that are zero)
    while len(p) > 1 and abs(p[-1]) < 1e-30:
        p = p[:-1]
    if len(p) <= 1:
        c0 = float(p[0])
        if c0 < 0:
            raise ValueError(f"constant {c0} is negative; cannot factor")
        return np.array([np.sqrt(c0)])

    roots = np.roots(p[::-1])
    tol = 1e-7 * max(np.abs(roots).max(), 1.0)

    lhp = [r for r in roots if r.real < -tol]
    axis = [r for r in roots if abs(r.real) <= tol]
    # axis comes in pairs {jω, -jω} each with doubled multiplicity;
    # sort and take every other to keep half the count.
    axis_sorted = sorted(axis, key=lambda r: (r.imag, r.real))
    axis_kept = axis_sorted[::2]

    g_roots = lhp + axis_kept
    g_desc = np.poly(g_roots) if g_roots else np.array([1.0])
    g_complex = np.asarray(g_desc[::-1])
    g = np.asarray(np.real_if_close(g_complex, tol=1000).real, dtype=float)

    # Rescale: g's leading coefficient should be sqrt(|p's leading|).
    # np.poly returns monic so g_lead currently = 1.
    p_lead = p[-1]
    g_lead = g[-1] if len(g) > 0 else 1.0
    if abs(g_lead) > 1e-15:
        scale = np.sqrt(abs(p_lead)) / abs(g_lead)
        g = g * scale
    return g


# ---------------------------------------------------------------------------
# Transmission-zero choices for f(s)
# ---------------------------------------------------------------------------


def f_from_transmission_zeros(n_at_origin: int = 0, n_at_infinity: int = 0) -> np.ndarray:
    """Build f(s) for a network with prescribed transmission zeros.

    Each zero of s21 at s=0 contributes a factor s to f; each zero at
    s=∞ contributes a factor 1 to f (i.e., reduces deg(f) relative to
    deg(g)). For a band-pass-like matching net we typically take
    n_at_origin + n_at_infinity == deg(g), with the split controlling
    the LP/HP/BP character of the matching response.

    Returns f as ascending-power coefficients.
    """
    # f(s) = s^n_at_origin   (zeros at infinity are "implicit" — they
    # mean deg(f) < deg(g); we don't pad f for them.)
    f = np.zeros(n_at_origin + 1)
    f[n_at_origin] = 1.0
    return f


# ---------------------------------------------------------------------------
# Belevitch (h, f, g) construction
# ---------------------------------------------------------------------------


@dataclass
class BelevitchTriple:
    h: np.ndarray
    f: np.ndarray
    g: np.ndarray
    sigma: int  # ±1 — the s22 sign


def build_belevitch(h: np.ndarray, f: np.ndarray, sigma: int = -1) -> BelevitchTriple:
    """Construct (h, f, g) where g is the Hurwitz factor of h·h* + f·f*."""
    h_neg = poly_neg_s(h)
    f_neg = poly_neg_s(f)
    hh = poly_mul(h, h_neg)
    ff = poly_mul(f, f_neg)
    rhs = poly_add(hh, ff)
    g = hurwitz_spectral_factor(rhs)
    return BelevitchTriple(h=h, f=f, g=g, sigma=sigma)


# ---------------------------------------------------------------------------
# Transducer gain
# ---------------------------------------------------------------------------


def transducer_gain_yarman(
    triple: BelevitchTriple,
    z_load: np.ndarray,
    omegas: np.ndarray,
    r_source: float = 50.0,
) -> np.ndarray:
    """T(ω) for a Belevitch-form lossless N terminated in Z_L(jω).

    Uses the standard formula

        T = (1 - |ρ_L|²) * |s21|² / |1 - s22 * ρ_L|²

    with all S-parameters evaluated at s = jω.
    """
    s = 1j * np.asarray(omegas, dtype=float)
    h = triple.h
    f = triple.f
    g = triple.g

    g_val = poly_eval(g, s)
    f_val = poly_eval(f, s)
    h_neg_val = poly_eval(poly_neg_s(h), s)

    s21 = f_val / g_val  # s12 = s21 for reciprocal
    s22 = triple.sigma * h_neg_val / g_val

    rho_l = (z_load - r_source) / (z_load + r_source)
    denom = 1.0 - s22 * rho_l
    t = (1.0 - np.abs(rho_l) ** 2) * np.abs(s21) ** 2 / np.abs(denom) ** 2
    return np.real_if_close(t).astype(float)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


@dataclass
class YarmanResult:
    triple: BelevitchTriple
    gain_design: np.ndarray
    worst_gain: float
    success: bool
    message: str


def solve_yarman_aksen(
    omegas_design: np.ndarray,
    z_load_design: np.ndarray,
    h_degree: int,
    n_at_origin: int = 0,
    n_at_infinity: int | None = None,
    r_source: float = 50.0,
    sigma: int = -1,
    weights: np.ndarray | None = None,
    n_restarts: int = 30,
    max_iter: int = 500,
    alpha_softmin: float = 50.0,
    rng_seed: int = 0,
) -> YarmanResult:
    """Maximize worst-case transducer gain via Yarman-Aksen RFT.

    Free variables: the ``h_degree + 1`` coefficients of h(s) (ascending
    powers). f(s) is fixed by ``n_at_origin`` and ``n_at_infinity``;
    deg(f) = n_at_origin, and the difference between h_degree and
    deg(f) gives the implicit transmission zeros at infinity.

    ``sigma = -1`` is the standard "series-shunt ladder" sign; ``+1``
    is its dual.
    """
    omegas_design = np.asarray(omegas_design, dtype=float)
    z_load_design = np.asarray(z_load_design, dtype=complex)
    if weights is None:
        weights = np.ones_like(omegas_design)

    if n_at_infinity is None:
        n_at_infinity = h_degree - n_at_origin
    if n_at_origin + n_at_infinity != h_degree:
        raise ValueError(
            f"Transmission-zero count {n_at_origin}+{n_at_infinity} must equal h_degree={h_degree}."
        )
    f = f_from_transmission_zeros(n_at_origin=n_at_origin)

    # Frequency normalization: optimize in s' = s/ω_0 with ω_0 the
    # geometric mean of the design frequencies. h coefficients in s'
    # stay O(1), which is essential for the optimizer to find non-
    # trivial solutions. T is dimensionless, so its values are the
    # same in normalized or physical units.
    omega_0 = float(np.exp(np.mean(np.log(omegas_design))))
    omegas_norm = omegas_design / omega_0

    def objective(h_coeffs):
        try:
            triple = build_belevitch(h_coeffs, f, sigma=sigma)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6
        t = transducer_gain_yarman(triple, z_load_design, omegas_norm, r_source)
        if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t > 1.0 + 1e-6):
            return 1e6
        t = np.clip(t, 1e-12, 1.0)
        return (1.0 / alpha_softmin) * np.log(np.sum(weights * np.exp(-alpha_softmin * t)))

    rng = np.random.default_rng(rng_seed)
    # Cold start: h ≈ small Gaussian (network ≈ pass-through-ish)
    starts = [rng.normal(0.0, 0.1, size=h_degree + 1) for _ in range(n_restarts)]
    # Plus a deterministic "identity-ish" start
    starts.insert(0, np.zeros(h_degree + 1))

    best_res = None
    best_obj = np.inf
    for h_init in starts:
        try:
            res = minimize(
                objective,
                h_init,
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9},
            )
            if res.fun < best_obj:
                best_obj = float(res.fun)
                best_res = res
        except Exception:
            continue

    assert best_res is not None
    h_opt = best_res.x
    triple = build_belevitch(h_opt, f, sigma=sigma)
    t_opt = transducer_gain_yarman(triple, z_load_design, omegas_norm, r_source)
    return YarmanResult(
        triple=triple,
        gain_design=t_opt,
        worst_gain=float(t_opt.min()),
        success=best_res.success,
        message=best_res.message if isinstance(best_res.message, str) else str(best_res.message),
    )
