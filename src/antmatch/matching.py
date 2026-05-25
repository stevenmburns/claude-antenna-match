"""Broadband matching network design.

Two methods live here:

1. **Direct LC-ladder optimization** (``solve_lc_ladder``) — the working
   v1. Fix a topology (cascaded series/shunt L's and C's), evaluate the
   transducer gain analytically at the design frequencies, optimize the
   element values to maximize the worst-case gain. Bode-Fano is enforced
   automatically because the parameterization is realizable by
   construction. **Use this.**

2. **Carlin's real-frequency technique (RFT)** (``solve_carlin_upper_bound``)
   — work-in-progress. Parameterizes R_11(ω) at a small number of
   break-points, recovers X_11(ω) via Bode's reactance integral, and
   optimizes the transducer gain. The classical Carlin formulation also
   enforces a realizability constraint on the bounded-real reflection
   coefficient ρ(s); **that constraint is NOT in v1**, so the gain
   returned is an *upper bound* (often achieving T=1) that is not
   physically realizable by a lossless network terminated in Z_L. Kept
   as a scaffold for the proper Yarman-Aksen v2 (see task #6).

Goal (common to both methods)
-----------------------------
Design a lossless 2-port matching network N between a real source
resistance R_s (typically 50 Ω) and a complex load impedance Z_L(jω),
such that the transducer power gain

    T(ω) = 4 · R_s · Re[Z_in(jω)] / |Z_in(jω) + R_s|²

is as close to 1 as possible at a finite set of design frequencies.
Here Z_in is the impedance seen by the source looking into N
terminated in Z_L.

References
----------
- Carlin, H. J. (1977). "A new approach to gain-bandwidth problems."
  IEEE Trans. Circuits and Systems, CAS-24(4), 170-175.
- Yarman, B. S. & Carlin, H. J. (1982). "A simplified 'real frequency'
  technique applied to broad-band multistage microwave amplifiers."
  IEEE Trans. MTT, 30(12), 2216-2222.
- Carlin, H. J. & Civalleri, P. P. (1998). *Wideband Circuit Design.*
  CRC Press. Chapters 6-8.
- Pozar, D. M. (2012). *Microwave Engineering* (4th ed.). Wiley.
  §5.1 (matching) and §5.6 (Bode-Fano).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Problem statement
# ---------------------------------------------------------------------------


@dataclass
class MatchingProblem:
    """Specification of a broadband matching problem.

    Attributes:
        omegas_design: angular frequencies (rad/s) where matching matters.
        z_load_design: Z_L(jω_k) complex samples at those design freqs.
        omegas_grid:   dense ω grid (rad/s) for the Hilbert integral.
            Should bracket the design frequencies with several decades
            of headroom on each side so the integral converges.
        z_load_grid:   Z_L(jω) sampled on omegas_grid.
        omegas_breakpts: ω values where R_11 is parameterized (the free
            variables). Piecewise-linear in log-ω between break points.
            Choose few enough (≈5-15) that the problem is genuinely
            Bode-Fano-constrained; with breakpts >> design points the
            optimizer trivially achieves T=1 everywhere.
        r_source:      source resistance (real, positive, default 50 Ω).
    """

    omegas_design: np.ndarray
    z_load_design: np.ndarray
    omegas_grid: np.ndarray
    z_load_grid: np.ndarray
    omegas_breakpts: np.ndarray
    r_source: float = 50.0
    weights: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if self.weights.size == 0:
            self.weights = np.ones_like(self.omegas_design)
        assert len(self.omegas_design) == len(self.z_load_design)
        assert len(self.omegas_grid) == len(self.z_load_grid)


# ---------------------------------------------------------------------------
# Hilbert transform via Bode integral
# ---------------------------------------------------------------------------


def hilbert_from_resistance(omegas: np.ndarray, r_omega: np.ndarray) -> np.ndarray:
    """Recover X(ω) from R(ω) for a minimum-phase, even-R(ω) Z(jω).

    For a PR minimum-phase Z(jω) = R(ω) + j X(ω) with R(ω) = R(-ω)
    (even) and X(ω) = -X(-ω) (odd), Bode's reactance theorem gives

        X(ω) = (2ω/π) · P.V. ∫_0^∞ R(u)/(u² - ω²) du            (*)

    We approximate (*) by trapezoidal quadrature on the given ω grid.
    The singularity at u = ω is handled by *exclusion*: at each ω_k
    on the grid we drop the offending sample. This is crude (O(Δω)
    error near the singularity) but adequate for v1; a Cauchy P.V.
    quadrature would be the proper upgrade.
    """
    omegas = np.asarray(omegas)
    r_omega = np.asarray(r_omega)
    n = len(omegas)
    x_omega = np.zeros(n)

    for k in range(n):
        w = omegas[k]
        u2 = omegas**2
        denom = u2 - w**2
        # exclude the singular point
        mask = np.abs(denom) > 1e-30
        integrand = np.where(mask, r_omega / np.where(mask, denom, 1.0), 0.0)
        # trapezoidal on the unmasked points (with the dropped sample
        # contributing 0; small bias near resonances of R)
        x_omega[k] = (2.0 * w / np.pi) * np.trapezoid(integrand, omegas)
    return x_omega


# ---------------------------------------------------------------------------
# Transducer power gain
# ---------------------------------------------------------------------------


def transducer_gain(
    r_in: np.ndarray,
    x_in: np.ndarray,
    z_load: np.ndarray,
    r_source: float,
) -> np.ndarray:
    """T(ω) at the supplied samples.

    Z_in is what the source sees looking into the matching network terminated
    in Z_L. For a lossless network, the *available* power into Z_in equals
    the power delivered to Z_L, so optimizing T(ω) over a parameterization
    of Z_in(jω) is equivalent to optimizing the source-to-load match.
    """
    z_in = r_in + 1j * x_in
    return 4.0 * r_source * z_in.real / np.abs(z_in + r_source) ** 2


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass
class CarlinResult:
    r11_grid: np.ndarray  # R_11(ω) on the dense grid
    x11_grid: np.ndarray  # X_11(ω) on the dense grid (from Hilbert)
    gain_design: np.ndarray  # T(ω_k) at the design frequencies
    worst_gain: float  # min_k T(ω_k)
    success: bool
    message: str


def _interp_at(
    omegas_grid: np.ndarray, values: np.ndarray, omegas_q: np.ndarray
) -> np.ndarray:
    return np.interp(omegas_q, omegas_grid, values)


def _r11_from_breakpts(
    omegas_grid: np.ndarray, omegas_breakpts: np.ndarray, r_breakpts: np.ndarray
) -> np.ndarray:
    """Piecewise-linear-in-log-ω interpolation from break points to grid."""
    return np.interp(np.log(omegas_grid), np.log(omegas_breakpts), r_breakpts)


def solve_carlin_upper_bound(
    problem: MatchingProblem,
    r_breakpts_init: np.ndarray | None = None,
    max_iter: int = 200,
) -> CarlinResult:
    """Hilbert-based Carlin RFT — UPPER BOUND, NOT REALIZABLE in v1.

    Free variables are R_11 sampled at ``problem.omegas_breakpts``.
    R_11 is linearly interpolated (in log-ω) onto ``omegas_grid`` for
    the Hilbert integral. Objective: maximize min_k T(ω_k).

    ⚠️  This v1 does NOT enforce the bounded-real factorization that
    forces Z_in(s) to come from a lossless network terminated in Z_L.
    The optimizer therefore routinely returns T=1 at every design
    frequency — an upper bound that no physical network achieves.
    Use ``solve_lc_ladder`` for an honest answer.
    """
    n_bp = len(problem.omegas_breakpts)

    if r_breakpts_init is None:
        r_breakpts_init = np.full(n_bp, problem.r_source)

    def unpack(x):
        return np.clip(x, 1e-6, None)

    def objective(x):
        r_bp = unpack(x)
        r11_grid = _r11_from_breakpts(
            problem.omegas_grid, problem.omegas_breakpts, r_bp
        )
        x11_grid = hilbert_from_resistance(problem.omegas_grid, r11_grid)
        r_at = _interp_at(problem.omegas_grid, r11_grid, problem.omegas_design)
        x_at = _interp_at(problem.omegas_grid, x11_grid, problem.omegas_design)
        t = transducer_gain(r_at, x_at, problem.z_load_design, problem.r_source)
        # Minimize -soft_min(t) = (1/α) log sum exp(-α t).
        alpha = 50.0
        return (1.0 / alpha) * np.log(np.sum(problem.weights * np.exp(-alpha * t)))

    bounds = [(1e-6, None)] * n_bp
    res = minimize(
        objective,
        r_breakpts_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-7},
    )

    r_bp = unpack(res.x)
    r11_grid = _r11_from_breakpts(problem.omegas_grid, problem.omegas_breakpts, r_bp)
    x11_grid = hilbert_from_resistance(problem.omegas_grid, r11_grid)
    r_at = _interp_at(problem.omegas_grid, r11_grid, problem.omegas_design)
    x_at = _interp_at(problem.omegas_grid, x11_grid, problem.omegas_design)
    t = transducer_gain(r_at, x_at, problem.z_load_design, problem.r_source)

    return CarlinResult(
        r11_grid=r11_grid,
        x11_grid=x11_grid,
        gain_design=t,
        worst_gain=float(t.min()),
        success=res.success,
        message=res.message if isinstance(res.message, str) else str(res.message),
    )


# ---------------------------------------------------------------------------
# Baseline (no matching network)
# ---------------------------------------------------------------------------


def baseline_gain(z_load: np.ndarray, r_source: float) -> np.ndarray:
    """T(ω) for the source connected directly to the load (no matching)."""
    return 4.0 * r_source * z_load.real / np.abs(z_load + r_source) ** 2


# ---------------------------------------------------------------------------
# Direct LC-ladder topology optimization (v1 — actually works)
# ---------------------------------------------------------------------------


# Element kinds for a ladder section. Series elements stack along the
# signal path; shunt elements drop to ground.
SECTION_KINDS = ("L_series", "C_series", "L_shunt", "C_shunt")


def cascade_zin(
    kinds: list[str],
    values: np.ndarray,
    z_load: np.ndarray,
    omegas: np.ndarray,
) -> np.ndarray:
    """Z_in(jω) of a cascaded LC ladder terminated in Z_load(jω).

    Ladder is processed from the load end back toward the source. At each
    section:
      * series element: Z_in_new = Z_in_old + Z_elem
      * shunt element:  Z_in_new = (Z_in_old · Z_elem) / (Z_in_old + Z_elem)

    The first kind in ``kinds`` is the section CLOSEST TO THE SOURCE,
    so iteration runs over kinds[::-1] starting from the load.
    """
    z = z_load.astype(complex).copy()
    for kind, val in zip(reversed(kinds), reversed(values), strict=True):
        if kind == "L_series":
            z = z + 1j * omegas * val
        elif kind == "C_series":
            z = z + 1.0 / (1j * omegas * val)
        elif kind == "L_shunt":
            z_e = 1j * omegas * val
            denom = z + z_e
            z = np.where(np.abs(denom) > 1e-30, (z * z_e) / denom, z_e)
        elif kind == "C_shunt":
            z_e = 1.0 / (1j * omegas * val)
            denom = z + z_e
            z = np.where(np.abs(denom) > 1e-30, (z * z_e) / denom, z_e)
        else:
            raise ValueError(f"unknown section kind {kind!r}")
    return z


@dataclass
class LadderResult:
    kinds: list[str]
    values: np.ndarray  # SI units (H for L, F for C)
    z_in_design: np.ndarray  # Z_in(jω) at design frequencies
    gain_design: np.ndarray  # T(ω_k) at design frequencies
    worst_gain: float
    success: bool
    message: str


def _initial_values(
    kinds: list[str], omega_center: float, r_source: float
) -> np.ndarray:
    """Reasonable cold-start values centered on the design band."""
    init = np.empty(len(kinds))
    for i, k in enumerate(kinds):
        if k.startswith("L"):
            init[i] = r_source / omega_center  # ωL ≈ R_s
        else:
            init[i] = 1.0 / (r_source * omega_center)  # 1/(ωC) ≈ R_s
    return init


def solve_lc_ladder(
    problem: MatchingProblem,
    kinds: list[str],
    init_values: np.ndarray | None = None,
    max_iter: int = 500,
    alpha_softmin: float = 50.0,
    n_restarts: int = 20,
    rng_seed: int = 0,
) -> LadderResult:
    """Optimize a fixed-topology LC ladder for best worst-case match.

    Args:
        problem: load specification (uses ``omegas_design``, ``z_load_design``,
            ``r_source``; ``omegas_grid``/``omegas_breakpts`` ignored).
        kinds:   ordered list of section kinds (source-to-load order). Use
            entries from ``SECTION_KINDS``.
        init_values: optional explicit start. If given, used as one start
            among the restarts.
        n_restarts: number of random log-perturbed starts (in addition to
            the physics-based start). Best worst-case gain across all
            starts wins. The objective has many local minima — restarts
            are essential.

    Returns the optimized element values and per-band gain.
    """
    if any(k not in SECTION_KINDS for k in kinds):
        bad = [k for k in kinds if k not in SECTION_KINDS]
        raise ValueError(f"unknown section kinds: {bad}")

    omega_center = float(np.exp(np.mean(np.log(problem.omegas_design))))
    physics = _initial_values(kinds, omega_center, problem.r_source)

    def objective(log_x):
        values = np.exp(log_x)
        z_in = cascade_zin(kinds, values, problem.z_load_design, problem.omegas_design)
        t = transducer_gain(
            z_in.real, z_in.imag, problem.z_load_design, problem.r_source
        )
        return (1.0 / alpha_softmin) * np.log(
            np.sum(problem.weights * np.exp(-alpha_softmin * t))
        )

    starts: list[np.ndarray] = [np.log(physics)]
    if init_values is not None:
        starts.append(np.log(init_values))
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_restarts):
        starts.append(np.log(physics) + rng.normal(0.0, 2.0, size=len(kinds)))

    best_res = None
    best_obj = np.inf
    for log_init in starts:
        res = minimize(
            objective,
            log_init,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9},
        )
        if res.fun < best_obj:
            best_obj = float(res.fun)
            best_res = res

    assert best_res is not None
    values_opt = np.exp(best_res.x)
    z_in_opt = cascade_zin(
        kinds, values_opt, problem.z_load_design, problem.omegas_design
    )
    t_opt = transducer_gain(
        z_in_opt.real, z_in_opt.imag, problem.z_load_design, problem.r_source
    )

    return LadderResult(
        kinds=list(kinds),
        values=values_opt,
        z_in_design=z_in_opt,
        gain_design=t_opt,
        worst_gain=float(t_opt.min()),
        success=best_res.success,
        message=best_res.message
        if isinstance(best_res.message, str)
        else str(best_res.message),
    )


def format_ladder(result: LadderResult) -> str:
    """Pretty-print an optimized ladder with SI units."""
    lines = []
    for i, (k, v) in enumerate(zip(result.kinds, result.values, strict=True)):
        if k.startswith("L"):
            lines.append(f"  sec{i:02d} {k:9s}  {v * 1e9:.4g} nH")
        else:
            lines.append(f"  sec{i:02d} {k:9s}  {v * 1e12:.4g} pF")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Joint optimization: dipole leg lengths AND matching ladder
# ---------------------------------------------------------------------------


@dataclass
class JointResult:
    leg_lengths_m: np.ndarray  # optimized dipole leg lengths
    ladder_kinds: list[str]
    ladder_values: np.ndarray  # SI element values
    z_load_design: np.ndarray  # antenna Z_a(jω_k) at optimum lengths
    z_in_design: np.ndarray  # Z seen by source after ladder
    gain_design: np.ndarray
    worst_gain: float
    success: bool
    message: str


def solve_joint_lengths_and_ladder(
    omegas_design: np.ndarray,
    n_dipoles: int,
    ladder_kinds: list[str],
    leg_bounds_m: tuple[float, float] = (0.1, 20.0),
    r_source: float = 50.0,
    weights: np.ndarray | None = None,
    n_restarts: int = 30,
    max_iter: int = 500,
    alpha_softmin: float = 50.0,
    rng_seed: int = 0,
    element_q: float = 10.0,
    element_r_rad: float = 65.0,
) -> JointResult:
    """Co-optimize dipole leg lengths and a matching LC ladder.

    Variables (all in log-space): ``n_dipoles`` leg lengths in meters,
    followed by ``len(ladder_kinds)`` element values in SI. Objective:
    maximize the worst-case transducer gain over the design frequencies.

    ``ladder_kinds`` may be empty — that runs a *length-only* optimization
    (no matching network), useful as a "just-pick-the-lengths-well"
    baseline.

    Returns the optimum lengths, ladder values, and per-band gains.
    """
    # Local import to avoid circular dependency at module load.
    from .antenna import (
        DipoleElement,
        evaluate_rational,
        fan_dipole_impedance,
    )

    if any(k not in SECTION_KINDS for k in ladder_kinds):
        bad = [k for k in ladder_kinds if k not in SECTION_KINDS]
        raise ValueError(f"unknown section kinds: {bad}")

    omegas_design = np.asarray(omegas_design)
    if weights is None:
        weights = np.ones_like(omegas_design)

    omega_center = float(np.exp(np.mean(np.log(omegas_design))))
    # Physics start: legs spaced log-uniformly to put each resonance near
    # one of n_dipoles geometric-mean clusters of design frequencies.
    cluster_centers = np.exp(
        np.linspace(
            np.log(omegas_design.min()),
            np.log(omegas_design.max()),
            n_dipoles,
        )
    )
    legs_phys = VF_BARE_WIRE * C_LIGHT_LOCAL / (2.0 * cluster_centers)
    ladder_phys = (
        _initial_values(ladder_kinds, omega_center, r_source)
        if ladder_kinds
        else np.array([])
    )

    log_lb = np.log(leg_bounds_m[0])
    log_ub = np.log(leg_bounds_m[1])

    def split(x):
        return x[:n_dipoles], x[n_dipoles:]

    def evaluate(legs_m, ladder_vals):
        elements = [
            DipoleElement(
                leg_length_m=float(L),
                r_rad=element_r_rad,
                q=element_q,
            )
            for L in legs_m
        ]
        num, den = fan_dipole_impedance(elements)
        z_load = evaluate_rational(num, den, 1j * omegas_design)
        z_in = cascade_zin(ladder_kinds, ladder_vals, z_load, omegas_design)
        t = transducer_gain(z_in.real, z_in.imag, z_load, r_source)
        return z_load, z_in, t

    def objective(log_x):
        log_legs, log_ladder = split(log_x)
        legs = np.exp(np.clip(log_legs, log_lb, log_ub))
        ladder = np.exp(log_ladder) if len(log_ladder) else np.array([])
        try:
            _, _, t = evaluate(legs, ladder)
        except Exception:
            return 1e6
        if not np.all(np.isfinite(t)):
            return 1e6
        return (1.0 / alpha_softmin) * np.log(
            np.sum(weights * np.exp(-alpha_softmin * t))
        )

    # Bounds: legs clipped to physical range; ladder values unbounded.
    bounds = [(log_lb, log_ub)] * n_dipoles + [(None, None)] * len(ladder_kinds)

    starts: list[np.ndarray] = [
        np.concatenate(
            [
                np.log(legs_phys),
                np.log(ladder_phys) if len(ladder_phys) else np.array([]),
            ]
        )
    ]
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_restarts):
        leg_perturb = rng.normal(0.0, 0.5, size=n_dipoles)  # ~×÷1.6
        ladder_perturb = (
            rng.normal(0.0, 2.0, size=len(ladder_kinds))
            if ladder_kinds
            else np.array([])
        )
        log_start = np.concatenate(
            [
                np.clip(np.log(legs_phys) + leg_perturb, log_lb, log_ub),
                np.log(ladder_phys) + ladder_perturb
                if len(ladder_phys)
                else np.array([]),
            ]
        )
        starts.append(log_start)

    best_res = None
    best_obj = np.inf
    for log_init in starts:
        res = minimize(
            objective,
            log_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9},
        )
        if res.fun < best_obj:
            best_obj = float(res.fun)
            best_res = res

    assert best_res is not None
    log_legs, log_ladder = split(best_res.x)
    legs_opt = np.exp(np.clip(log_legs, log_lb, log_ub))
    ladder_opt = np.exp(log_ladder) if len(log_ladder) else np.array([])
    z_load_opt, z_in_opt, t_opt = evaluate(legs_opt, ladder_opt)

    return JointResult(
        leg_lengths_m=legs_opt,
        ladder_kinds=list(ladder_kinds),
        ladder_values=ladder_opt,
        z_load_design=z_load_opt,
        z_in_design=z_in_opt,
        gain_design=t_opt,
        worst_gain=float(t_opt.min()),
        success=best_res.success,
        message=best_res.message
        if isinstance(best_res.message, str)
        else str(best_res.message),
    )


# Constants used above. Cannot import from .antenna at module scope
# without inducing a circular import; copy them here.
C_LIGHT_LOCAL = 299_792_458.0
VF_BARE_WIRE = 0.95
