"""Base density computation for rough-Heston Fourier pricing.

Evaluates the base characteristic function phi0 (expansion order=0)
and provides panel-quadrature heuristics for the Heston measure.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Panel heuristics (Heston-measure specific)
# ---------------------------------------------------------------------------

_MEASURE_PANEL_TEMPLATE = np.array([
    0.0, 3.0, 7.0, 12.0, 20.0, 35.0,
    55.0, 80.0, 110.0, 150.0, 200.0,
    270.0, 350.0, 500.0, 700.0, 1000.0,
    1400.0, 2000.0, 2800.0, 4000.0, 5600.0,
    8000.0, 11000.0, 16000.0, 22000.0, 32000.0,
    45000.0, 64000.0, 90000.0, 128000.0, 180000.0,
    256000.0,
])

# |phi| at the heuristic cutoff L.
_MEASURE_L_EPS = 1e-10


def _heuristic_measure_L(T: float, param, H_values) -> float:
    """Fourier cutoff where |phi| has decayed to about ``_MEASURE_L_EPS`` for every H.

    For large u the Riccati solution sits at its stable root, Re h ~ -u sqrt(1 - rho^2)/nu,
    so log|phi(u)| ~ -c u with c = sqrt(1 - rho^2)/nu * (V0 T^(1-a)/Gamma(2-a) + theta T)
    and a = H + 1/2.  For small u the decay is Gaussian, log|phi(u)| ~ -u^2/2 int_0^T xi,
    with the forward variance xi bounded from below by _integrated_variance_lower.  L is the
    larger of the two points where these reach log(eps), rounded up to a panel-template
    endpoint and capped at the last one.
    """
    log_eps = -np.log(_MEASURE_L_EPS)
    L = 0.0
    with np.errstate(divide="ignore"):
        for H in np.atleast_1d(H_values):
            a = float(H) + 0.5
            c = np.sqrt(1.0 - param.rho ** 2) / param.nu * (
                param.V0 * T ** (1.0 - a) / gamma(2.0 - a) + param.theta * T)
            L = max(L, log_eps / c, np.sqrt(2.0 * log_eps / _integrated_variance_lower(T, param, a)))
    idx = min(int(np.searchsorted(_MEASURE_PANEL_TEMPLATE, L)), len(_MEASURE_PANEL_TEMPLATE) - 1)
    return float(_MEASURE_PANEL_TEMPLATE[idx])


def _integrated_variance_lower(T: float, param, a: float) -> float:
    """Lower bound of int_0^T xi, the forward variance xi(t) = E[V_t].

    xi = theta/lam + (V0 - theta/lam) E_a(-lam t^a) moves monotonically from V0 towards
    theta/lam.  With y = t^a/Gamma(1 + a), two bounds on the Mittag-Leffler function give a
    lower bound either way.  Rising, E_a(-x) <= 1/(1 + x/Gamma(1 + a)) (T. Simon, 2015) gives
    xi(t) >= (V0 + theta y) / (1 + lam y): exact at lam = 0, and positive at V0 = 0.  Falling,
    E_a(-x) is completely monotone, hence log-convex, and so lies above its tangent in log
    scale, exp(-x/Gamma(1 + a)): xi(t) >= theta/lam + (V0 - theta/lam) exp(-lam y), exact at
    a = 1, and positive at theta = 0.
    """
    y = lambda t: t ** a / gamma(1.0 + a)
    if param.theta < param.lam * param.V0:
        m = param.theta / param.lam
        xi = lambda t: m + (param.V0 - m) * np.exp(-param.lam * y(t))
    else:
        xi = lambda t: (param.V0 + param.theta * y(t)) / (1.0 + param.lam * y(t))
    return quad(xi, 0.0, T)[0]


def _heuristic_measure_panels(L: float) -> np.ndarray:
    L = float(L)
    panels = _MEASURE_PANEL_TEMPLATE[_MEASURE_PANEL_TEMPLATE < L]
    if len(panels) == 0:
        return np.array([0.0, L], dtype=float)
    if panels[0] != 0.0:
        panels = np.concatenate([[0.0], panels])
    if panels[-1] < L:
        panels = np.concatenate([panels, [L]])
    return panels.astype(float)


def _default_heston_measure_panels(L: float) -> np.ndarray:
    base = np.array([0.0, 5.0, 15.0, 35.0, 75.0, 150.0, 250.0, 400.0, 700.0])
    inner = base[base < L]
    if len(inner) == 0 or inner[0] != 0.0:
        inner = np.concatenate([[0.0], inner])
    if inner[-1] < L:
        inner = np.concatenate([inner, [float(L)]])
    return inner


def _as_strict_panels(panels, L=None) -> np.ndarray:
    from roughprix.fourier import _as_strict_panels as _validate
    if panels is None:
        if L is None:
            raise ValueError("Either panels or L must be provided.")
        panels = _heuristic_measure_panels(float(L))
    return _validate(panels)


def _nested_measure_L_candidates(
    T: float,
    param,
    H_values,
    *,
    max_L: float | None = None,
) -> list[float]:
    """Return L candidates that are exactly _MEASURE_PANEL_TEMPLATE endpoints and nested.

    Starting from the heuristic cutoff L0 (itself a template endpoint), returns
    all remaining template endpoints up to max_L.
    Because candidates are exact template endpoints, _heuristic_measure_panels
    produces genuinely nested grids as L increases.
    """
    L0 = _heuristic_measure_L(T, param, H_values)
    endpoints = _MEASURE_PANEL_TEMPLATE[_MEASURE_PANEL_TEMPLATE > 0.0]
    idx_start = int(np.searchsorted(endpoints, L0, side='left'))
    candidates = endpoints[idx_start:]
    if max_L is not None:
        candidates = candidates[candidates <= float(max_L)]
    return [float(v) for v in candidates]


def _validate_nested_measure_L_candidates(candidates) -> list[float]:
    """Validate and return sorted unique nested template-endpoint L candidates.

    Raises ValueError if any candidate is not a _MEASURE_PANEL_TEMPLATE endpoint
    or if consecutive candidates produce non-nested panel grids.
    """
    cands = sorted(set(float(c) for c in candidates))
    template_set = set(_MEASURE_PANEL_TEMPLATE.tolist())
    for c in cands:
        if c <= 0:
            raise ValueError(f"L candidate {c} must be strictly positive.")
        if c not in template_set:
            raise ValueError(
                f"L candidate {c} is not a _MEASURE_PANEL_TEMPLATE endpoint. "
                "adaptive_panel incremental cache requires L candidates to be nested "
                "panel-template endpoints. Use candidates from _MEASURE_PANEL_TEMPLATE "
                "or disable incremental caching."
            )
    for L_old, L_new in zip(cands[:-1], cands[1:]):
        old = _heuristic_measure_panels(L_old)
        new = _heuristic_measure_panels(L_new)
        if not np.allclose(old, new[: len(old)]):
            raise ValueError(
                "adaptive_panel incremental cache requires L candidates to be nested "
                f"panel-template endpoints. L={L_old} panels are not a prefix of "
                f"L={L_new} panels. Use candidates from _MEASURE_PANEL_TEMPLATE or "
                "disable incremental caching."
            )
    return cands


