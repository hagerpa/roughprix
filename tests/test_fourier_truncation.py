"""Default Fourier cutoff L, set from the decay of the characteristic function.

For large u the Riccati solution approaches its stable root, so log|phi(u - i/2)| ~ -c u with
c = sqrt(1 - rho^2)/nu (V0 T^(1-a)/Gamma(2-a) + theta T) and a = H + 1/2: rough, short-dated
smiles decay slowly and need a large cutoff.  For small u the decay is Gaussian,
log|phi(u)| ~ -u^2/2 int_0^T E[V_t] dt, with E[V_t] bounded from below through the
Mittag-Leffler function; the bound stays positive at V0 = 0 and at theta = 0.  L is the larger
of the two points where |phi| reaches 1e-10, rounded up to a panel-template endpoint.  The
adaptive panel starts at L, widens it, and warns when no two successive cutoffs agree.
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from roughprix import RoughHeston, RoughHestonParam
from roughprix.heston.base_solution import _MEASURE_PANEL_TEMPLATE, _heuristic_measure_L

SETS = {"A": dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02),
        "B": dict(lam=1.0, theta=0.025, nu=0.4, rho=-0.65, V0=0.025),
        "C": dict(lam=0.0, theta=0.0, nu=0.5, rho=-0.9, V0=0.04)}
# set, H, T: default cutoffs 4000 and 8000 at one week, 150 at two years.
CASES = {"C_H0.1_1w": ("C", 0.1, 1 / 52),
         "A_H-0.2_1w": ("A", -0.2, 1 / 52),
         "B_H0.3_T2": ("B", 0.3, 2.0)}
L_WIDE = 64000.0  # at least 4x the default cutoff in every case


def _setup(case):
    set_name, H, T = CASES[case]
    m = RoughHeston(RoughHestonParam(**SETS[set_name], H=H), riskfree=0.0)
    K = np.exp(np.array([-3, -2, -1, 0, 1, 2, 3]) * np.sqrt(SETS[set_name]["V0"] * T))
    return m, K, T


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("integration", ["panel", "adaptive_panel"])
def test_default_L_converged_vanillas(integration, case):
    """Implied vols at the default cutoff match those at L_WIDE to 0.01 bp."""
    m, K, T = _setup(case)
    iv = m.eur_call_put(K=K, maturity=T, implied_vol=True, integration=integration)
    iv_wide = m.eur_call_put(K=K, maturity=T, implied_vol=True, L=L_WIDE)
    np.testing.assert_allclose(iv, iv_wide, rtol=0, atol=1e-6)


@pytest.mark.parametrize("case", list(CASES))
def test_default_L_converged_digitals(case):
    m, K, T = _setup(case)
    np.testing.assert_allclose(m.eur_call_put(K=K, maturity=T, digital=True),
                               m.eur_call_put(K=K, maturity=T, digital=True, L=L_WIDE),
                               rtol=0, atol=1e-10)


def test_heuristic_L_tracks_decay_rate():
    """Shorter maturities, rougher paths and steeper skews need a larger cutoff."""
    p = RoughHestonParam(**SETS["A"], H=0.1)
    L = _heuristic_measure_L(0.25, p, [0.1])
    assert L in _MEASURE_PANEL_TEMPLATE
    assert _heuristic_measure_L(1 / 52, p, [0.1]) > L
    assert _heuristic_measure_L(0.25, p, [-0.2]) > L
    assert _heuristic_measure_L(0.25, replace(p, rho=-0.95), [0.1]) > L
    # Several H: the cutoff of the slowest-decaying one.
    assert _heuristic_measure_L(0.25, p, [0.1, -0.2]) == _heuristic_measure_L(0.25, p, [-0.2])


@pytest.mark.parametrize("change, sd", [(dict(V0=0.0), 0.035), (dict(theta=0.0), 0.093)],
                         ids=["V0_zero", "theta_zero"])
def test_cutoff_below_the_cap(change, sd):
    """E[V_t] rises from V0 = 0, or falls towards theta/lam = 0; either way its lower bound stays
    positive and keeps the cutoff (4000, 700) below the cap 256000."""
    p = RoughHestonParam(**dict(SETS["A"], **change), H=0.1)
    assert _heuristic_measure_L(0.5, p, [0.1]) < _MEASURE_PANEL_TEMPLATE[-1]
    m = RoughHeston(p, riskfree=0.0)
    K = np.exp(np.array([-3, -2, -1, 0, 1, 2, 3]) * sd)   # sd: standard deviation of log S_T
    np.testing.assert_allclose(m.eur_call_put(K=K, maturity=0.5, implied_vol=True),
                               m.eur_call_put(K=K, maturity=0.5, implied_vol=True, L=256000.0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.eur_call_put(K=K, maturity=0.5, digital=True),
                               m.eur_call_put(K=K, maturity=0.5, digital=True, L=256000.0), rtol=0, atol=1e-10)


@pytest.mark.parametrize("candidates, warns", [(None, False), ([20.0, 35.0, 55.0], True)],
                         ids=["default", "far_too_small"])
def test_adaptive_panel_warns_unless_converged(candidates, warns):
    m, K, T = _setup("A_H-0.2_1w")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.eur_call_put(K=K, maturity=T, implied_vol=True, integration="adaptive_panel",
                       adaptive_L_candidates=candidates)
    assert any("adaptive_panel did not converge" in str(w.message) for w in caught) == warns
