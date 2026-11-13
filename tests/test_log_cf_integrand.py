"""The log characteristic function from the Riccati solution h, or from F(h).

log phi(u, T) = iurT + int_0^T F(h(s)) g(T - s) ds with g(t) = V0 + theta t^alpha / Gamma(1 + alpha),
the formula of Hager & Kreher (2026), Eq. (1.1), which ``log_cf_integrand="rhs"`` integrates.  Since
h = I^alpha F(h), the integral equals theta * I^1 h(T) + V0 * I^(1-alpha) h(T), which ``"solution"``
integrates.  Both are exact for the exact h.  For an approximate h, "rhs" is "solution" after one
fixed-point step of the target equation: it restores the short-time behaviour t^alpha but amplifies
the error of h by about nu|u|.  Root-Pade near H = 0 is several times more accurate with
"solution"; expansions from H0 = 1/2 are more accurate with "rhs".  The default "auto" uses "rhs"
for expansions from H0 = 1/2 and "solution" otherwise.  The default time grid is graded with 400
steps.
"""
import numpy as np
import pytest

from roughprix import RoughHeston, RoughHestonParam

_A = dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02)
_ARG = np.linspace(0.1, 60.0, 25) - 0.5j


def test_invalid_log_cf_integrand_raises():
    with pytest.raises(ValueError, match="log_cf_integrand"):
        RoughHeston(RoughHestonParam(**_A, H=0.1), 0.0, log_cf_integrand="F")


@pytest.mark.parametrize("H", [0.1, -0.2])
def test_forms_agree_for_an_accurate_solution(H):
    """Implicit-pi on a fine grid: both formulas give the same log CF."""
    p = RoughHestonParam(**_A, H=H)
    kw = dict(maturity=1.0, time_steps=2000)
    sol = RoughHeston(p, 0.0, base_solution="direct_riccati").log_charfun(_ARG, **kw)
    rhs = RoughHeston(p, 0.0, base_solution="direct_riccati", log_cf_integrand="rhs").log_charfun(_ARG, **kw)
    np.testing.assert_allclose(sol, rhs, rtol=0, atol=1e-4)


@pytest.mark.filterwarnings("ignore:base='direct_riccati' at anchor H = 1/2")
def test_solution_form_at_half():
    """At alpha = 1, I^(1 - alpha) is the identity; implicit-pi then matches the closed form."""
    p = RoughHestonParam(**_A, H=0.5)
    num = RoughHeston(p, 0.0, base_solution="direct_riccati").log_charfun(_ARG, maturity=1.0)
    exact = RoughHeston(p, 0.0).log_charfun(_ARG, maturity=1.0)
    np.testing.assert_allclose(num, exact, rtol=0, atol=3e-5)


def test_default_time_grid_is_graded_400():
    m = RoughHeston(RoughHestonParam(**_A, H=0.1), 0.0)
    K = np.array([0.9, 1.0, 1.1])
    np.testing.assert_array_equal(
        np.asarray(m.eur_call_put(K=K, maturity=0.5)),
        np.asarray(m.eur_call_put(K=K, maturity=0.5, time_grid="graded", time_steps=400)))
    np.testing.assert_array_equal(
        m.log_charfun(_ARG, maturity=0.5),
        m.log_charfun(_ARG, maturity=0.5, time_grid="graded", time_steps=400))


# Root-Pade near H = 0 with high vol-of-vol: 0.1-0.15 bp off with "solution", about 2 bp with "rhs".
_RP_CASES = {
    "nu1.2_H0_T3m": (0.0, 0.25),
    "nu1.2_H0.02_T1m": (0.02, 1 / 12),
}


@pytest.mark.parametrize("case", list(_RP_CASES))
def test_root_pade_smile_with_solution_form(case):
    H, T = _RP_CASES[case]
    p = {**_A, "nu": 1.2}
    param = RoughHestonParam(**p, H=H)
    K = np.exp(np.array([-2, -1, 0, 1, 2]) * np.sqrt(p["V0"] * T))
    iv = RoughHeston(param, 0.0, base_solution="root_pade").eur_call_put(K=K, maturity=T, implied_vol=True)
    ref = RoughHeston(param, 0.0, base_solution="direct_riccati").eur_call_put(
        K=K, maturity=T, implied_vol=True, time_grid="graded", time_steps=400)
    assert np.max(np.abs(iv - ref)) * 1e4 < 1.5


@pytest.mark.parametrize("H0, H, form", [(0.5, 0.4, "rhs"), (0.5, 0.5, "solution"), (0.0, 0.1, "solution"),
                                         (0.1, 0.1, "solution")])
def test_auto_form(H0, H, form):
    """"auto" uses "rhs" for expansions from H0 = 1/2 and "solution" otherwise."""
    p = RoughHestonParam(**_A, H=H0)
    kw = dict(maturity=0.5, order=2, H=H)
    np.testing.assert_array_equal(RoughHeston(p, 0.0).log_charfun(_ARG, **kw),
                                  RoughHeston(p, 0.0, log_cf_integrand=form).log_charfun(_ARG, **kw))


def test_auto_form_in_pricing_from_half():
    p = RoughHestonParam(**_A, H=0.5)
    kw = dict(K=np.array([0.9, 1.0, 1.1]), maturity=0.5, expansion_order=2, H=0.4)
    np.testing.assert_array_equal(np.asarray(RoughHeston(p, 0.0).eur_call_put(**kw)),
                                  np.asarray(RoughHeston(p, 0.0, log_cf_integrand="rhs").eur_call_put(**kw)))
