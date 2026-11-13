"""Exact Heston base at H = 1/2: the log CF in closed form, and h = 0 where c vanishes.

At H = 1/2 the variance kernel is V0 + theta*(T - s), so integrating F(h) against it
by parts gives log phi = iurT + V0*h(T) + theta*int_0^T h(s) ds.  The exact Heston base
evaluates both terms in closed form, so no time grid enters.  At u = 0 and u = -i the
constant c = -u(u+i)/2 vanishes and h = 0, but G = (beta - D)/(beta + D) is 0/0 or x/0
there: at u = 0 for lam = 0, at u = -i for lam <= rho*nu.  The base returns h = 0 at those
nodes.  The expansion from H0 = 1/2 integrates the exponential kernel of its linearized
equation exactly, which keeps it stable at large u.  The other bases approximate the
solution at H = 1/2 and warn there.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from roughprix import RoughHeston, RoughHestonParam

SETS = {"A": dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02),
        # No mean reversion: beta = D = 0 at u = 0.
        "C": dict(lam=0.0, theta=0.0, nu=0.5, rho=-0.9, V0=0.04),
        # Positive correlation with lam < rho*nu: beta + D = 0 at u = -i.
        "D": dict(lam=0.1, theta=0.004, nu=0.5, rho=0.5, V0=0.03),
        # lam = rho*nu: beta = D = 0 at u = -i.
        "F": dict(lam=0.25, theta=0.01, nu=0.5, rho=0.5, V0=0.03)}
RATE = 0.03
K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])


def _model(set_name, **kw):
    return RoughHeston(RoughHestonParam(**SETS[set_name], H=0.5), riskfree=RATE, **kw)


def _ode_log_phi(p, a, T):
    """log phi at H = 1/2 from an adaptive ODE solve of h' = F(h), I' = h, without a time grid."""
    c = -0.5 * a * (a + 1j)
    beta = p["lam"] - p["rho"] * p["nu"] * a * 1j
    gam = p["nu"] ** 2

    def rhs(t, y):
        h = y[0] + 1j * y[1]
        dh = c - beta * h + 0.5 * gam * h * h
        return [dh.real, dh.imag, h.real, h.imag]

    y = solve_ivp(rhs, (0.0, T), [0.0] * 4, method="DOP853", rtol=1e-13, atol=1e-15).y[:, -1]
    return 1j * a * RATE * T + p["V0"] * (y[0] + 1j * y[1]) + p["theta"] * (y[2] + 1j * y[3])


@pytest.mark.parametrize("set_name", ["A", "C", "D"])
@pytest.mark.parametrize("base", ["auto", "exact_heston"])
def test_log_charfun_in_closed_form(base, set_name):
    """Five time steps match an adaptive ODE solve: the log CF does not depend on the grid."""
    u = np.array([0.01, 0.7, 2.0, 6.0, 20.0])
    arg = np.concatenate([u - 0.5j, -u - 1.2j, -u + 0.8j])
    T = 2.0
    log_phi = _model(set_name, base_solution=base).log_charfun(arg, maturity=T, time_steps=5)
    ref = np.array([_ode_log_phi(SETS[set_name], a, T) for a in arg])
    np.testing.assert_allclose(log_phi[:, 0], ref, rtol=0, atol=1e-10)


_PRICING = {"lewis": dict(),
            "otm_R1.2": dict(R=1.2, payoff_selection="otm"),
            "digital": dict(digital=True),
            "implied_vol": dict(implied_vol=True),
            "expansion_order_2": dict(expansion_order=2),
            "adaptive_panel": dict(integration="adaptive_panel")}


@pytest.mark.parametrize("kind", list(_PRICING))
def test_prices_do_not_depend_on_time_steps(kind):
    m = _model("A")
    kw = dict(K=K, maturity=1.0, **_PRICING[kind])
    np.testing.assert_array_equal(m.eur_call_put(time_steps=8, **kw),
                                  m.eur_call_put(time_steps=400, **kw))


def test_multi_H_column_at_half_is_closed_form():
    """The H = 1/2 column of a multi-H call equals the single-H price, at any time_steps."""
    m = _model("A")
    multi = m.eur_call_put(K=K, maturity=1.0, expansion_order=1, H=[0.5, 0.45], time_steps=50)
    single = m.eur_call_put(K=K, maturity=1.0, time_steps=400)
    np.testing.assert_array_equal(multi[:, 0], single)


@pytest.mark.parametrize("set_name", ["A", "C", "D", "F"])
@pytest.mark.parametrize("order, H", [(0, None), (2, None), (2, 0.45)], ids=["o0", "o2", "o2_H0.45"])
def test_charfun_at_zero_and_minus_i(order, H, set_name):
    """log phi(0) = 0 and log phi(-i) = rT, at the anchor and away from it."""
    T = 1.5
    log_phi = _model(set_name).log_charfun(np.array([0.0, -1j]), maturity=T, order=order,
                                           time_steps=50, H=H)
    np.testing.assert_array_equal(log_phi.ravel(), [0.0, RATE * T])


@pytest.mark.parametrize("set_name", ["A", "D", "F"])
def test_digital_calls_on_contour_through_minus_i(set_name):
    """R = 1 passes through u = -i at u = 0, a node for hN when expansion_order >= 1."""
    m = _model(set_name)
    kw = dict(K=K, maturity=1.0, digital=True, payoff_selection="as_requested",
              expansion_order=1, H=0.45, time_steps=100)
    np.testing.assert_allclose(m.eur_call_put(R=1.0, **kw), m.eur_call_put(R=1.2, **kw),
                               atol=1e-8, rtol=0)


def test_expansion_from_half_accurate_at_large_u():
    """Order 4 from the Heston anchor on the default 400-step grid; high vol-of-vol, u up to ~2000.

    There |D| dt reaches about 1.  The linearized solve integrates exp(-D (t - s)) exactly
    against piecewise-linear data, so it stays stable; a quadrature of the growing exp(D s)
    would not.
    """
    p = dict(lam=0.3, theta=0.006, nu=1.2, rho=-0.7, V0=0.02)
    T, H = 0.25, 0.4
    K = np.exp(np.array([-2, -1, 0, 1, 2]) * np.sqrt(p["V0"] * T))
    iv = RoughHeston(RoughHestonParam(**p, H=0.5), 0.0).eur_call_put(
        K=K, maturity=T, implied_vol=True, expansion_order=4, H=H)
    ref = RoughHeston(RoughHestonParam(**p, H=H), 0.0).eur_call_put(K=K, maturity=T, implied_vol=True)
    assert np.max(np.abs(np.asarray(iv) - np.asarray(ref))) * 1e4 < 0.5


@pytest.mark.parametrize("base", ["root_pade", "pade", "direct_riccati"])
def test_approximate_bases_at_half_warn(base):
    """Exact Heston is exact at H = 1/2.  Expansions from an approximate base there use an explicit
    left-rectangle scan, which is unstable once dt |F'(h)| exceeds about 2 (NaN on the default
    400 steps at nu = 1.2, order 4)."""
    with pytest.warns(UserWarning, match=f"base='{base}' at anchor H = 1/2 is approximate; use 'exact_heston'"):
        _model("A", base_solution=base).eur_call_put(K=K, maturity=1.0)
