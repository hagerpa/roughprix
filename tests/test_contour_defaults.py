"""The default Fourier contour R = 0.5, and digital put-call parity.

A damped contour at height R prices calls on +R and puts on -R, which needs
E[S_T^R] and E[S_T^-R] to be finite.  Moments of order p outside [0, 1] can
explode in finite time, negative ones the sooner the steeper the skew; at
H = 1/2 the explosion time is in closed form (``heston_explosion_time``), and
for SET_C E[S_T^-1.2] explodes at T = 2.78.  The default R = 0.5 prices every
strike as a call and the puts by parity, and E[S_T^0.5] <= S_0^0.5 e^{rT/2} is
finite for every T.  For digitals R = 0.5 gives the same prices as R = 1.2
wherever that contour exists.

Digital (cash-or-nothing) parity is C + P = e^{-rT}, not the vanilla
C - P = S_0 - K e^{-rT}.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from roughprix import RoughHeston, RoughHestonParam

# Defaults-like parameters, and a steep-skew set without mean reversion.
SET_A = dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02)
SET_C = dict(lam=0.0, theta=0.0, nu=0.5, rho=-0.9, V0=0.04)
RATE = 0.03

def heston_explosion_time(p, lam, nu, rho, **_):
    """Closed-form explosion time of E[S_T^p] in classical Heston."""
    if p * (p - 1) <= 0:
        return np.inf
    chi = rho * nu * p - lam
    delta = chi ** 2 - nu ** 2 * p * (p - 1)
    if delta >= 0:
        if chi < 0:
            return np.inf
        s = np.sqrt(delta)
        return np.log((chi + s) / (chi - s)) / s
    s = np.sqrt(-delta)
    return 2 / s * (np.pi * (chi < 0) + np.arctan(s / chi))


def heston_cf(z, T, lam, theta, nu, rho, V0):
    """E[exp(i z X)] with X = log(S_T / F) in classical Heston, drift theta - lam V."""
    iz = 1j * z
    b = lam - rho * nu * iz
    d = np.sqrt(b * b + nu ** 2 * (iz + z * z))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    Dt = (b - d) / nu ** 2 * (1 - e) / (1 - g * e)
    Ct = theta / nu ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return np.exp(Ct + Dt * V0)


def reference_put(K, T, params, r=RATE, S_0=1.0):
    """Put price from the Lewis formula, integrated with scipy.integrate.quad."""
    D = np.exp(-r * T)
    F = S_0 / D
    x = np.log(F / K)

    def integrand(u):
        return (np.exp(1j * u * x) * heston_cf(u - 0.5j, T, **params)).real / (u * u + 0.25)

    I = integrate.quad(integrand, 0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-12)[0]
    call = D * (F - np.sqrt(F * K) / np.pi * I)
    return call - S_0 + K * D


def reference_digital_put(K, T, params, r=RATE):
    """Cash-or-nothing put price from the Gil-Pelaez formula, via scipy.integrate.quad."""
    D = np.exp(-r * T)
    k = np.log(K * D)   # log(K / F) with S_0 = 1

    def integrand(u):
        return (np.exp(-1j * u * k) * heston_cf(u, T, **params) / (1j * u)).real

    pieces = ((0, 1), (1, 10), (10, 100), (100, 1000), (1000, np.inf))
    I = sum(integrate.quad(integrand, a, b, limit=2000, epsabs=1e-14, epsrel=1e-12)[0]
            for a, b in pieces)
    return D * (0.5 - I / np.pi)


def model(params, H=0.5, base="auto"):
    return RoughHeston(RoughHestonParam(**params, H=H), riskfree=RATE, base_solution=base)


# ---------------------------------------------------------------------------
# Vanillas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integration", ["panel", "adaptive_panel"])
def test_vanilla_default_is_lewis(integration):
    m = model(SET_A, H=0.1)
    K = np.array([0.8, 0.95, 1.0, 1.1, 1.3])
    kw = dict(K=K, maturity=0.5, time_steps=100, integration=integration,
              adaptive_metric="price")
    np.testing.assert_array_equal(m.eur_call_put(**kw), m.eur_call_put(R=0.5, **kw))


def test_vanilla_puts_valid_when_negative_moment_explodes():
    # T = 4 is past the explosion of E[S_T^-1.2], so the put contour of R = 1.2
    # does not exist; priced on it, these puts are off by up to 0.04.
    T = 4.0
    assert T > heston_explosion_time(-1.2, **SET_C)
    K = np.array([0.3, 0.5, 0.7, 0.9])
    puts = np.asarray(model(SET_C).eur_call_put(K=K, maturity=T, call=False))
    ref = np.array([reference_put(k, T, SET_C) for k in K])
    np.testing.assert_allclose(puts, ref, atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# Digitals
# ---------------------------------------------------------------------------

def digital_call_from_vanillas(m, K, T, h=1e-4):
    """Digital call = -dC/dK, by central differences of vanilla calls."""
    def call(k):
        return np.asarray(m.eur_call_put(K=k, maturity=T, payoff_selection="as_requested"))
    return -(call(K + h) - call(K - h)) / (2 * h)


@pytest.mark.parametrize("R", ["default", 1.2, 0.5])
@pytest.mark.parametrize("payoff_selection", ["otm", "as_requested"])
@pytest.mark.parametrize("call", [True, False])
def test_digital_matches_strike_derivative(R, payoff_selection, call):
    T = 1.0
    m = model(SET_A)
    K = np.array([0.8, 0.9, 0.97, 1.05, 1.1, 1.2])   # forward is 1.0305
    D = np.exp(-RATE * T)
    dc = digital_call_from_vanillas(m, K, T)
    expected = dc if call else D - dc
    R_kw = {} if R == "default" else dict(R=R)
    got = np.asarray(m.eur_call_put(K=K, maturity=T, digital=True, call=call,
                                    payoff_selection=payoff_selection, **R_kw))
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("payoff_selection", ["otm", "as_requested"])
def test_digital_default_is_half(payoff_selection):
    m = model(SET_A, H=0.1)
    K = np.array([0.8, 0.95, 1.0, 1.1, 1.3])
    kw = dict(K=K, maturity=0.5, time_steps=100, digital=True, payoff_selection=payoff_selection)
    np.testing.assert_array_equal(m.eur_call_put(**kw), m.eur_call_put(R=0.5, **kw))


# SET_C has no mean reversion, so most variance paths are absorbed at zero
# by these maturities.  That piles up probability near the forward, where the
# Fourier truncation error of a digital is large; the strikes below stay
# away from the forward (1.09 at T = 3, 1.13 at T = 4).
_DIGITAL_K = np.array([0.4, 0.5, 0.7, 0.9, 1.6, 2.0])


@pytest.mark.parametrize("payoff_selection", ["otm", "as_requested"])
def test_digital_puts_valid_when_negative_moment_explodes(payoff_selection):
    # T = 3 is past the explosion of E[S_T^-1.2]; priced on R = 1.2, these puts are off by up to 0.19.
    T = 3.0
    assert T > heston_explosion_time(-1.2, **SET_C)
    got = np.asarray(model(SET_C).eur_call_put(
        K=_DIGITAL_K, maturity=T, digital=True, call=False,
        payoff_selection=payoff_selection,
    ))
    ref = np.array([reference_digital_put(k, T, SET_C) for k in _DIGITAL_K])
    np.testing.assert_allclose(got, ref, atol=5e-6, rtol=0)


def test_digital_default_valid_adaptive_panel():
    T = 4.0
    got = np.asarray(model(SET_C).eur_call_put(
        K=_DIGITAL_K, maturity=T, digital=True, call=False,
        integration="adaptive_panel", adaptive_metric="price",
    ))
    ref = np.array([reference_digital_put(k, T, SET_C) for k in _DIGITAL_K])
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)
