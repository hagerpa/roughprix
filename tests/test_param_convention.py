"""The drift convention of RoughHestonParam, as stated in the README.

The variance drift is theta - lam*V: theta is a raw drift constant and the
long-run variance is theta/lam.  At H = 1/2 the model is classical Heston
with dV = kappa*(vbar - V)dt + nu*sqrt(V)dB, where kappa = lam and
vbar = theta/lam.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from roughprix import RoughHeston, RoughHestonParam

KAPPA, VBAR = 0.3, 0.02
PARAM = dict(lam=KAPPA, theta=KAPPA * VBAR, nu=0.3, rho=-0.7, V0=0.04)
T = 2.0
K = np.array([0.7, 0.85, 1.0, 1.15, 1.3])


def classical_heston_cf(z, T, kappa, vbar, nu, rho, V0):
    """E[exp(i z log(S_T / S_0))] for dV = kappa*(vbar - V)dt + nu*sqrt(V)dB, zero rate."""
    iz = 1j * z
    b = kappa - rho * nu * iz
    d = np.sqrt(b * b + nu ** 2 * (iz + z * z))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / nu ** 2 * (1 - e) / (1 - g * e)
    C = kappa * vbar / nu ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return np.exp(C + D * V0)


def reference_calls(K, T, **heston):
    """Call prices with S_0 = 1 from the Lewis formula, via quad."""
    def call(k):
        def integrand(u):
            return (np.exp(-1j * u * np.log(k)) * classical_heston_cf(u - 0.5j, T, **heston)).real / (u * u + 0.25)

        return 1.0 - np.sqrt(k) / np.pi * integrate.quad(
            integrand, 0, np.inf, limit=500, epsabs=1e-13, epsrel=1e-12)[0]

    return np.array([call(k) for k in K])


@pytest.mark.filterwarnings("ignore:base='direct_riccati' at anchor H = 1/2")
@pytest.mark.parametrize("base", ["exact_heston", "direct_riccati"])
def test_theta_is_raw_drift_constant(base):
    model = RoughHeston(RoughHestonParam(**PARAM, H=0.5), riskfree=0.0, base_solution=base)
    prices = model.eur_call_put(K=K, maturity=T, time_steps=400)

    raw = reference_calls(K, T, kappa=PARAM["lam"], vbar=PARAM["theta"] / PARAM["lam"],
                          nu=PARAM["nu"], rho=PARAM["rho"], V0=PARAM["V0"])
    np.testing.assert_allclose(prices, raw, rtol=0, atol=1e-6)

    # Reading theta as the long-run variance gives clearly different prices, so the test tells the two apart.
    misread = reference_calls(K, T, kappa=PARAM["lam"], vbar=PARAM["theta"],
                              nu=PARAM["nu"], rho=PARAM["rho"], V0=PARAM["V0"])
    assert np.max(np.abs(prices - misread)) > 1e-3


def test_exact_heston_prices_in_closed_form():
    """At H = 1/2 the exact-Heston log CF is in closed form, so 10 time steps reach the reference."""
    model = RoughHeston(RoughHestonParam(**PARAM, H=0.5), riskfree=0.0, base_solution="exact_heston")
    prices = model.eur_call_put(K=K, maturity=T, time_steps=10)

    raw = reference_calls(K, T, kappa=PARAM["lam"], vbar=PARAM["theta"] / PARAM["lam"],
                          nu=PARAM["nu"], rho=PARAM["rho"], V0=PARAM["V0"])
    np.testing.assert_allclose(prices, raw, rtol=0, atol=1e-10)
