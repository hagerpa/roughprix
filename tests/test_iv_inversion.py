"""The Black-Scholes implied-vol inversion in fourier.py.

The inversion bisects on the bracket [1e-10, 10].  It returns NaN for
non-finite prices, for prices outside the no-arbitrage bounds and for vols
outside the bracket, and stops after at most ``max_iter`` halvings.  A NaN
compares false against everything and would freeze its bracket; an autouse
fixture caps the number of Black-Scholes evaluations, so a loop that does not
terminate fails these tests instead of hanging the suite.
"""
from __future__ import annotations

import numpy as np
import pytest

import roughprix.fourier as fourier
from roughprix.fourier import _bs_price, _iv_bisect, _iv_eur, otm_prices_to_ivs

CALL_BUDGET = 1000

# The default bracket [1e-10, 10] needs 37 halvings to reach tol = 1e-10.
BISECTION_STEPS = 37

S0 = 1.0
Z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])          # strikes in standard deviations
SIGMA = np.array([0.35, 0.25, 0.20, 0.18, 0.22])   # a skewed smile


def strikes(T):
    return S0 * np.exp(Z * SIGMA * np.sqrt(T))


@pytest.fixture(autouse=True)
def bs_calls(monkeypatch):
    """Count Black-Scholes evaluations made by _iv_eur and stop a runaway loop."""
    calls = {"n": 0}
    original = fourier._bs_price

    def counted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > CALL_BUDGET:
            raise RuntimeError("implied-vol bisection did not terminate")
        return original(*args, **kwargs)

    monkeypatch.setattr(fourier, "_bs_price", counted)
    return calls


@pytest.mark.parametrize("payoff", ["call", "put"])
@pytest.mark.parametrize("T, r", [(1 / 52, 0.0), (1.0, 0.03), (5.0, -0.01)])
def test_round_trip(payoff, T, r):
    K = strikes(T)
    price = _bs_price(S0, K, SIGMA, T, r=r, call=payoff == "call")
    iv = _iv_eur(S_0=S0, K=K, T=T, price=price, payoff=payoff, r=r)
    np.testing.assert_allclose(iv, SIGMA, atol=1e-9, rtol=0)


def test_nan_price_returns_nan_without_stalling(bs_calls):
    iv = _iv_eur(S_0=S0, K=np.array([1.0]), T=1.0, price=np.array([np.nan]), payoff="call")
    assert np.isnan(iv).all()
    # The NaN entry must not keep the loop running up to its iteration cap.
    assert bs_calls["n"] <= BISECTION_STEPS + 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_price_is_nan_and_leaves_others_intact(bad):
    K = strikes(1.0)
    price = _bs_price(S0, K, SIGMA, 1.0)
    price[1] = bad
    iv = _iv_eur(S_0=S0, K=K, T=1.0, price=price, payoff="call")
    assert np.isnan(iv[1])
    keep = np.arange(len(K)) != 1
    np.testing.assert_allclose(iv[keep], SIGMA[keep], atol=1e-9, rtol=0)


K0, T0, R0 = 0.9, 1.0, 0.02
D0 = np.exp(-R0 * T0)


@pytest.mark.parametrize("payoff, price", [
    ("call", S0 + 1e-3),              # above the call upper bound S_0
    ("call", S0 - K0 * D0 - 1e-3),    # below the call intrinsic value
    ("put", K0 * D0 + 1e-3),          # above the put upper bound K e^{-rT}
    ("put", -1e-3),                   # negative put price
])
def test_price_outside_no_arbitrage_bounds_is_nan(payoff, price):
    iv = _iv_eur(S_0=S0, K=np.array([K0]), T=T0, price=np.array([price]),
                 payoff=payoff, r=R0)
    assert np.isnan(iv).all()


def test_vol_beyond_bracket_is_nan_not_clipped():
    # A valid price whose implied vol exceeds the bracket's upper end (10) is
    # not resolved: NaN, not the clipped value 10.
    K = np.array([1.0])
    price = 0.5 * (_bs_price(S0, K, 10.0, 1.0) + S0)
    iv = _iv_eur(S_0=S0, K=K, T=1.0, price=price, payoff="call")
    assert np.isnan(iv).all()


def test_bisection_terminates_when_price_fun_is_nan():
    calls = {"n": 0}

    def price_fun(sigma):
        calls["n"] += 1
        if calls["n"] > CALL_BUDGET:
            raise RuntimeError("implied-vol bisection did not terminate")
        return np.full(3, np.nan)

    iv = _iv_bisect(price_fun, np.array([0.1, 0.2, 0.3]), max_iter=50)
    assert np.isnan(iv).all()
    assert calls["n"] <= 50


def test_vector_maturities_with_nan():
    T = np.array([0.25, 1.0])
    K = strikes(1.0)
    price = _bs_price(S0, K, SIGMA, T[:, None])
    price[0, 2] = np.nan
    iv = _iv_eur(S_0=S0, K=K, T=T, price=price, payoff="call")
    assert iv.shape == (2, len(K))
    assert np.isnan(iv[0, 2])
    expected = np.broadcast_to(SIGMA, iv.shape)
    finite = ~np.isnan(price)
    np.testing.assert_allclose(iv[finite], expected[finite], atol=1e-9, rtol=0)


def test_otm_prices_to_ivs_with_nan():
    K = strikes(1.0)
    call_mask = K >= S0
    put_mask = ~call_mask
    price = np.where(call_mask,
                     _bs_price(S0, K, SIGMA, 1.0, call=True),
                     _bs_price(S0, K, SIGMA, 1.0, call=False))
    price[[0, 3]] = np.nan   # one put and one call
    iv = otm_prices_to_ivs(price, K, 1.0, S0, call_mask, put_mask)
    assert np.isnan(iv[[0, 3]]).all()
    keep = np.ones(len(K), dtype=bool)
    keep[[0, 3]] = False
    np.testing.assert_allclose(iv[keep], SIGMA[keep], atol=1e-9, rtol=0)
