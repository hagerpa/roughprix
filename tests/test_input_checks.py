"""eur_call_put on every contour, spot and integration mode.

- ``payoff_selection`` is validated on every contour.
- The Fourier integral on a damped contour at height R is the price when R lies in the
  payoff's strip: R > 1 for vanilla calls, R > 0 for digital calls, R < 0 for puts.  Each
  pole of the payoff transform between R and that strip adds its residue, half of it when
  the contour runs through the pole: S_0 at R = 1 (vanillas), K e^{-rT} (vanillas) or
  e^{-rT} (digitals) at R = 0.  Every R therefore gives the default prices.
- The payoff transform is in log K and the characteristic function is that of
  log(S_T / S_0), so a damped contour pairs them through S_0: vanilla prices scale as
  C(S_0, K) = S_0 C(1, K / S_0), digitals not at all.
- At R = 0 the call and put contours, 0.0 and -0.0, are one contour, also in the adaptive
  panel.
- Digital prices have no vanilla Black-Scholes implied vol: ``implied_vol=True`` raises,
  and the adaptive panel compares digital prices, not vols.
"""
import functools

import numpy as np
import pytest

from roughprix import RoughHeston, RoughHestonParam

_A = dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=0.1)
_T = 0.5
_K = np.exp(np.array([-2, -1, 0, 1, 2]) * np.sqrt(_A["V0"] * _T))   # F = 1 at zero rate


def _model():
    return RoughHeston(RoughHestonParam(**_A), 0.0)


@pytest.mark.parametrize("R, digital", [(0.5, False), (0.5, True), (1.5, False), (1.5, True)])
@pytest.mark.parametrize("integration", ["panel", "adaptive_panel"])
def test_payoff_selection_is_checked(R, digital, integration):
    with pytest.raises(ValueError, match="payoff_selection"):
        _model().eur_call_put(K=_K, maturity=_T, R=R, digital=digital, integration=integration,
                              payoff_selection="itm")


_SELECTIONS = {"otm": dict(), "calls": dict(payoff_selection="as_requested"),
               "puts": dict(payoff_selection="as_requested", call=False)}


@functools.lru_cache
def _default_contour(digital, selection):
    return _model().eur_call_put(K=_K, maturity=_T, digital=digital, **_SELECTIONS[selection])


@pytest.mark.parametrize("R", [0.0, 0.3, 0.95, 1.0, 1.05, -0.7, -1.5])
@pytest.mark.parametrize("digital", [False, True], ids=["vanilla", "digital"])
@pytest.mark.parametrize("selection", list(_SELECTIONS))
def test_every_contour_gives_the_default_prices(R, digital, selection):
    """Calls sit on +|R|, puts on -|R|, and the residues of the poles in between are added."""
    got = _model().eur_call_put(K=_K, maturity=_T, digital=digital, R=R, **_SELECTIONS[selection])
    np.testing.assert_allclose(got, _default_contour(digital, selection), atol=1e-12, rtol=0)


@pytest.mark.parametrize("R", [0.5, 0.3, 1.5])
@pytest.mark.parametrize("digital", [False, True], ids=["vanilla", "digital"])
@pytest.mark.parametrize("selection", list(_SELECTIONS))
def test_prices_scale_with_the_spot(R, digital, selection):
    """C(S_0, K) = S_0 C(1, K / S_0); digitals do not scale."""
    m = RoughHeston(RoughHestonParam(**_A), 0.03)
    kw = dict(maturity=_T, digital=digital, R=R, **_SELECTIONS[selection])
    got = m.eur_call_put(K=2.0 * _K, S_0=2.0, **kw)
    np.testing.assert_allclose(got, (1.0 if digital else 2.0) * m.eur_call_put(K=_K, **kw), atol=1e-12, rtol=0)


@pytest.mark.parametrize("expansion", [dict(), dict(expansion_order=1, H=0.12)], ids=["order0", "order1"])
@pytest.mark.parametrize("digital", [False, True], ids=["vanilla", "digital"])
def test_adaptive_panel_takes_R_zero(digital, expansion):
    """Strikes on both sides of the forward: calls on 0.0 and puts on -0.0 are one contour."""
    kw = dict(K=_K, maturity=_T, digital=digital, R=0.0, **expansion)
    got = _model().eur_call_put(integration="adaptive_panel", **kw)
    np.testing.assert_allclose(got, _model().eur_call_put(**kw), atol=1e-12, rtol=0)


@pytest.mark.parametrize("integration", ["panel", "adaptive_panel"])
def test_digital_implied_vols_raise(integration):
    with pytest.raises(ValueError, match="implied_vol=True is for vanillas"):
        _model().eur_call_put(K=_K, maturity=_T, digital=True, implied_vol=True, integration=integration)


def test_adaptive_digitals_converge_on_prices(capsys):
    """At one week the first two candidate L already agree on the digital prices: one comparison."""
    K = np.exp(np.linspace(-3, 3, 9) * np.sqrt(_A["V0"] / 52))
    m = _model()
    got = m.eur_call_put(K=K, maturity=1 / 52, digital=True, integration="adaptive_panel", adaptive_verbose=True)
    steps = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[adaptive_panel] iter=")]
    assert len(steps) == 2
    np.testing.assert_allclose(got, m.eur_call_put(K=K, maturity=1 / 52, digital=True, L=200000.0),
                               atol=1e-12, rtol=0)
