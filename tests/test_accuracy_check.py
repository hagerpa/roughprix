"""The built-in accuracy check of eur_call_put (``accuracy_check``, ``accuracy_tol``).

For ``expansion_order = N >= 1`` the check compares the implied vols of orders N and N+1: the
next term, (H - H0)^(N+1)/(N+1)! f_(N+1), estimates the truncation error of order N, and it
grows with |H - H0|, so the lowest and the highest target H bound it.  For the Pade bases away
from H0 = 1/2 it compares the two ``log_cf_integrand`` formulas at the anchor: both are exact
for the exact h, and "rhs" amplifies the error of an approximate h, so their difference is about
the error of "rhs" (for root-Pade with "solution" a third of it is used).  The check warns when
the estimate exceeds ``accuracy_tol`` (1 bp) and leaves the returned values unchanged.  True
errors quoted below are against implicit-pi at 1600 steps at the target H.
"""
import warnings

import numpy as np
import pytest

from roughprix import RoughHeston, RoughHestonParam

_A = dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02)
_STEEP = dict(_A, rho=-0.95)


def _strikes(T):
    return np.exp(np.array([-2, -1, -0.5, 0, 0.5, 1, 2]) * np.sqrt(_A["V0"] * T))


def _silent(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return f()


# (anchor, target, maturity, order): true errors 6.5, 33 and 23 bp; estimates 6.6, 34 and 18 bp
_UNSAFE = {"0_to_0.15_1w": (0.0, 0.15, 1 / 52, 4), "0_to_0.2_1w": (0.0, 0.2, 1 / 52, 4),
           "half_to_0.3_1w": (0.5, 0.3, 1 / 52, 2)}
# true errors <= 0.3 bp; the last is the README example
_SAFE = {"0_to_0.05_1y": (0.0, 0.05, 1.0, 4), "0_to_0.1_3m": (0.0, 0.1, 0.25, 4),
         "half_to_0.4_1y": (0.5, 0.4, 1.0, 2)}


@pytest.mark.parametrize("case", list(_UNSAFE))
def test_warns_when_the_expansion_is_inaccurate(case):
    H0, H, T, N = _UNSAFE[case]
    m = RoughHeston(RoughHestonParam(**_A, H=H0), 0.0)
    with pytest.warns(UserWarning, match=f"H-expansion, orders {N} and {N + 1}"):
        m.eur_call_put(K=_strikes(T), maturity=T, implied_vol=True, expansion_order=N, H=H)


@pytest.mark.parametrize("case", list(_SAFE))
def test_silent_when_the_expansion_is_accurate(case):
    H0, H, T, N = _SAFE[case]
    m = RoughHeston(RoughHestonParam(**_A, H=H0), 0.0)
    _silent(lambda: m.eur_call_put(K=_strikes(T), maturity=T, implied_vol=True, expansion_order=N, H=H))


def test_warns_on_invalid_implied_vols():
    """0 -> 0.2 at one day, order 4: the +2 sd vol is NaN; the main pass's NaN must not hide it."""
    m = RoughHeston(RoughHestonParam(**_A, H=0.0), 0.0)
    with pytest.warns(UserWarning, match="estimated error unbounded: non-finite implied vols"):
        iv = m.eur_call_put(K=_strikes(1 / 365), maturity=1 / 365, implied_vol=True, expansion_order=4, H=0.2)
    assert np.isnan(iv[-1])


def test_multi_H_warns_for_an_inaccurate_extreme_target():
    m = RoughHeston(RoughHestonParam(**_A, H=0.0), 0.0)
    with pytest.warns(UserWarning, match="H-expansion"):
        m.eur_call_put(K=_strikes(1 / 52), maturity=1 / 52, expansion_order=4, H=[0.02, 0.05, 0.08, 0.2])


def test_multi_H_silent_when_every_target_is_accurate():
    m = RoughHeston(RoughHestonParam(**_A, H=0.0), 0.0)
    _silent(lambda: m.eur_call_put(K=_strikes(1.0), maturity=1.0, expansion_order=4, H=[-0.05, 0.0, 0.02, 0.05]))


def test_root_pade_steep_skew_warns():
    """rho = -0.95, H = -0.4, one month: root-Pade is 18 bp off; estimate 12 bp (a third of 37 bp)."""
    m = RoughHeston(RoughHestonParam(**_STEEP, H=-0.4), 0.0, base_solution="root_pade")
    with pytest.warns(UserWarning, match="root_pade base at H = -0.4"):
        m.eur_call_put(K=_strikes(1 / 12), maturity=1 / 12, implied_vol=True)


def test_root_pade_checked_at_the_anchor_of_an_expansion():
    m = RoughHeston(RoughHestonParam(**_STEEP, H=-0.4), 0.0, base_solution="root_pade")
    with pytest.warns(UserWarning, match="root_pade base at H = -0.4"):
        m.eur_call_put(K=_strikes(1 / 12), maturity=1 / 12, expansion_order=1, H=-0.39)


def test_root_pade_estimate_depends_on_the_formula():
    """nu = 1.2, H = 0, one week: the formulas differ by 2.6 bp; "solution" is 0.2 bp off, "rhs" 2.4 bp."""
    p = RoughHestonParam(**dict(_A, nu=1.2), H=0.0)
    kw = dict(K=_strikes(1 / 52), maturity=1 / 52, implied_vol=True)
    _silent(lambda: RoughHeston(p, 0.0, base_solution="root_pade").eur_call_put(**kw))
    with pytest.warns(UserWarning, match="root_pade base at H = 0, from the two log_cf_integrand formulas"):
        RoughHeston(p, 0.0, base_solution="root_pade", log_cf_integrand="rhs").eur_call_put(**kw)


@pytest.mark.parametrize("H, T", [(0.1, 1.0), (0.0, 1 / 12)])
def test_root_pade_silent_when_accurate(H, T):
    m = RoughHeston(RoughHestonParam(**_A, H=H), 0.0, base_solution="root_pade")
    _silent(lambda: m.eur_call_put(K=_strikes(T), maturity=T, implied_vol=True))


def test_check_off_and_tolerance():
    m = RoughHeston(RoughHestonParam(**_A, H=0.0), 0.0)
    kw = dict(K=_strikes(1 / 52), maturity=1 / 52, implied_vol=True, expansion_order=4, H=0.15)
    _silent(lambda: m.eur_call_put(accuracy_check=False, **kw))
    _silent(lambda: m.eur_call_put(accuracy_tol=1e-3, **kw))   # estimate 6.6 bp < 10 bp


def test_digitals_are_not_checked():
    m = RoughHeston(RoughHestonParam(**_A, H=0.0), 0.0)
    _silent(lambda: m.eur_call_put(K=_strikes(1 / 52), maturity=1 / 52, digital=True, expansion_order=4, H=0.2))


_RESULT_CASES = {
    "expansion_iv": (_A, 0.0, {}, dict(expansion_order=4, H=0.15, maturity=1 / 52, implied_vol=True)),
    "multi_H_prices": (_A, 0.0, {}, dict(expansion_order=2, H=[0.02, 0.05, 0.1, 0.2], maturity=1 / 52)),
    "multi_H_damped": (_A, 0.0, {}, dict(expansion_order=2, H=[0.02, 0.05, 0.1], maturity=0.5, R=1.5,
                                         implied_vol=True)),
    "from_half": (_A, 0.5, {}, dict(expansion_order=2, H=0.3, maturity=1 / 52, implied_vol=True)),
    "root_pade": (_STEEP, -0.4, {"base_solution": "root_pade"}, dict(maturity=1 / 12, implied_vol=True)),
    "root_pade_adaptive": (_A, 0.0, {"base_solution": "root_pade"},
                           dict(expansion_order=3, H=[0.05, 0.1, 0.12], maturity=0.5, integration="adaptive_panel")),
}


@pytest.mark.parametrize("case", list(_RESULT_CASES))
def test_check_leaves_results_unchanged(case):
    p, H0, model_kw, kw = _RESULT_CASES[case]
    m = RoughHeston(RoughHestonParam(**p, H=H0), 0.0, **model_kw)
    K = _strikes(kw["maturity"])
    off = np.asarray(m.eur_call_put(K=K, accuracy_check=False, **kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = np.asarray(m.eur_call_put(K=K, **kw))
    np.testing.assert_array_equal(on, off)
    assert m._field_cache is None and m._extra_order == 0 and m.log_cf_integrand == "auto"


@pytest.mark.parametrize("kw", [dict(H=list(np.linspace(-0.1, 0.15, 7))), dict(H=[-0.05, 0.02, 0.1], R=1.5),
                                dict(H=0.1, R=1.5)])
def test_check_reuses_the_expansion_fields(kw):
    """The check prices from the main pass's cached fields, so it solves for no further anchor fields."""
    m = RoughHeston(RoughHestonParam(**_A, H=0.0), 0.0)
    builds = []
    build = m._build_anchor_ops
    m._build_anchor_ops = lambda *a, **k: (builds.append(1), build(*a, **k))[1]
    m.eur_call_put(K=_strikes(0.5), maturity=0.5, expansion_order=2, accuracy_check=False, **kw)
    n_off = len(builds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.eur_call_put(K=_strikes(0.5), maturity=0.5, expansion_order=2, **kw)
    assert len(builds) == 2 * n_off


def test_state_is_reset_after_an_error():
    m = RoughHeston(RoughHestonParam(**_A, H=0.0), 0.0)
    with pytest.raises(ValueError, match="panels=None"):
        m.eur_call_put(K=_strikes(0.5), maturity=0.5, expansion_order=1, H=0.1, integration="adaptive_panel",
                       panels=[0.0, 50.0, 100.0])
    assert m._field_cache is None and m._extra_order == 0
