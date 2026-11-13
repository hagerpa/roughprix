"""Default prices against an independent reference, across parameter regimes.

tests/data/reference_smiles.json holds smiles from reference_pricer.py, which shares no code
with roughprix: implicit product integration of the Riccati equation on a 2000-step graded mesh
(the classical Heston closed form at H = 1/2) and the Lewis formula on Gauss-Legendre panels out
to where the integrand falls below 1e-17.  `python tests/reference_pricer.py` recomputes the
file.  The smiles cover sets A, B and C (C: steep skew, no mean reversion), H from 0.5 to -0.4,
maturities from 1 week to 5 years and strikes out to three standard deviations.  Per strike the
file carries two error estimates of the reference vol: from halving the time steps, and from
refining the Fourier nodes with the cutoff 1.5 times as far.

Vols are compared where the out-of-the-money price is at least 1e-6, as in the accuracy check;
below that, rounding dominates the vols of both pricers.  Where they are compared, both
estimates stay below 0.002 bp, and the defaults match to 0.003 bp, well inside the tolerance
of 0.02 bp.  The default run prices 12 of the 72 smiles, one per set and maturity; --run-slow
prices all, and checks that reference_pricer.py reproduces the stored prices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import reference_pricer
from roughprix import RoughHeston, RoughHestonParam

_DATA = json.loads((Path(__file__).parent / "data" / "reference_smiles.json").read_text())
_MATURITIES = {1 / 52: "1w", 1 / 12: "1m", 1.0: "1y", 5.0: "5y"}
_SMILES = {f"{s['set']}_H{s['H']:+.1f}_{_MATURITIES[s['T']]}": s for s in _DATA["smiles"]}
_DEFAULT_RUN = ["A_H-0.2_1w", "A_H+0.3_1m", "A_H+0.1_1y", "A_H-0.4_5y",
                "B_H+0.1_1w", "B_H-0.4_1m", "B_H+0.5_1y", "B_H+0.3_5y",
                "C_H-0.4_1w", "C_H+0.0_1m", "C_H-0.2_1y", "C_H+0.5_5y"]
PRICE_FLOOR = 1e-6
TOL = 2e-6   # implied vol, 0.02 bp


def _priced(s):
    return np.array(s["price"]) >= PRICE_FLOOR


@pytest.mark.parametrize("case", [pytest.param(c, marks=() if c in _DEFAULT_RUN else pytest.mark.slow)
                                  for c in _SMILES])
def test_default_smile(case):
    s = _SMILES[case]
    model = RoughHeston(RoughHestonParam(**_DATA["params"][s["set"]], H=s["H"]), riskfree=0.0)
    iv = model.eur_call_put(K=np.array(s["K"]), maturity=s["T"], implied_vol=True)
    np.testing.assert_allclose(iv[_priced(s)], np.array(s["iv"])[_priced(s)], rtol=0, atol=TOL)


def test_reference_converged():
    """Both error estimates of the reference stay below a tenth of the tolerance."""
    for s in _DATA["smiles"]:
        assert np.max(np.maximum(s["err_time"], s["err_fourier"])[_priced(s)]) < TOL / 10


@pytest.mark.slow
@pytest.mark.parametrize("case", ["C_H-0.4_1w", "B_H+0.5_1y"])
def test_reference_reproducible(case):
    """reference_pricer.py recomputes the stored prices, on the rough and on the classical path."""
    s = _SMILES[case]
    p = reference_pricer.PARAMS[s["set"]]
    K = np.array(s["K"])
    price = reference_pricer.otm_prices(p, s["H"], s["T"], K,
                                        reference_pricer.fourier_cutoff(p, s["H"], s["T"]))
    np.testing.assert_allclose(price, s["price"], rtol=1e-10, atol=1e-15)
