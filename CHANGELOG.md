# Changelog

## Unreleased

- Documentation site, https://hagerpa.github.io/roughprix/: guides, API reference, and accuracy and speed against the reference smiles.
- README: the Zenodo DOI; the description of `R` in the `eur_call_put` table is shown in full on GitHub and PyPI.

## 0.1.0

First release on PyPI. Most default prices change. The test suite holds 72 reference smiles from a separate pricer that shares no code with the package (implicit product integration with 2000 steps and the Lewis formula; $H$ from $-0.4$ to $\tfrac12$, one week to five years, strikes out to three standard deviations). Where their out-of-the-money prices are at least $10^{-6} S_0$, implied vols at the 0.0.1 default settings are off by more than 1 bp in 39 smiles and by more than 100 bp, or NaN, in 10; at the 0.1.0 defaults by at most 0.003 bp.

### Changed defaults (prices change)

- `base_solution="auto"` uses implicit-π (`"direct_riccati"`) for anchors $H_0 \neq \tfrac12$ instead of root-Padé; exact Heston at $H_0 = \tfrac12$ as before. `direct_riccati_solver="auto"` is always implicit-π (was Adams for $H_0 \ge 0$ on uniform grids).
- `time_grid="graded"` and `time_steps=400` in `eur_call_put`, `log_charfun`, `charfun` and `expansion_state` (were `"uniform"`, with 1000 steps in `eur_call_put` and 100 elsewhere). `direct_riccati_solver="adams"` therefore raises `ValueError` unless `time_grid="uniform"` is passed, and it can need more steps (NaN at 400, not at 1000, for the README parameters).
- `R=0.5` for every payoff (was `1.2`): all strikes are priced as calls, vanillas on the Lewis contour, and puts follow by parity. This needs only $E[S_T^{1/2}]$, which is finite. With `R=1.2` the puts need $E[S_T^{-1.2}]$, which is infinite beyond a finite maturity on steep skews: for `lam=0`, `nu=0.5`, `rho=-0.9` and $H = \tfrac12$ beyond $T = 2.78$; there the 0.0.1 put prices at $T = 4$ were off by up to $0.037\,S_0$ (strikes from $0.3\,S_0$ to $0.9\,S_0$).
- `L=None` takes the first panel endpoint at which $|\varphi|$ has decayed below about $10^{-10}$, estimated from the model parameters, the maturity, every target $H$ and the anchor, up to 256000; it no longer depends on the strikes (was $160\,(1 + 0.75 \max|\log(K/F)|)/\sqrt{T}$, rounded up to 150, 250, 350, 500, 700 or 1000 and capped at 1000). `integration="adaptive_panel"` starts at this `L` (0.0.1 started at 110 and, with the default `adaptive_max_iter=8`, ended at 1000).
- Root-Padé: `root_pade_degree=9` and `root_pade_tail_match=True` (were 7 and `False`). With `base_solution="root_pade"` and the other 0.1.0 defaults, implied vols are within 3.2 bp of the 60 reference smiles with $H < \tfrac12$; with all 0.0.1 settings they were off by up to 1352 bp ($H = -0.4$, $T = 5$, `rho=-0.9`).
- The log characteristic function integrates the Riccati solution $h$ instead of $F(h)$, except for expansions from $H_0 = \tfrac12$ (`log_cf_integrand`, see Added). This includes `log_charfun_from_field`.
- For anchors $H_0 \neq \tfrac12$, `base_solution="root_pade"`, `root_pade_degree=7`, `root_pade_tail_match=False`, `log_cf_integrand="rhs"`, `R=1.2`, `time_grid="uniform"`, `time_steps=1000` and the 0.0.1 `L` reproduce the 0.0.1 prices up to rounding, except where an entry under Fixed applies. The README section "Reproducing the paper" shows how to reproduce the paper's numbers: exactly with an earlier commit, or with 0.1.0 up to one of the fixes.

### Fixed

- `eur_call_put` raised `ModuleNotFoundError`: 0.0.1 shipped the module `roughprix.heston.base_solution` as `roughprix.heston.expansion_base`.
- Implied vols: a NaN price made the bisection loop forever, and prices above the no-arbitrage bound or with vols above 1000 % returned 10. All of these now give NaN.
- Damped contours (vanillas at `R` other than 0.5, and all digitals; in 0.0.1 this included the default `R=1.2`) priced as if `S_0=1` and lacked the residues of the payoff poles they pass: vanilla calls with $|R| < 1$ lacked $S_0$ ($S_0/2$ at $|R| = 1$), and at `R=0` half the residue of the pole at 0 was missing as well: $-K e^{-rT}/2$ for vanilla calls, $K e^{-rT}/2$ for vanilla puts and $e^{-rT}/2$ for digitals.
- Digitals: `payoff_selection="otm"` (the default) converted between digital calls and puts with the vanilla parity instead of $C + P = e^{-rT}$, so digital calls below the forward and digital puts above it were wrong. `implied_vol=True` now raises `ValueError` for digitals (it inverted digital prices as vanilla prices), and the adaptive panel compares digital prices instead of implied vols.
- Exact Heston at $H = \tfrac12$: the log characteristic function is in closed form, so prices no longer depend on `time_steps` (0.0.1 integrated it on the time grid, with price errors of $10^{-8}\,S_0$ to $10^{-7}\,S_0$ at its default 1000 steps for maturities of one and two years); $\varphi(0)$ was NaN for `lam=0`.
- Padé bases: `pade_order="44"` uses the nonlinear long-time coefficients of Gatheral's [4/4] approximant (0.0.1 used their linearisation); at $H = 0$, where one of them has a pole, it returns the limit $H \to 0$. Its values change except at $H = \tfrac12$. Padé and root-Padé return the exact $h = 0$ at $u = 0$ and $u = -i$ (0.0.1: NaN, and `LinAlgError` for `"44"`).
- Large `L`: the H-expansion from the exact-Heston anchor integrates the exponential kernel of its linearised equation exactly and stays finite (0.0.1 returned NaN once $\nu\sqrt{1-\rho^2}\,L\,T$ exceeded about 700, e.g. `L=4000` at $T = 1$ for the README parameters). `L` above 2000 is split into panels up to 256000 (0.0.1: one panel from 2000 to `L`).
- `payoff_selection` is checked on the Lewis contour too. The adaptive panel prices `R=0` with strikes on both sides of the forward (raised `TypeError`) and raises `ValueError` when there is no candidate `L` (returned `None`).
- Documentation: `RoughHestonParam` has the variance drift $\theta - \lambda V$, so `theta` is a raw drift constant and the long-run variance is `theta/lam`, as computed; 0.0.1 documented $\lambda(\theta - V)$.

### Added

- `log_cf_integrand` in `RoughHeston`: `"solution"` integrates $h$, $\theta\, I^1 h(T) + V_0\, I^{1-\alpha} h(T)$ (El Euch & Rosenbaum 2019); `"rhs"` integrates $F(h)$ against $V_0 + \theta (T-s)^\alpha/\Gamma(1+\alpha)$, Eq. (1.1) of Hager & Kreher (2026, arXiv:2606.16619v1), as 0.0.1 did. Both are exact for the exact $h$. `"rhs"` amplifies the error of an approximate $h$ but restores the short-time behaviour $t^\alpha$ of the target $H$, which a truncated H-expansion only approximates. `"auto"` (default) uses `"rhs"` for expansions from $H_0 = \tfrac12$ and `"solution"` otherwise.
- `accuracy_check=True` and `accuracy_tol=1e-4` in `eur_call_put`: for vanillas, estimates the implied-vol error of the H-expansion (from the next order) and of the Padé bases (from the two `log_cf_integrand` formulas) and warns (`UserWarning`) above `accuracy_tol`; returned values do not change.
- Digitals on `R=0.5` (0.0.1 raised). Warnings for an approximate base at $H = \tfrac12$ and for an adaptive panel that does not converge.
- `roughprix.fourier.damped_contour_residues`, `measure_weights_from_phi(..., S_0=...)`, `HestonExpansionState.target_H` and `.anchor_H`, `AdamsFractionalRiccatiSolver(..., F_args=...)`.

### Removed

- The `tail_match_alpha_tol` keyword of `rough_heston_root_pade_f0`: the tail match now applies at every $H$, including $\tfrac12$. Its `degree` now defaults to 9.
- `Integrate.weights_matrix` of a definite fractional integral (`kernel="frac"` with `stop` set), which now stores only the row it applies; it raises `ValueError`.

### Performance

- The time-stepping scans compile once per array shape, not on every call. Implicit-π and the linear Volterra solves of the H-expansion run in blocks of 8 steps, with the history of earlier blocks as one matrix product; the dense terms of each kernel operator of the expansion are summed into one matrix; Gauss–Legendre nodes are cached.
- With the default settings, a call with nine strikes takes about 40 to 90 ms on an Apple M3. The first call in a session compiles.

### Packaging

- Published on PyPI: `pip install roughprix`. A published GitHub release runs the tests on Python 3.10–3.13 and then uploads the build.
- Each GitHub release is archived on Zenodo with its own DOI.
- `LICENSE` is the canonical Apache License 2.0 text, with the copyright notice in `NOTICE`; new `CITATION.cff`.
- The build requires `setuptools>=77` and no longer `wheel`. `pytest` collects only `tests/`.

## 0.0.1

First release on GitHub.
