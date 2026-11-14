# roughprix: option pricing in the rough Heston model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24191763.svg)](https://doi.org/10.5281/zenodo.24191763)

Documentation: <https://hagerpa.github.io/roughprix/>

<!-- --8<-- [start:intro] -->
European option pricing for rough stochastic volatility models via Fourier inversion. The current implementation focuses on the **rough Heston model**; further models will follow.

Pricing rests on the log-price characteristic function, which is expressed through a Riccati-type Volterra equation. `roughprix` provides several methods for solving that equation, ranging from direct numerical Volterra solvers to analytic approximations, and adds a **Taylor expansion in the Hurst exponent** $H$ (Hager & Kreher 2026, [arXiv:2606.16619](https://arxiv.org/abs/2606.16619)) that makes evaluating the characteristic function across a range of roughness values nearly free once the anchor solve is done. The numerical solvers are implemented in **JAX** for JIT-compiled execution.

<!-- --8<-- [start:methods] -->
### Methods implemented

| Method | `base_solution` | Description |
|--------|-----------------|-------------|
| Adams–PECE | `"direct_riccati"` | Fractional Adams predictor–corrector (El Euch & Rosenbaum 2019); high accuracy, $\mathcal{O}(J^2)$ in time steps |
| Implicit-$\pi$ | `"direct_riccati"` | Product-integration implicit solver; the default away from $H = \tfrac{1}{2}$, more stable than Adams for small $H$ or non-uniform grids |
| Padé $[m/m]$ | `"pade"` | Gatheral closed-form Padé approximant to the Riccati solution; fast, no ODE solve, but much less accurate than root-Padé; kept for comparison |
| Root-Padé $[m/(m+1)]$ | `"root_pade"` | Root-transformed Padé that matches the short-time series and the leading long-time tail (Hager & Kreher 2026, App. D.3); fast, and much more accurate than the plain Padé near $H = 0$ |
| $H$-expansion | any base | Taylor series $\sum_{n=0}^{N} \frac{(\Delta H)^n}{n!} f_n$ around an anchor $H_0$; coefficients $f_n$ obtained by recursive differentiation of the Riccati equation |

With `base_solution="direct_riccati"`, the default `direct_riccati_solver="auto"` uses implicit-$\pi$. Adams is opt-in via `direct_riccati_solver="adams"` and requires a uniform time grid (`time_grid="uniform"`; the default is `"graded"`); it can return NaN for $H \le 0$ or on coarse grids.
<!-- --8<-- [end:methods] -->
<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:install] -->
## Installation

```bash
pip install roughprix
```

or, from a clone of the repository, `pip install -e .`.

Requires Python ≥ 3.10, NumPy, SciPy, JAX, and Numba.
<!-- --8<-- [end:install] -->

<!-- --8<-- [start:model] -->
## Model

The spot $S$ and instantaneous variance $V$ satisfy

$$
\frac{dS_t}{S_t} = \sqrt{V_t}\, dW_t,
$$

$$
V_t = V_0 + \frac{1}{\Gamma(H+\tfrac{1}{2})} \int_0^t (t-s)^{H-\tfrac{1}{2}}
\Bigl[(\theta - \lambda V_s)\, ds + \nu\sqrt{V_s}\, dB_s\Bigr],
$$

$$
d\langle W, B\rangle_t = \rho\, dt,
$$

where $H \in (-\tfrac{1}{2}, \tfrac{1}{2}]$ is the Hurst exponent. $H = \tfrac{1}{2}$ recovers classical Heston.

The coefficients are raw: $\theta$ is the drift constant, and the long-run variance is $\theta/\lambda$. For the usual form $\kappa(\bar v - V_s)$, set $\lambda = \kappa$ and $\theta = \kappa \bar v$.
<!-- --8<-- [end:model] -->

<!-- --8<-- [start:quickstart] -->
## Quick start

```python
import numpy as np
from roughprix import RoughHeston, RoughHestonParam

param = RoughHestonParam(
    lam=0.3,        # mean-reversion speed λ
    theta=0.3*0.02, # drift constant θ; long-run variance θ/λ = 0.02
    nu=0.3,         # vol-of-vol ν
    rho=-0.7,       # spot–vol correlation ρ
    V0=0.02,        # initial variance V₀
    H=0.1,          # Hurst exponent H
)

model = RoughHeston(param, riskfree=0.0)

K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
T = 1.0
```
<!-- --8<-- [end:quickstart] -->

## Pricing
<!-- --8<-- [start:pricing] -->

The characteristic function is evaluated numerically at `param.H` and inverted via Fourier quadrature. The `base_solution` constructor argument selects the Riccati solver:

| `base_solution`   | When to use |
|--------------------|-------------|
| `"auto"` (default) | `"exact_heston"` if $H = \tfrac{1}{2}$, else `"direct_riccati"` with implicit-$\pi$ |
| `"root_pade"`      | $H$ away from $\tfrac{1}{2}$; fast closed-form approximation, less accurate at long maturities and on steep skews |
| `"exact_heston"`   | $H = \tfrac{1}{2}$ exactly (classical Heston closed form); the other bases warn there |
| `"pade"`           | Gatheral $[m/m]$ Padé, for comparison: much less accurate than `"root_pade"` (the accuracy check warns), and order `"44"` can fail near $H = 0$ |
| `"direct_riccati"` | Full numerical Volterra solve; most stable and accurate, slower than the Padé bases |

```python
ivs = model.eur_call_put(
    K=K,
    maturity=T,
    implied_vol=True,
    payoff_selection="otm",   # calls for K >= F, puts for K < F
    integration="panel",
    L=200.0,                  # Fourier truncation
    time_steps=500,           # time-grid nodes for Volterra solve
)
print(ivs)
```

```python
# Direct numerical Riccati solve
benchmark_model = RoughHeston(
    param, riskfree=0.0, base_solution="direct_riccati"
)
ivs_benchmark = benchmark_model.eur_call_put(
    K=K, maturity=T, implied_vol=True, L=200.0
)
```

### Adaptive Fourier truncation

Use `integration="adaptive_panel"` to let the pricer automatically select $L$:

```python
ivs = model.eur_call_put(
    K=K,
    maturity=T,
    implied_vol=True,
    expansion_order=0,
    integration="adaptive_panel",
    adaptive_tol=1e-4,        # stop when max IV change < 1e-4
    adaptive_verbose=True,
)
```
<!-- --8<-- [end:pricing] -->

## Pricing with the H-expansion
<!-- --8<-- [start:expansion] -->

The H-expansion approximates the log characteristic function as a Taylor series in $\Delta H = H - H_0$ around an anchor $H_0$:

$$
\log \varphi(u, T;\, H) \;\approx\; \sum_{n=0}^{N} \frac{(\Delta H)^n}{n!}\, f_n(u, T),
$$

where the coefficients $f_n$ are obtained by differentiating the Riccati–Volterra equation recursively. At order $N = 0$ the result equals the quasi-exact benchmark. Higher orders extend the valid range of $|\Delta H|$.

### Anchor at $H_0 = 0$ (`"root_pade"` base)

Useful when pricing across a range of small $H$ values.

```python
# Anchor at H₀ = 0, target H = 0.1
param_anchor0 = RoughHestonParam(
    lam=0.3, theta=0.3*0.02, nu=0.3, rho=-0.7, V0=0.02,
    H=0.0,              # anchor H₀ = 0
)
model_expansion = RoughHeston(
    param_anchor0, riskfree=0.0, base_solution="root_pade"
)

ivs_N4 = model_expansion.eur_call_put(
    K=K,
    maturity=T,
    implied_vol=True,
    expansion_order=4,  # 4th-order Taylor in H
    H=0.1,              # evaluate at this target H
    L=200.0,
)
```

### Anchor at $H_0 = \tfrac{1}{2}$ (`"exact_heston"` base)

Useful near the classical Heston regime.

```python
param_half = RoughHestonParam(
    lam=0.3, theta=0.3*0.02, nu=0.3, rho=-0.7, V0=0.02,
    H=0.5,              # anchor: classical Heston
)
model_half = RoughHeston(
    param_half, riskfree=0.0, base_solution="exact_heston"
)

ivs_N2 = model_half.eur_call_put(
    K=K,
    maturity=T,
    implied_vol=True,
    expansion_order=2,  # 2nd-order Taylor in H around ½
    H=0.4,              # evaluate at H = 0.4
    L=200.0,
)
```

### Pricing across multiple $H$ values

Pass an array to `H` to price all values simultaneously. The expensive Volterra fields are computed once and shared:

```python
H_values = np.array([0.05, 0.1, 0.15, 0.2])

# Returns shape (N_K, N_H)
ivs_grid = model_expansion.eur_call_put(
    K=K,
    maturity=T,
    implied_vol=True,
    expansion_order=4,
    H=H_values,
    L=200.0,
)
print(ivs_grid.shape)  # (5, 4)
```

### Choosing expansion order

| Order $N$ | Truncation error | Cost |
|-----------|-------|------|
| 0 | $\mathcal{O}(\Delta H)$; the base solution itself at $H = H_0$ | baseline |
| 1 | $\mathcal{O}(\Delta H^2)$ | $+1$ Volterra solve |
| 2 | $\mathcal{O}(\Delta H^3)$ | $+2$ solves |
| 4 | $\mathcal{O}(\Delta H^5)$ | $+4$ solves |

### Accuracy

The error depends on $|\Delta H|$, the maturity, how far out of the money the strikes are, and the model parameters. We compared order 4 from an anchor at $H_0 = 0$ (`"root_pade"` base, library defaults) with a converged direct solve at the target $H$, on 18 parameter sets. Largest implied-vol error over strikes within two standard deviations of the forward, $\lvert\log(K/F)\rvert \le 2\sqrt{V_0 T}$, for the 15 sets with $\rho \ge -0.7$:

| $\lvert\Delta H\rvert$ | 1 day | 1 week | 1 month | 3 months | 1 year | 5 years |
|---|---|---|---|---|---|---|
| 0.05 | 0.6 bp | 0.3 bp | 0.2 bp | 0.2 bp | 0.1 bp | 0.06 bp |
| 0.1 | 30 bp | 3 bp | 0.3 bp | 0.2 bp | 0.1 bp | 0.06 bp |
| 0.2 | fails | 170 bp | 15 bp | 4.5 bp | 0.2 bp | 0.1 bp |

("fails": some prices come out invalid.)

- **Strikes further out are harder at short maturities.** At one week and $\lvert\Delta H\rvert = 0.1$, the largest error is 0.7 bp within one standard deviation, 3 bp within two and 48 bp within three. At one and two years it stays below 0.4 bp out to three standard deviations.
- **Steep skews** ($\rho \le -0.9$) raise the largest error to about 2–10 bp from one month to one year. Most of that is the root-Padé base: with `base_solution="direct_riccati"` it stays below 1 bp for $\lvert\Delta H\rvert \le 0.1$.
- **Targets above the anchor are harder than targets below it** at short maturities: at one day, $H = 0.1$ gives 30 bp and $H = -0.1$ gives 8 bp.
- **At a week or less with $\lvert\Delta H\rvert \gtrsim 0.1$**, price at the target directly (anchor at the target, order 0).

`eur_call_put` checks its own accuracy and warns (`UserWarning`) when the estimated implied-vol error exceeds `accuracy_tol` (default 1 bp) at a strike whose out-of-the-money price is at least $10^{-6} S_0$. The returned values do not change.

- **H-expansion** ($N \ge 1$): it also computes order $N+1$ and takes the change of the implied vols at the lowest and the highest target $H$. On the same sets and maturities (orders 1–3, from $H_0 = 0$ with the implicit-$\pi$ base and from $H_0 = \tfrac12$ with exact Heston; 4158 smile–order cases) the estimate was within 0.76–1.18 times the true error for 80 % of the cases above 1 bp. It missed 21 of the 2247 cases above 1 bp, by at most 1.7 bp, and warned on 10 of the 1911 below.
- **Padé bases** away from $H_0 = \tfrac12$: it compares the two `log_cf_integrand` formulas at the anchor. Both are exact for the exact Riccati solution, and `"rhs"` amplifies the error of an approximate one, so they differ by about the error of `"rhs"`. With `"root_pade"` the default `"solution"` is 3–10 times more accurate than that difference, so there the check uses a third of it. On 756 root-Padé smiles it warned on 25 of the 26 above 1 bp (it missed one at 1.4 bp) and on 11 of the 730 below.

The check adds about 10–35 % run time. `accuracy_check=False` turns it off.
<!-- --8<-- [end:expansion] -->

## API reference
<!-- --8<-- [start:api] -->

### `RoughHestonParam`

| Field | Type | Description |
|-------|------|-------------|
| `lam` | `float` | Mean-reversion speed $\lambda$ |
| `theta` | `float` | Drift constant $\theta$; the long-run variance is $\theta/\lambda$ |
| `nu` | `float` | Vol-of-vol $\nu$ |
| `rho` | `float` | Spot–vol correlation $\rho \in (-1, 1)$ |
| `V0` | `float` | Initial variance $V_0$ |
| `H` | `float` | Hurst exponent $H \in (-\tfrac{1}{2}, \tfrac{1}{2}]$; $H = \tfrac{1}{2}$ is classical Heston |

### `RoughHeston(param, riskfree, base_solution="auto", ...)`

| Argument | Description |
|----------|-------------|
| `param` | `RoughHestonParam` instance |
| `riskfree` | Continuously compounded risk-free rate $r$ |
| `base_solution` | Riccati solver: `"auto"`, `"root_pade"`, `"exact_heston"`, `"pade"`, `"direct_riccati"` |
| `pade_order` | `"22"`, `"33"`, or `"44"` (only for `"pade"`) |
| `root_pade_degree` | Padé numerator degree $m$ (default 9; Hager & Kreher 2026 use 7) |
| `root_pade_tail_match` | Replace the highest short-time Padé equation with the exact large-$x$ tail constraint (default `True`) |
| `log_cf_integrand` | What the log characteristic function integrates: `"solution"` integrates the Riccati solution $h$, $\theta\, I^1 h(T) + V_0\, I^{1-\alpha} h(T)$ (El Euch & Rosenbaum 2019); `"rhs"` integrates the right-hand side, $\int_0^T F(h(s))\,(V_0 + \theta (T-s)^\alpha/\Gamma(1+\alpha))\, ds$, as in Hager & Kreher (2026), Eq. (1.1). Both are exact for the exact $h$. `"solution"` does not amplify the error of an approximate $h$; `"rhs"` restores the target's short-time behaviour, which helps expansions from $H_0 = \tfrac12$. `"auto"` (default) uses `"rhs"` for expansions from $H_0 = \tfrac12$ and `"solution"` otherwise. For the paper's settings see [Reproducing the paper](#reproducing-the-paper) |

### `model.eur_call_put(K, maturity, ...)`

| Argument | Default | Description |
|----------|---------|-------------|
| `K` | — | Strike(s), scalar or array |
| `maturity` | — | Option maturity $T > 0$ |
| `S_0` | `1.0` | Initial spot $S_0$ |
| `call` | `True` | `True` returns calls, `False` puts |
| `digital` | `False` | Cash-or-nothing digitals instead of vanillas |
| `implied_vol` | `False` | Return Black–Scholes implied vols $\sigma_\text{BS}$ (vanillas only) |
| `payoff_selection` | `"otm"` | `"otm"` prices the out-of-the-money option at each strike (calls for $K \ge F$, puts otherwise) and converts to `call` by put–call parity; `"as_requested"` prices the requested option directly |
| `expansion_order` | `0` | Taylor order $N$ in $H$ |
| `H` | `None` | Target $H$ (scalar or array; `None` uses `param.H`) |
| `L` | `None` | Fourier truncation $L$. `None` picks where $\lvert\varphi\rvert$ has decayed to about $10^{-10}$, from the model parameters and maturity |
| `integration` | `"panel"` | `"panel"` (fixed $L$) or `"adaptive_panel"` (auto-select $L$; warns when no two successive $L$ agree) |
| `time_steps` | `400` | Time-grid nodes for the expansion Volterra solve |
| `time_grid` | `"graded"` | `"graded"` clusters nodes near $t = 0$, where the solution is rough; `"uniform"` or `"symmetric_graded"`. Adams needs `"uniform"` |
| `R` | `0.5` | Fourier contour height. $R = 0.5$ prices every strike as a call and gets puts by parity (the Lewis contour for vanillas); it needs only $E[S_T^{1/2}]$, which is always finite. Other values use damped contours at $\pm \lvert R\rvert$, which need $E[S_T^{\pm \lvert R\rvert}] < \infty$. Below $\lvert R\rvert = 1$ vanilla calls get the residue $S_0$ of the payoff's pole at 1 added, as in the Lewis formula (half of it at 1); $R = 0$ runs through the pole at 0 and adds half of its residue. Near a pole the integrand has a peak of width $\lvert R - \text{pole}\rvert$ at $u = 0$: $10^{-2}$ from a pole prices are off by $4 \cdot 10^{-9}$, $10^{-3}$ from it by up to $2 \cdot 10^{-3} S_0$ |
| `m_cheb` | `16` | Chebyshev nodes per Fourier panel |
| `adaptive_tol` | `1e-4` | Max implied-vol change threshold for adaptive convergence (digitals compare prices instead) |
| `accuracy_check` | `True` | Estimate the error of the H-expansion and of the Padé bases and warn when it exceeds `accuracy_tol`; see [Accuracy](#accuracy) |
| `accuracy_tol` | `1e-4` | Warning threshold of the check, in implied vol (1 bp) |
<!-- --8<-- [end:api] -->

<!-- --8<-- [start:roadmap] -->
## Roadmap

The following extensions are planned:

- **Further European payoffs** — spreads, variance swaps.
- **Path-dependent options** — barriers, Asians, and lookbacks.
- **Rough Bergomi model.**
- **American options.**
<!-- --8<-- [end:roadmap] -->

## Reproducing the paper
<!-- --8<-- [start:paper] -->

The numerical results in Section 4 of Hager & Kreher (2026, [arXiv:2606.16619v1](https://arxiv.org/abs/2606.16619v1)) were computed with the code as of June 2026. Version 0.1.0 refines that code and changes its defaults (see the [changelog](https://github.com/hagerpa/roughprix/blob/main/CHANGELOG.md)). Only one of these changes affects the published numbers, and only slightly. Around $H_0 = \tfrac12$ the linearised equations of the expansion have an exponential kernel, which 0.1.0 integrates exactly; the earlier code used the trapezoidal rule. This moves the smiles of Figure 2 with $N \ge 1$ by up to $1.4 \cdot 10^{-5}$ in implied volatility, and it changes Table 1 at $(H, T, N) = (0.2, 0.019, 4)$ from 0.3965 to 0.3966.

To reproduce the published numbers exactly, install commit [`320dfd9`](https://github.com/hagerpa/roughprix/tree/320dfd9792f8f2b32a1e85ef66309440e37a7f35) in a separate environment. It is version 0.0.1 with its import error fixed.

```bash
pip install "git+https://github.com/hagerpa/roughprix@320dfd9792f8f2b32a1e85ef66309440e37a7f35"
```

With the paper's parameters and strikes, these calls reproduce Figures 2 and 3:

```python
import numpy as np
from dataclasses import replace
from roughprix import RoughHeston, RoughHestonParam

param = RoughHestonParam(lam=0.3, theta=0.3*0.02, nu=0.3, rho=-0.7, V0=0.02, H=0.5)
T = 0.083                                              # 0.019, 0.083, 0.25 or 1
K = np.exp(np.sqrt(T) * np.linspace(-1.0, 0.5, 76))   # S_0 = 1, r = 0
pricing = dict(implied_vol=True, integration="adaptive_panel",
               adaptive_L_candidates=[110, 150, 200, 270, 350, 500, 700, 1000])

# Benchmark at each target H: implicit product integration with 2000 steps
bench = RoughHeston(replace(param, H=0.2), 0.0, base_solution="direct_riccati",
                    direct_riccati_solver="implicit_pi")
iv_bench = bench.eur_call_put(K, T, R=0.5, time_steps=2000, time_grid="uniform",
                              adaptive_tol=1.5e-4, **pricing)

# Expansion around H0 = 1/2 (Figure 2)
half = RoughHeston(replace(param, H=0.5), 0.0, base_solution="exact_heston")
iv_half = half.eur_call_put(K, T, H=[0.2, 0.3, 0.4], expansion_order=4, R=0.5,
                            time_steps=500, time_grid="symmetric_graded", **pricing)

# Expansion around H0 = 0 (Figure 3): root-Padé of degree 7 with tail matching
zero = RoughHeston(replace(param, H=0.0), 0.0, base_solution="root_pade",
                   root_pade_degree=7, root_pade_tail_match=True)
iv_zero = zero.eur_call_put(K, T, H=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2], expansion_order=4,
                            R=0.5, time_steps=500, time_grid="symmetric_graded", **pricing)
```

- Tables 1 and 2 price the expansions with `R=1.2` instead of `R=0.5`, and Table 2 also uses `root_pade_tail_match=False`. The benchmark is the same.
- Price all target $H$ of a figure in one call, as above. The adaptive loop then raises the cutoff until the implied vols at all of them have converged.
- Figure 1 uses the root-Padé base of degree 7 without tail matching at both expansion points, $H_0 = 0$ and $H_0 = \tfrac12$:

```python
u = np.linspace(1, 300, 300)
z = 0.5 - 1j * u                                  # R - iu with R = 1/2
fbar = {}
for H0 in (0.0, 0.5):
    model = RoughHeston(replace(param, H=H0), 0.0, base_solution="root_pade",
                        root_pade_degree=7, root_pade_tail_match=False)
    state = model.expansion_state(-1j * z, maturity=1.0, order=12,   # order=20 for R*
                                  time_steps=500, time_grid="graded", return_lists=True)
    # max over the time grid of |d^n/dH^n psi(t; H0, z)|, n = 0, ..., 12
    fbar[H0] = [np.abs(np.asarray(f.values)).max(axis=-1) for f in state.f_list]
```

With commit `320dfd9` these calls reproduce all 64 table entries in all printed digits. They reproduce the smiles of Figures 2 and 3 to $10^{-8}$ in implied volatility, the precision of the published figures, and the data of Figure 1 to $10^{-14}$.

With version 0.1.0, add `log_cf_integrand="rhs"` to each `RoughHeston(...)`. Otherwise 0.1.0 uses another formula for the log characteristic function at anchors $H_0 \neq \tfrac12$, which includes the benchmark. The results then agree with the published numbers to the same precision, apart from the change above. In 0.1.0 many of these calls warn, either from the accuracy check or because the cutoff reaches 1000 without meeting `adaptive_tol` (mostly at $T \le 0.083$). The warnings do not change the values.
<!-- --8<-- [end:paper] -->

## Citation
<!-- --8<-- [start:citation] -->

If you use `roughprix` in your research, please cite the paper

```bibtex
@misc{HagerKreher2026,
  author        = {Hager, Paul P. and Kreher, D{\"o}rte},
  title         = {Expanding the rough {H}eston model in {$H$}},
  year          = {2026},
  eprint        = {2606.16619},
  archivePrefix = {arXiv},
}
```

To cite the software itself, use [`CITATION.cff`](https://github.com/hagerpa/roughprix/blob/main/CITATION.cff) ("Cite this repository" on GitHub); every release is archived on Zenodo with its own DOI.

## References

- El Euch, O. & Rosenbaum, M. (2019). *The characteristic function of rough Heston models.* Mathematical Finance, 29(1), 3–38. [doi:10.1111/mafi.12173](https://doi.org/10.1111/mafi.12173)
- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). *Volatility is rough.* Quantitative Finance, 18(6), 933–949. [doi:10.1080/14697688.2017.1393551](https://doi.org/10.1080/14697688.2017.1393551)
- Diethelm, K., Ford, N. J. & Freed, A. D. (2002). *A predictor-corrector approach for the numerical solution of fractional differential equations.* Nonlinear Dynamics, 29, 3–22. [doi:10.1023/A:1016592219341](https://doi.org/10.1023/A:1016592219341) — Adams–PECE solver.
- Garrappa, R. (2018). *Numerical solution of fractional differential equations: A survey and a software tutorial.* Mathematics, 6(2), 16. [doi:10.3390/math6020016](https://doi.org/10.3390/math6020016) — implicit trapezoidal product-integration rule.
- Gatheral, J. & Radoičić, R. (2019). *Rational approximation of the rough Heston solution.* International Journal of Theoretical and Applied Finance, 22(3), 1950010. [doi:10.1142/S0219024919500109](https://doi.org/10.1142/S0219024919500109)
- Gatheral, J. & Radoičić, R. (2024). *A generalization of the rational rough Heston approximation.* Quantitative Finance, 24(2), 329–335. [doi:10.1080/14697688.2024.2302055](https://doi.org/10.1080/14697688.2024.2302055)
- Hager, P. P. & Kreher, D. (2026). *Expanding the rough Heston model in $H$.* Preprint. [arXiv:2606.16619](https://arxiv.org/abs/2606.16619)
<!-- --8<-- [end:citation] -->