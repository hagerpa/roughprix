# Accuracy and speed

## Accuracy

The tables compare implied volatilities with 72 reference smiles computed by
[`tests/reference_pricer.py`](https://github.com/hagerpa/roughprix/blob/main/tests/reference_pricer.py),
a separate implementation that shares no code with roughprix. It solves the Riccati equation by
implicit product integration on a 2000-step graded mesh, uses the classical Heston closed form at
$H = \tfrac12$, and prices with the Lewis formula on Gauss–Legendre panels out to where the integrand
falls below $10^{-17}$. Each reference vol carries two error estimates: the change from halving the
time steps, and the change when the Fourier cutoff moves 1.5 times as far.

Away from $H = \tfrac12$ the reference uses the same kind of scheme as the default solver of
roughprix, implemented separately and run much finer. Agreement therefore shows that the default
settings are converged; the error estimates bound the error of the reference itself.

The smiles cover three parameter sets, each at $H \in \{\tfrac12, 0.3, 0.1, 0, -0.2, -0.4\}$ and
four maturities, with nine strikes at $\log K = z \sqrt{V_0 T}$ for $z$ from $-3$ to $3$
($S_0 = 1$, $r = 0$):

| Set | $\lambda$ | $\theta$ | $\nu$ | $\rho$ | $V_0$ |
|---|---|---|---|---|---|
| A | 0.3 | 0.006 | 0.3 | −0.7 | 0.02 |
| B | 1.0 | 0.025 | 0.4 | −0.65 | 0.025 |
| C | 0 | 0 | 0.5 | −0.9 | 0.04 |

Set C has a steep skew and no mean reversion. Each entry is the largest difference in implied
volatility, in basis points, over the values of $H$ and the strikes. Strikes count where the
out-of-the-money price is at least $10^{-6} S_0$; below that, rounding dominates the vols of both
pricers. "no vol" means that a counted strike got no implied vol.

--8<-- "docs/generated/accuracy.md"

The H-expansion rows price the target $H$ from the anchor $H_0$ with the default base at the anchor:
implicit product integration at $H_0 = 0$ and the classical Heston solution at $H_0 = \tfrac12$.
Each entry is the largest error over the three parameter sets. The largest errors sit at the outer
strikes, two to three standard deviations from the money. At five years from $H_0 = \tfrac12$ they
come from set C: 15 bp at order 4, against 0.4 bp or less for sets A and B. In such cases the
accuracy check warns, with estimates close to the errors, for example 17.7 bp for set C at five
years. For more on the accuracy of the expansion see
[Pricing with the H-expansion](expansion.md#accuracy).

## Speed

Wall-clock times of `eur_call_put`, pricing nine strikes of set A with the default settings unless
stated otherwise. The defaults include the accuracy check, which adds about 10–35 % run time.

--8<-- "docs/generated/speed.md"

To reproduce the tables, run `python docs/scripts/benchmarks.py` from the repository root. It
rewrites `docs/generated/accuracy.md` and `docs/generated/speed.md` in a few minutes.
