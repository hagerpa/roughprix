### Time per call

| Nine strikes, set A | 1 week | 1 month | 1 year | 5 years |
|---|---|---|---|---|
| Default, $H = 0.1$ | 119 | 108 | 86 | 72 |
| Root-Padé base, $H = 0.1$ | 164 | 147 | 114 | 115 |
| Exact Heston, $H = \tfrac12$ | 100 | 91 | 93 | 80 |
| Expansion $N = 4$, $H_0 = 0 \to H = 0.1$ | 384 | 320 | 253 | 220 |

### Many values of H in one call

| Nine strikes, 21 values of $H$ in $[-0.2, 0.2]$ | Expansion $N = 4$ from $H_0 = 0$, one call | Default solver, one call per $H$ |
|---|---|---|
| 1 month | 573 | 2332 |
| 1 year | 384 | 1935 |

Times in milliseconds, medians of 7 calls after a first call. The first call in a session compiles; here it took 1.9 s. Machine: Apple M3, 8 cores; Python 3.13.9, JAX 0.9.2, NumPy 2.3.3, roughprix 0.1.0. Load average during the timings: 12.3.
