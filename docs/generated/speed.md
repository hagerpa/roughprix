### Time per call

| Nine strikes, set A | 1 week | 1 month | 1 year | 5 years |
|---|---|---|---|---|
| Default, $H = 0.1$ | 75 | 74 | 60 | 41 |
| Root-Padé base, $H = 0.1$ | 119 | 111 | 74 | 47 |
| Exact Heston, $H = \tfrac12$ | 69 | 65 | 77 | 57 |
| Expansion $N = 4$, $H_0 = 0 \to H = 0.1$ | 224 | 201 | 164 | 140 |

### Many values of H in one call

| Nine strikes, 21 values of $H$ in $[-0.2, 0.2]$ | Expansion $N = 4$ from $H_0 = 0$, one call | Default solver, one call per $H$ |
|---|---|---|
| 1 month | 316 | 1396 |
| 1 year | 228 | 1060 |

Times in milliseconds, medians of 7 calls after a first call. The first call in a session compiles; here it took 1.3 s. Machine: Apple M3, 8 cores; Python 3.13.9, JAX 0.9.2, NumPy 2.3.3, roughprix 0.1.0. Load average during the timings: 4.0.
