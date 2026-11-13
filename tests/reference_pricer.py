"""Independent reference pricer behind tests/data/reference_smiles.json.

It shares no code with roughprix.  Run as a script, it recomputes the file (about 6 minutes on
8 cores):  python tests/reference_pricer.py

Model, in roughprix's raw convention, with alpha = H + 1/2, S_0 = 1 and r = 0:
    V_t = V_0 + 1/Gamma(alpha) int_0^t (t - s)^(alpha-1) [(theta - lam V_s) ds + nu sqrt(V_s) dB_s],
    dS_t = S_t sqrt(V_t) dW_t,   d<W, B>_t = rho dt.
At a = u - i/2 the characteristic function of X = log S_T is
    log phi(a) = int_0^T [theta h(s) + V_0 F(h(s))] ds,   h = I^alpha F(h),
    F(h) = -a(a + i)/2 - (lam - i rho nu a) h + nu^2 h^2 / 2,
with I^alpha the Riemann-Liouville integral.  At H = 1/2 the classical Heston closed form is used.

Riccati equation: F(h) is interpolated linearly between the points of the graded mesh
t_n = T (n/N)^r, r = min(2/alpha, 10), and the kernel is integrated exactly against each piece,
    h_n = sum_(j<n) [wL_nj F(h_j) + wR_nj F(h_(j+1))].
The last term holds F(h_n), so with S_n the sum of the known terms and omega = wR_(n,n-1),
h_n = S_n + omega F(h_n) is a quadratic in h_n.  It is solved as
    h_n = 2Q / (P + sqrt(P^2 - 2 omega nu^2 Q)),   P = 1 + omega (lam - i rho nu a),
    Q = S_n - omega a(a + i)/2,
with the branch Re(sqrt(.) conj(P)) >= 0: the root that tends to Q/P as nu -> 0, in a form
free of cancellation.  log phi is the trapezoid rule on the same mesh.

Out-of-the-money prices from the Lewis formula
    C(K) = 1 - sqrt(K)/pi int_0^inf Re(e^(-iu log K) phi(u - i/2)) / (u^2 + 1/4) du,
    P(K) = C(K) - 1 + K,
on Gauss-Legendre panels out to the u where |phi(u - i/2)| / u^2 falls below 1e-17; implied
vols by Brent's method.  Each smile carries two error estimates of the vol per strike:
|iv(N) - iv(N/2)| / 3, as the time stepping is of second order, and the change when the
cutoff moves 1.5 times as far, on 144 panels of 28 nodes instead of 96 of 20.
"""
import json
import math
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.optimize import brentq
from scipy.stats import norm

PARAMS = {"A": dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02),     # README defaults
          "B": dict(lam=1.0, theta=0.025, nu=0.4, rho=-0.65, V0=0.025),   # higher vol of vol
          "C": dict(lam=0.0, theta=0.0, nu=0.5, rho=-0.9, V0=0.04)}       # steep skew, no mean reversion
H_VALUES = [0.5, 0.3, 0.1, 0.0, -0.2, -0.4]
MATURITIES = [1 / 52, 1 / 12, 1.0, 5.0]
STRIKES_SD = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], float)   # log K in units of sqrt(V_0 T)
N_STEPS, PANELS, NODES = 2000, 96, 20
OUT = Path(__file__).parent / "data" / "reference_smiles.json"


@njit(cache=True)
def _weights(t, n, alpha, ginv, wL, wR):
    """Weights of F(h_j) and F(h_(j+1)) in h_n from the interval [t_j, t_(j+1)].

    I0 and I1 are the integrals of (t_n - s)^(alpha-1) and (s - t_j)(t_n - s)^(alpha-1) over the
    interval, written without cancellation (a series in x = h_j / (t_n - t_j) for small x).
    """
    tn = t[n]
    for j in range(n):
        A = tn - t[j]
        hj = t[j + 1] - t[j]
        if j == n - 1:
            I0 = A ** alpha / alpha
            I1 = A ** (alpha + 1.0) / (alpha * (alpha + 1.0))
        else:
            x = hj / A
            Aa = A ** alpha
            I0 = Aa * (-math.expm1(alpha * math.log1p(-x))) / alpha
            if x < 1e-4:
                c1 = 1.0 - alpha
                c2 = c1 * (2.0 - alpha) / 2.0
                c3 = c2 * (3.0 - alpha) / 3.0
                I1 = (Aa / A) * hj * hj * (0.5 + c1 * x / 3.0 + c2 * x * x / 4.0 + c3 * x ** 3 / 5.0)
            else:
                I1 = A * I0 - A * Aa * (-math.expm1((alpha + 1.0) * math.log1p(-x))) / (alpha + 1.0)
        wr = I1 / hj * ginv
        wR[j] = wr
        wL[j] = I0 * ginv - wr


@njit(cache=True, parallel=True)
def riccati(a, lam, rho, nu, alpha, t):
    """h(t_n) and F(h(t_n)) for each a, by the implicit scheme above."""
    U = a.size
    N = t.size - 1
    ginv = 1.0 / math.gamma(alpha)
    h = np.zeros((U, N + 1), dtype=np.complex128)
    Fv = np.zeros((U, N + 1), dtype=np.complex128)
    c = -0.5 * a * (a + 1j)
    b = -(lam - 1j * rho * nu * a)
    q = 0.5 * nu * nu
    for k in range(U):
        Fv[k, 0] = c[k]
    wL = np.empty(N)
    wR = np.empty(N)
    for n in range(1, N + 1):
        _weights(t, n, alpha, ginv, wL, wR)
        omega = wR[n - 1]
        A2 = omega * q
        for k in prange(U):
            S = 0j
            for j in range(n):
                S += wL[j] * Fv[k, j]
            for j in range(n - 1):
                S += wR[j] * Fv[k, j + 1]
            P = 1.0 - omega * b[k]
            Qc = omega * c[k] + S
            sq = np.sqrt(P * P - 4.0 * A2 * Qc)
            if sq.real * P.real + sq.imag * P.imag < 0.0:
                sq = -sq
            hn = 2.0 * Qc / (P + sq)
            h[k, n] = hn
            Fv[k, n] = c[k] + b[k] * hn + q * hn * hn
    return h, Fv


def heston_log_cf(a, T, lam, theta, nu, rho, V0):
    """Classical Heston log characteristic function, in the form of Albrecher et al. (2007)."""
    beta = lam - 1j * rho * nu * a
    d = np.sqrt(beta ** 2 + nu ** 2 * (a * a + 1j * a))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    D = (beta - d) / nu ** 2 * (1 - e) / (1 - g * e)
    C = theta / nu ** 2 * ((beta - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    return C + D * V0


def log_cf(p, H, T, u, N=N_STEPS):
    """log phi(u - i/2) at the nodes u, with N time steps."""
    a = u - 0.5j
    if H == 0.5:
        return heston_log_cf(a, T, **p)
    alpha = H + 0.5
    t = T * np.linspace(0.0, 1.0, N + 1) ** min(2.0 / alpha, 10.0)
    t[0], t[-1] = 0.0, T
    h, F = riccati(a, p["lam"], p["rho"], p["nu"], alpha, t)
    return np.trapezoid(p["theta"] * h + p["V0"] * F, t, axis=-1)


def fourier_cutoff(p, H, T):
    """The u beyond which |phi(u - i/2)| / u^2 < 1e-17, at least 50.

    Searched on 70 geometric points from 5 to 2e5, with 300 time steps.
    """
    u = np.geomspace(5.0, 2e5, 70)
    above = np.nonzero(log_cf(p, H, T, u, 300).real - 2 * np.log(u) > np.log(1e-17))[0]
    if len(above) == 0:
        return 50.0
    return float(max(50.0, u[min(above[-1] + 1, len(u) - 1)]))


def gauss_legendre_panels(umax, panels, nodes):
    """Nodes and weights on [0, umax]: a first panel up to min(2, umax/panels), then
    geometrically growing panels."""
    edges = np.concatenate([[0.0], np.geomspace(min(2.0, umax / panels), umax, panels)])
    x, w = np.polynomial.legendre.leggauss(nodes)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


def otm_prices(p, H, T, K, umax, N=N_STEPS, panels=PANELS, nodes=NODES):
    """Out-of-the-money prices by the Lewis formula."""
    u, w = gauss_legendre_panels(umax, panels, nodes)
    k = -np.log(K)
    integrand = np.real(np.exp(1j * np.outer(k, u) + log_cf(p, H, T, u, N))) / (u ** 2 + 0.25)
    calls = 1.0 - np.sqrt(K) / np.pi * (integrand @ w)
    return np.where(K >= 1.0, calls, calls - 1.0 + K)


def _bs_otm(K, T, sig):
    sT = sig * np.sqrt(T)
    d1 = (-np.log(K) + 0.5 * sT ** 2) / sT
    d2 = d1 - sT
    return norm.cdf(d1) - K * norm.cdf(d2) if K >= 1.0 else K * norm.cdf(-d2) - norm.cdf(-d1)


def implied_vols(price, K, T):
    """Black-Scholes vols of out-of-the-money prices; NaN for prices below 1e-15."""
    out = np.full(len(K), np.nan)
    for i, (pr, k) in enumerate(zip(price, K)):
        if pr > 1e-15:
            try:
                out[i] = brentq(lambda s: _bs_otm(k, T, s) - pr, 1e-6, 5.0, xtol=1e-14, rtol=1e-14,
                                maxiter=500)
            except ValueError:
                pass
    return out


def reference_smile(name, H, T):
    """Strikes, prices and vols of one smile, with the two error estimates of the vols."""
    p = PARAMS[name]
    K = np.exp(STRIKES_SD * np.sqrt(p["V0"] * T))
    umax = fourier_cutoff(p, H, T)
    price = otm_prices(p, H, T, K, umax)
    iv = implied_vols(price, K, T)
    iv_half = implied_vols(otm_prices(p, H, T, K, umax, N=N_STEPS // 2), K, T)
    iv_fine = implied_vols(otm_prices(p, H, T, K, 1.5 * umax, panels=144, nodes=28), K, T)
    est = lambda e: [float(f"{x:.2e}") for x in e]
    return dict(set=name, H=H, T=T, K=K.tolist(), price=price.tolist(), iv=iv.tolist(),
                err_time=est(np.abs(iv - iv_half) / 3), err_fourier=est(np.abs(iv_fine - iv)))


if __name__ == "__main__":
    smiles = []
    for name in PARAMS:
        for H in H_VALUES:
            for T in MATURITIES:
                smiles.append(reference_smile(name, H, T))
                print(f"{name} H={H:+.1f} T={T:.4f}", flush=True)
    head = dict(source="tests/reference_pricer.py", params=PARAMS, strikes_sd=STRIKES_SD.tolist(),
                price="out-of-the-money, S_0 = 1, r = 0")
    OUT.write_text(json.dumps(head)[:-1] + ', "smiles": [\n'
                   + ",\n".join(json.dumps(s) for s in smiles) + "\n]}\n")
    print("wrote", OUT)
