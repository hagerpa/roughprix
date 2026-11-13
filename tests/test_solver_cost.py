"""The solvers' cost: scans compiled once per shape, one matrix product per kernel operator,
forward substitution in blocks.

The scans of the linear Volterra solve, of implicit-pi, of the linearized solve at anchor 1/2
and of Adams are jitted at module level and take the model coefficients as arguments, so
pricing again with other parameters on the same Fourier nodes and time grid compiles nothing.
Each kernel-derivative operator K[n] = sum_j c_nj / Gamma(H0 + 1/2) (t-s)^(H0-1/2) log^j(t-s)
is one precombined matrix (at H0 = 1/2 plus the cumulative sum of the regular j = 0 term), so
applying it is one matrix product.  Implicit-pi and the linear
Volterra solve run forward substitution in blocks: the history of earlier blocks is one matrix
product per block.  The plain forward substitution below, which sums the whole history at every
step, is the reference for their results and their speed.  The Gauss-Legendre nodes of the
Fourier panels are computed once per order.
"""
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from scipy.special import gamma

from roughprix import RoughHeston, RoughHestonParam
from roughprix.heston.integration import Integrate, LinearVolterraSolver, ProductIntegrationRiccatiSolver, TimeGrid
from roughprix.heston.rough_heston import (
    _build_kernel_derivative_ops, _raw_kernel_op, _relative_gamma_poly_coeffs,
)

_A = dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02)
_B = dict(lam=1.0, theta=0.025, nu=0.4, rho=-0.65, V0=0.025)


# One listener for the module: the jax versions for Python 3.10 (0.6) cannot unregister one.
_EVENTS = []
jax.monitoring.register_event_duration_secs_listener(lambda event, duration, **kw: _EVENTS.append(event))


def _backend_compiles(f):
    start = len(_EVENTS)
    f()
    return _EVENTS[start:].count("/jax/core/compile/backend_compile_duration")


# anchor, model keywords, pricing keywords
_CASES = {
    "implicit_pi": (0.1, {}, {}),
    "implicit_pi_expansion": (0.0, {}, dict(expansion_order=2, H=0.1)),
    "root_pade_expansion": (0.0, dict(base_solution="root_pade"), dict(expansion_order=2, H=[0.05, 0.1])),
    "direct_riccati_from_half": (0.5, dict(base_solution="direct_riccati"), dict(expansion_order=2, H=0.4)),
    "exact_heston_expansion": (0.5, {}, dict(expansion_order=2, H=0.4)),
    "adams": (0.1, dict(base_solution="direct_riccati", direct_riccati_solver="adams"), dict(time_grid="uniform")),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_pricing_again_compiles_nothing(case):
    """Other parameters on the same Fourier nodes and time grid reuse every compiled scan."""
    H0, model_kw, kw = _CASES[case]
    kw = dict(kw, K=np.exp(np.linspace(-0.3, 0.2, 9)), maturity=0.5, L=200.0, implied_vol=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RoughHeston(RoughHestonParam(**_A, H=H0), 0.0, **model_kw).eur_call_put(**kw)
        m = RoughHeston(RoughHestonParam(**_B, H=H0), 0.0, **model_kw)
        assert _backend_compiles(lambda: m.eur_call_put(**kw)) == 0


def test_pricing_again_reuses_the_gauss_legendre_nodes(monkeypatch):
    """The Gauss-Legendre nodes of the Fourier panels, an eigenvalue problem, are computed once
    per order: pricing again with the default settings computes none."""
    kw = dict(K=np.exp(np.linspace(-0.3, 0.2, 9)), maturity=0.5)
    RoughHeston(RoughHestonParam(**_A, H=0.1), 0.0).eur_call_put(**kw)
    calls = []
    leggauss = np.polynomial.legendre.leggauss
    monkeypatch.setattr(np.polynomial.legendre, "leggauss", lambda deg: (calls.append(deg), leggauss(deg))[1])
    RoughHeston(RoughHestonParam(**_B, H=0.1), 0.0).eur_call_put(**kw)
    assert calls == []


def _field(grid):
    return grid.field(np.exp(1j * grid.t) * np.arange(1, 4)[:, None] + grid.t ** 0.3, has_time_axis=True)


@pytest.mark.parametrize("anchor", [0.5, 0.3, 0.0, -0.3])
def test_kernel_derivative_operators_apply_one_product(anchor, monkeypatch):
    grid = TimeGrid.graded(0.7, 60, power=2.0)
    K = _build_kernel_derivative_ops(grid, anchor, 4)
    f = _field(grid)
    calls = []
    matmul = jnp.matmul
    monkeypatch.setattr(jnp, "matmul", lambda *a, **kw: (calls.append(1), matmul(*a, **kw))[1])
    for n in range(1, 5):
        calls.clear()
        K[n] @ f
        assert len(calls) == 1, f"K[{n}]"


@pytest.mark.parametrize("anchor", [0.5, 0.3, 0.0, -0.3])
def test_kernel_derivative_operators_match_their_terms(anchor):
    """K[n] = sum_j c_nj / Gamma(anchor + 1/2) * (raw kernel with log^j), the regular term included."""
    grid = TimeGrid.graded(0.7, 60, power=2.0)
    K = _build_kernel_derivative_ops(grid, anchor, 4)
    polys = _relative_gamma_poly_coeffs(anchor, 4)
    f = _field(grid)
    for n in range(5):
        want = sum(c / gamma(anchor + 0.5) * np.asarray((_raw_kernel_op(grid, anchor, j) @ f).values)
                   for j, c in enumerate(polys[n]))
        np.testing.assert_allclose(np.asarray((K[n] @ f).values), want, rtol=1e-12, atol=1e-13 * np.abs(want).max())


@jax.jit
def _implicit_pi_step_by_step(Omega, c, beta, half_gam):
    """Implicit-pi by plain forward substitution: every step sums the whole zero-padded history."""
    def step(carry, j):
        psi, F = carry
        w = Omega[j, j]
        rhs = F @ Omega[j] + w * c
        denom = 1.0 + w * beta
        lin = rhs / denom
        a = half_gam * w
        sq = jnp.sqrt(denom ** 2 - 4.0 * a * rhs)
        sq = jnp.where(jnp.real(jnp.conj(denom) * sq) < 0, -sq, sq)
        r1, r2 = (denom + sq) / (2.0 * a), 2.0 * rhs / (denom + sq)
        p = jnp.where(jnp.abs(a) >= 1e-14, jnp.where(jnp.abs(r1 - lin) <= jnp.abs(r2 - lin), r1, r2), lin)
        return (psi.at[:, j].set(p), F.at[:, j].set(c - beta * p + half_gam * p ** 2)), None

    z = jnp.zeros((c.shape[0], Omega.shape[0]), dtype=jnp.complex128)
    (psi, _), _ = lax.scan(step, (z, z.at[:, 0].set(c)), jnp.arange(1, Omega.shape[0]))
    return psi


@jax.jit
def _linear_step_by_step(Omega, R, C, scale):
    """The linear Volterra solve by plain forward substitution: every step sums the whole zero-padded history."""
    def step(x, j):
        hist = jnp.sum(Omega[j] * C * x, axis=-1)
        return x.at[..., j].set((R[..., j] + scale * hist) / (1.0 - scale * Omega[j, j] * C[..., j])), None

    return lax.scan(step, jnp.zeros_like(R), jnp.arange(R.shape[-1]))[0]


def _grid(J):
    return TimeGrid.graded(0.9, J, power=2.4)


def _implicit_pi(n_u, J, alpha=0.6, gam=0.09):
    """(blocked solve, step-by-step solve) of one Riccati equation."""
    u = np.concatenate([np.linspace(0.01, 200, n_u - 1), [0.0]]) - 0.5j
    solver = ProductIntegrationRiccatiSolver(_grid(J), alpha, -0.5 * u * (u + 1j), 0.3 - 0.21j * u, gam)
    args = (jnp.asarray(solver._Omega), jnp.asarray(solver.c), jnp.asarray(solver.beta), jnp.array(0.5 * gam))
    return lambda: solver.solve().values, lambda: _implicit_pi_step_by_step(*args)


def _linear(n, J, scale=1.0, kernel=dict(kernel="frac", order=-0.4)):
    """(blocked solve, step-by-step solve) of one linear Volterra equation."""
    grid = _grid(J)
    K = Integrate(grid, **kernel)
    rng = np.random.default_rng(0)
    R = jnp.asarray(rng.standard_normal((n, J + 1)) + 1j * rng.standard_normal((n, J + 1)))
    C = jnp.asarray(0.5 * rng.standard_normal((n, J + 1)) - 2j * rng.random((n, J + 1)))
    solver = LinearVolterraSolver(grid, K, scale=scale)
    return lambda: solver.solve(R, C).values, lambda: _linear_step_by_step(jnp.asarray(K.weights_matrix), R, C, scale)


@pytest.mark.parametrize("J", [1, 7, 8, 9, 60])
@pytest.mark.parametrize("alpha", [0.1, 0.6, 1.0])
def test_implicit_pi_blocks_match_step_by_step(J, alpha):
    """Same equations, same roots; only the order of the history sum differs (blocks of 8, padded)."""
    blocked, step_by_step = _implicit_pi(40, J, alpha)
    want = np.asarray(step_by_step())
    np.testing.assert_allclose(np.asarray(blocked()), want, rtol=0, atol=1e-13 * np.abs(want).max())


@pytest.mark.parametrize("J", [1, 7, 8, 9, 60])
@pytest.mark.parametrize("scale", [1.0, 0.3 - 0.2j])
@pytest.mark.parametrize("kernel", [dict(kernel="frac", order=-0.4), dict(kernel="fraclog", order=2, frac_power=-0.3)])
def test_linear_blocks_match_step_by_step(J, scale, kernel):
    blocked, step_by_step = _linear(6, J, scale, kernel)
    want = np.asarray(step_by_step())
    np.testing.assert_allclose(np.asarray(blocked()), want, rtol=0, atol=1e-13 * np.abs(want).max())


def _best_time(f):
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        f().block_until_ready()
        times.append(time.perf_counter() - t0)
    return min(times)


@pytest.mark.parametrize("solve", [_implicit_pi, _linear])
def test_blocks_are_faster_than_step_by_step(solve):
    blocked, step_by_step = solve(384, 640)
    blocked().block_until_ready()
    step_by_step().block_until_ready()
    assert _best_time(step_by_step) > 2 * _best_time(blocked)   # typically 7x (implicit-pi), 24x (linear)
