import warnings
from dataclasses import dataclass
from math import comb, factorial
from typing import Callable, Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from scipy.special import gamma

from .integration import (
    Field, Integrate, TimeGrid, exp_kernel_cumulative, exp_kernel_weights, suggested_time_grid_power,
)


@dataclass(slots=True)
class RoughHestonParam:
    """Parameters for the rough Heston stochastic volatility model.

    The variance process satisfies

        dS_t = sqrt(V_t) S_t dW_t
        V_t  = V0 + (1/Γ(H+½)) ∫₀ᵗ (t-s)^(H-½) [(theta - lam*V_s) ds
                                                    + nu*sqrt(V_s) dB_s]
        corr(dW_t, dB_t) = rho

    where H ∈ (-½, ½] is the Hurst exponent; H = ½ recovers classical Heston.
    All parameters are the *raw* (un-normalised) Riccati coefficients, so the
    drift constant is theta itself and not lam*theta: in the usual Heston
    parameterisation dV = kappa(vbar - V)dt, set lam = kappa and
    theta = kappa*vbar.

    Parameters
    ----------
    lam : float
        Mean-reversion speed of the variance process.
    theta : float
        Raw drift constant. The long-run variance is theta/lam, which the
        defaults below make equal to V0.
    nu : float
        Vol-of-vol (raw, un-normalised).
    rho : float
        Correlation between price and variance Brownian motions; typically < 0.
    V0 : float
        Initial instantaneous variance.
    H : float
        Hurst exponent; must lie in (-½, ½].  H = ½ → classical Heston.
    """
    lam: float = 0.3
    theta: float = 0.3 * 0.02
    nu: float = 0.3
    rho: float = -0.7
    V0: float = 0.02
    H: float = 0.4



@dataclass(slots=True)
class HestonExpansionState:
    """Intermediate state returned by :meth:`RoughHeston.expansion_state`.

    Attributes
    ----------
    grid : TimeGrid
        Time discretisation used for the Volterra solve.
    u : np.ndarray
        Complex Fourier argument array, shape ``(N_u,)``.
    alpha : float
        ``Γ(anchor_H + ½)`` — the normalisation factor for the fractional kernel.
    f0 : Field
        Base Riccati solution ``h(u, ·)`` at the expansion anchor, shape ``(N_u, N_t)``.
    expansion : Field
        Taylor-summed approximation ``Σ_{n=0}^{N} (H-H₀)ⁿ/n! fₙ``, same shape as ``f0``.
    F_ricc : Callable
        Unscaled Riccati right-hand side ``F_raw(x) = c - β x + ½γ x²``.
    kappa_path : Field
        Conditional-mean path ``V0 + (θ/Γ(H+3/2)) t^(H+½)``, shape ``(N_t,)``.
    kappa_path_reflected : Field
        Same path evaluated at ``T - t`` (used for the measure change).
    f_list : list[Field] or None
        Expansion coefficients ``[f_0, f_1, …, f_N]``; populated only when
        ``return_lists=True`` is passed to :meth:`~RoughHeston.expansion_state`.
    F_list : list[Field] or None
        Corresponding Riccati RHS evaluations; populated alongside ``f_list``.
    target_H : float or None
        Hurst exponent at which ``expansion`` is evaluated.
    anchor_H : float or None
        Expansion anchor H₀.
    """
    grid: TimeGrid
    u: np.ndarray
    alpha: float
    f0: object  # Field
    expansion: object  # Field
    F_ricc: Callable
    kappa_path: object  # Field
    kappa_path_reflected: object  # Field
    f_list: list | None = None
    F_list: list | None = None
    target_H: float | None = None
    anchor_H: float | None = None


@dataclass(slots=True)
class RiccatiContext:
    """Shared setup that is independent of the expansion anchor point."""
    grid: TimeGrid
    u: np.ndarray
    Z: object  # Field: grid.field(u)
    time: object  # Field: grid.field(grid.t)
    beta: object  # Field: lam_raw - rho*nu_raw*u*1j  (unscaled)
    gam: float  # nu_raw^2  (unscaled)
    alpha: float  # Gamma(anchor_H+0.5); stored for direct-Riccati and exact-Heston solvers
    anchor_H: float  # anchor Hurst exponent
    theta: float
    V0: float
    F_ricc: Callable  # Riccati RHS F_raw (unscaled; kernel ops include 1/Γ)
    kappa_path: object  # Field: V0 + (theta/Γ(H+3/2)) * t^(H+1/2)
    kappa_path_reflected: object  # Field: same but evaluated at T-t


@dataclass(slots=True)
class AnchorExpansionOps:
    """Anchor-specific objects needed to run the expansion recursion."""
    anchor_H: float  # anchor point, e.g. 0.5 for the Heston branch
    f0: object  # Field: base Riccati solution at anchor
    K: object  # dict[int, Integrate]: log^k kernel operators
    solve_linearized: Callable  # [[Field], Field]: linear resolvent
    log_phi_anchor: np.ndarray | None = None  # closed-form log CF at target H = anchor, if known


class _LinearCombOp:
    """Weighted sum of integration operators: result = Σ_i c_i * (K_i @ obj).

    The dense terms are summed into one matrix up front, so an application costs one matrix
    product, not one per term; a kernel='regular' term keeps its O(J) cumulative sum.
    """

    def __init__(self, terms: list):
        self._grid = terms[0][1].grid
        self._regular = [(c, K) for c, K in terms if K.kernel == "regular"]
        self._Omega = None  # Σ c_i Ω_i over the dense terms
        for c, K in terms:
            if K.kernel != "regular":
                m = c * K.weights_matrix
                self._Omega = m if self._Omega is None else self._Omega + m
        self._Omega_T = None if self._Omega is None else jnp.asarray(self._Omega.T)

    def __matmul__(self, obj):
        field = self._grid.field(obj)
        result = None if self._Omega_T is None else Field(jnp.matmul(field.values, self._Omega_T), self._grid)
        for c, K in self._regular:
            term = c * (K @ field)
            result = term if result is None else result + term
        return result

    @property
    def weights_matrix(self):
        if self._regular:
            raise ValueError("weights_matrix is not available for kernel='regular'")
        return self._Omega


def _raw_kernel_op(grid: TimeGrid, anchor_H: float, log_order: int) -> Integrate:
    """Return the operator for (t-s)^(anchor_H-1/2) * log(t-s)^log_order."""
    q = anchor_H - 0.5
    if log_order == 0:
        if abs(q) < 1e-15:
            return Integrate(grid, kernel="regular", start=0)
        return Integrate(grid, kernel="frac", order=q, start=0)
    if abs(q) < 1e-15:
        return Integrate(grid, kernel="log", order=log_order, start=0)
    return Integrate(grid, kernel="fraclog", order=log_order, frac_power=q, start=0)


def _poly_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.convolve(p, q)


def _relative_gamma_poly_coeffs(anchor_H: float, order: int) -> list:
    """Return polys[n] = coefficients of P_n(L), n=0..order.

    The n-th derivative kernel at H=anchor_H is
        K_n(t) = t^(anchor_H-1/2) * P_n(log t)
    where P_n satisfies the recurrence
        P_{n+1} = sum_{k=0}^n C(n,k) * P_{n-k} * x_{k+1}
    with x_1(L) = L - digamma(a) and x_m = -polygamma(m-1, a) for m>=2.

    coeffs[j] = coefficient of L**j.
    """
    from scipy.special import digamma, polygamma

    a = anchor_H + 0.5
    polys = [np.array([1.0], dtype=float)]

    x = [None] * (order + 1)
    if order >= 1:
        x[1] = np.array([-float(digamma(a)), 1.0], dtype=float)
    for m in range(2, order + 1):
        x[m] = np.array([-float(polygamma(m - 1, a))], dtype=float)

    for n in range(order):
        P_next = np.zeros(n + 2, dtype=float)
        for k in range(n + 1):
            term = _poly_mul(polys[n - k], x[k + 1])
            P_next[:len(term)] += comb(n, k) * term
        polys.append(P_next)

    return polys


def _build_kernel_derivative_ops(
        grid: TimeGrid,
        anchor_H: float,
        order: int,
) -> dict:
    """Build the dict of kernel-derivative operators K[0], ..., K[order].

    Each K[n] is the n-th H-derivative of the full kernel (t-s)^(H-1/2)/Γ(H+1/2)
    evaluated at H=anchor_H, so K[0] is the proper fractional integral
    (1/Γ(anchor_H+1/2)) * ∫(t-s)^(anchor_H-1/2) [.] ds.  The 1/Γ factor is
    absorbed here so callers can work with the unscaled Riccati RHS F_raw.
    """
    alpha = gamma(anchor_H + 0.5)
    polys = _relative_gamma_poly_coeffs(anchor_H, order)
    raw_ops = {j: _raw_kernel_op(grid, anchor_H, j) for j in range(order + 1)}

    tol = 1e-14
    K = {}
    for n in range(order + 1):
        terms = [
            (float(c) / alpha, raw_ops[j])
            for j, c in enumerate(polys[n])
            if abs(c) > tol
        ]
        if len(terms) == 1 and abs(terms[0][0] - 1.0) <= tol:
            K[n] = terms[0][1]
        else:
            K[n] = _LinearCombOp(terms)
    return K


# Jitted at module level: a closure built per call would be traced and compiled on every solve.
@jax.jit
def _left_rectangle_scan(R, dt, Fx0):
    def _step(acc, j):
        x_j = R[..., j] + acc
        return acc + dt[j] * Fx0[..., j] * x_j, x_j

    _, x_cols = lax.scan(_step, jnp.zeros(R.shape[:-1], dtype=jnp.complex128), jnp.arange(R.shape[-1]))
    return jnp.moveaxis(x_cols, 0, -1)


# Module level, so the Adams scan, which takes it as a static argument, compiles once per shape.
def _riccati_rhs(j, x, c, beta, half_gam):
    return c - beta * x + half_gam * x * x


class RoughHeston(object):
    """Rough Heston model pricer based on a fractional-Riccati Taylor expansion.

    The log-price characteristic function is approximated by expanding the
    fractional Riccati solution around an anchor Hurst exponent ``H₀ = param.H``
    to arbitrary order in ``ΔH = H_target - H₀``.  At order 0 the expansion
    collapses to the exact model characteristic function at the anchor.

    The anchor Riccati solution ``h(u, t; H₀)`` is computed by the method
    selected via ``base_solution``:

    * ``"root_pade"``  — root-transformed Padé approximant.
    * ``"exact_heston"`` — closed-form classical Heston (only valid at H₀ = ½).
    * ``"pade"``       — Gatheral Padé [m/m] approximant.
    * ``"direct_riccati"`` — numerical fractional ODE solver (Adams or implicit-π).
    * ``"auto"``       — ``"exact_heston"`` when ``H₀ = ½``, else ``"direct_riccati"``
      with implicit-π.

    European option prices are computed via Fourier inversion on the Lewis
    contour (``R = 0.5``) or a damped-payoff contour (any other ``R``).
    """

    def __init__(
            self,
            param: RoughHestonParam,
            riskfree: float,
            base_solution: str = "auto",
            pade_order: str = "22",
            root_pade_series_order: int | None = None,
            root_pade_degree: int = 9,
            root_pade_tail_match: bool = True,
            direct_riccati_solver: Literal["auto", "adams", "implicit_pi"] = "auto",
            log_cf_integrand: Literal["auto", "solution", "rhs"] = "auto",
    ):
        """Initialise the rough Heston pricer.

        Parameters
        ----------
        param : RoughHestonParam
            Model parameters.  ``param.H`` is used as the expansion anchor H₀.
        riskfree : float
            Continuously compounded risk-free rate.
        base_solution : {"auto", "root_pade", "exact_heston", "pade", "direct_riccati"}
            Method used to solve the base Riccati equation at the anchor.
            ``"auto"`` selects ``"exact_heston"`` when ``param.H == 0.5``,
            otherwise ``"direct_riccati"`` with the implicit-π solver, the most
            stable and accurate choice; ``"root_pade"`` is faster.  The
            approximate bases warn at ``param.H == 0.5``, where
            ``"exact_heston"`` is exact.
        pade_order : {"22", "33", "44"}
            Padé order; only used when ``base_solution="pade"``.
        root_pade_series_order : int or None
            Number of short-time series coefficients for the root-Padé solve;
            ``None`` defaults to ``2 * root_pade_degree + 1``.
        root_pade_degree : int
            Numerator degree ``m`` of the ``[m/(m+1)]`` root-Padé approximant
            (default 9; Hager & Kreher 2026 use 7).
        root_pade_tail_match : bool
            If ``True`` (the default), replace the highest short-time Padé
            equation with the exact leading ``1/x`` tail constraint.  This is
            much more accurate for H < 0 and for steep skews.
        direct_riccati_solver : {"auto", "adams", "implicit_pi"}
            Fractional ODE solver when ``base_solution="direct_riccati"``.
            ``"auto"`` selects ``"implicit_pi"``.  ``"adams"`` is opt-in and
            requires a uniform time grid.
        log_cf_integrand : {"auto", "solution", "rhs"}
            What the log characteristic function integrates over ``[0, T]``.
            ``"solution"`` integrates the Riccati solution h:
            ``θ·I¹h(T) + V0·I^(1-α)h(T)`` (El Euch & Rosenbaum 2019).
            ``"rhs"`` integrates the Riccati right-hand side F(h):
            ``∫ F(h(s)) (V0 + θ (T-s)^α / Γ(1+α)) ds``, the formula of
            Hager & Kreher (2026), Eq. (1.1), which the paper's numbers use
            (see "Reproducing the paper" in the README for the other
            settings).  Both are exact for the exact h.  ``"rhs"`` equals
            ``"solution"`` after one fixed-point step with the target kernel:
            that restores the target's short-time behaviour t^α, which the
            H-expansion carries only through powers of ΔH·log t, but amplifies
            the error of an approximate h by about ν|u|.  ``"auto"`` (the
            default) uses ``"rhs"`` for expansions from the anchor H₀ = ½,
            where it is the more accurate of the two in tests, and
            ``"solution"`` otherwise.
        """
        if log_cf_integrand not in ("auto", "solution", "rhs"):
            raise ValueError(f"log_cf_integrand must be 'auto', 'solution' or 'rhs', got {log_cf_integrand!r}")
        self.param = param
        self.riskfree = riskfree
        self.base_solution = base_solution
        self.pade_order = pade_order
        self.root_pade_series_order = root_pade_series_order
        self.root_pade_degree = root_pade_degree
        self.root_pade_tail_match = root_pade_tail_match
        self.direct_riccati_solver = direct_riccati_solver
        self.log_cf_integrand = log_cf_integrand
        self._field_cache = None   # dict while eur_call_put runs its accuracy check
        self._extra_order = 0      # 1 while that check needs the next expansion order

    def expansion_state(
            self,
            arg: np.ndarray,
            maturity: float = 1.0,
            order: int = 0,
            time_steps: int = 400,
            H: float | None = None,
            return_lists: bool = False,
            anchor: float | None = None,
            base: Literal["auto", "exact_heston", "pade", "root_pade", "direct_riccati"] | None = None,
            pade_order: str | None = None,
            root_pade_degree: int | None = None,
            root_pade_series_order: int | None = None,
            root_pade_tail_match: bool | None = None,
            time_grid: Literal["uniform", "graded", "symmetric_graded"] = "graded",
            grid_power: float | None = None,
            direct_riccati_solver: Literal["auto", "adams", "implicit_pi"] | None = None,
    ) -> HestonExpansionState:
        """Compute the fractional-Riccati expansion state at a given Fourier argument.

        Solves the base Riccati equation at the anchor ``H₀`` and accumulates
        the H-derivative fields ``f_0, f_1, …, f_order`` on the time grid.
        The result can be passed directly to :meth:`log_charfun_from_field` or
        inspected for diagnostics.

        Parameters
        ----------
        arg : np.ndarray
            Complex Fourier argument, shape ``(N_u,)``.  Typically
            ``-1j*(R - 1j*u)`` for a contour at height ``R``.
        maturity : float
            Option maturity T > 0.
        order : int
            Taylor expansion order N.  Order 0 gives the exact anchor solution.
        time_steps : int
            Number of time-grid nodes on ``[0, T]``.
        H : float or None
            Target Hurst exponent for the expansion evaluation.
            ``None`` uses ``self.param.H`` (i.e. evaluates at the anchor itself).
        return_lists : bool
            If ``True``, populate ``HestonExpansionState.f_list`` and ``F_list``
            with the individual coefficient fields ``f_n``.
        anchor : float or None
            Override the expansion anchor for this call only.
            ``None`` (default) uses ``self.param.H``.
        base : str or None
            Override ``self.base_solution`` for this call only.
        pade_order, root_pade_degree, root_pade_series_order, root_pade_tail_match
            Per-call overrides for the corresponding constructor settings.
        time_grid : {"uniform", "graded", "symmetric_graded"}
            Time-node distribution.  ``"graded"`` clusters nodes near ``t = 0``
            (recommended for small anchors); ``"symmetric_graded"`` additionally
            clusters near ``t = T``.
        grid_power : float or None
            Grading exponent; ``None`` auto-selects via
            :func:`suggested_time_grid_power`.
        direct_riccati_solver : {"auto", "adams", "implicit_pi"} or None
            Per-call override for the fractional ODE solver.

        Returns
        -------
        HestonExpansionState
        """
        target_H = float(H) if H is not None else self.param.H
        anchor_eff = float(anchor) if anchor is not None else self.param.H
        base_eff = base if base is not None else self.base_solution
        po_eff = pade_order if pade_order is not None else self.pade_order
        rpd_eff = root_pade_degree if root_pade_degree is not None else self.root_pade_degree
        rpso_eff = root_pade_series_order if root_pade_series_order is not None else self.root_pade_series_order
        rptm_eff = root_pade_tail_match if root_pade_tail_match is not None else self.root_pade_tail_match
        drs_eff = direct_riccati_solver if direct_riccati_solver is not None else self.direct_riccati_solver

        alpha_anchor = anchor_eff + 0.5
        if not (0.0 < alpha_anchor <= 1.0):
            raise ValueError(f"anchor must be in (-0.5, 0.5], got {anchor_eff!r}")
        if base_eff not in ("auto", "exact_heston", "pade", "root_pade", "direct_riccati"):
            raise ValueError(
                f"base must be one of 'auto', 'exact_heston', 'pade', 'root_pade', "
                f"'direct_riccati', got {base_eff!r}"
            )
        if base_eff == "direct_riccati" and drs_eff == "adams" and time_grid != "uniform":
            raise ValueError(
                f"base='direct_riccati' with direct_riccati_solver='adams' requires "
                f"time_grid='uniform' (got {time_grid!r})."
            )
        ctx = self._build_convention_context(
            arg, maturity, time_steps,
            anchor_H=anchor_eff, target_H=target_H,
            time_grid=time_grid, grid_power=grid_power,
        )
        ops = self._build_anchor_ops(
            ctx, order, base_eff, anchor_eff,
            pade_order=po_eff,
            root_pade_degree=rpd_eff,
            root_pade_series_order=rpso_eff,
            root_pade_tail_match=rptm_eff,
            direct_riccati_solver=drs_eff,
        )
        expansion, f_list, F_list = self._run_expansion_recursion(
            ctx=ctx, ops=ops, target_H=target_H, order=order, return_lists=return_lists,
        )
        return HestonExpansionState(
            grid=ctx.grid,
            u=ctx.u,
            alpha=ctx.alpha,
            f0=ops.f0,
            expansion=expansion,
            F_ricc=ctx.F_ricc,
            kappa_path=ctx.kappa_path,
            kappa_path_reflected=ctx.kappa_path_reflected,
            f_list=f_list,
            F_list=F_list,
            target_H=target_H,
            anchor_H=anchor_eff,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _build_convention_context(
            self,
            arg: np.ndarray,
            maturity: float,
            time_steps: int,
            *,
            anchor_H: float,
            target_H: float | None = None,
            time_grid: str = "uniform",
            grid_power: float | None = None,
    ) -> RiccatiContext:
        """Build the shared Riccati context. driver_alpha = Gamma(anchor_H + 0.5)."""
        if target_H is None:
            target_H = self.param.H

        if time_steps < 1:
            raise ValueError(f"time_steps must be >= 1, got {time_steps!r}")
        if maturity <= 0:
            raise ValueError(f"maturity must be > 0, got {maturity!r}")
        if grid_power is not None and grid_power < 1.0:
            raise ValueError(f"grid_power must be >= 1, got {grid_power!r}")

        driver_alpha = gamma(anchor_H + 0.5)

        lam_raw = self.param.lam
        nu_raw = self.param.nu
        rho = self.param.rho
        theta = self.param.theta
        V0 = self.param.V0

        u = np.asarray(arg, dtype=np.complex128).reshape(-1)
        if time_grid == "uniform":
            grid = TimeGrid.uniform(T=maturity, J=int(time_steps))
        elif time_grid == "graded":
            power = grid_power if grid_power is not None else suggested_time_grid_power(anchor_H)
            grid = TimeGrid.graded(T=maturity, J=int(time_steps), power=power)
        elif time_grid == "symmetric_graded":
            power = grid_power if grid_power is not None else suggested_time_grid_power(anchor_H)
            grid = TimeGrid.symmetric_graded(T=maturity, J=int(time_steps), power=power)
        else:
            raise ValueError(
                f"time_grid must be 'uniform', 'graded', or 'symmetric_graded', got {time_grid!r}"
            )
        time = grid.field(grid.t, has_time_axis=True)

        beta = grid.field(lam_raw - rho * nu_raw * u * 1j)
        gam = nu_raw ** 2
        Z = grid.field(u)

        F_ricc = lambda x: (
                (-0.5) * Z * (Z + 1j)
                - beta * x
                + 0.5 * gam * (x * x)
        )

        kappa_path = V0 + (theta / gamma(target_H + 1.5)) * np.power(time, target_H + 0.5)

        t_ref = grid.field(np.maximum(grid.T - grid.t, 0.0), has_time_axis=True)
        kappa_path_reflected = V0 + (theta / gamma(target_H + 1.5)) * np.power(t_ref, target_H + 0.5)

        return RiccatiContext(
            grid=grid, u=u, Z=Z, time=time, beta=beta,
            gam=gam, alpha=driver_alpha,
            anchor_H=anchor_H, theta=theta, V0=V0,
            F_ricc=F_ricc, kappa_path=kappa_path,
            kappa_path_reflected=kappa_path_reflected,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _build_exact_heston_anchor_ops(
            self,
            ctx: RiccatiContext,
            order: int,
    ) -> AnchorExpansionOps:
        lam_eff = self.param.lam / ctx.alpha
        nu_eff = self.param.nu / ctx.alpha
        rho = self.param.rho

        # Heston discriminant (scalar arrays over u).  At u = 0 and u = -i the constant
        # c = -u(u+i)/2 vanishes and h = 0 is exact, but G is 0/0 or x/0 there (u = 0
        # with lam = 0, u = -i with lam <= rho*nu).  D = beta and G = 0 give h = 0.
        beta_eff = lam_eff - rho * nu_eff * ctx.u * 1j
        trivial = ctx.u * (ctx.u + 1j) == 0
        D_raw = np.where(trivial, beta_eff, np.sqrt(beta_eff ** 2 + (ctx.u + 1j) * ctx.u * nu_eff ** 2))
        G_raw = np.divide(beta_eff - D_raw, beta_eff + D_raw, out=np.zeros_like(beta_eff), where=~trivial)

        # Closed-form log CF at H = 1/2.  The variance kernel is V0 + theta*(T - s), so
        # integrating F(h(s)) against it by parts gives V0*h(T) + theta*int_0^T h(s) ds.
        eDT = np.exp(-D_raw * ctx.grid.T)
        h_T = (beta_eff - D_raw) * (1 - eDT) / (ctx.gam * (1 - G_raw * eDT))
        int_h = ((beta_eff - D_raw) * ctx.grid.T - 2 * np.log((1 - G_raw * eDT) / (1 - G_raw))) / ctx.gam
        log_phi_anchor = 1j * ctx.u * self.riskfree * ctx.grid.T + ctx.V0 * h_T + ctx.theta * int_h

        D = ctx.grid.field(D_raw)
        G = ctx.grid.field(G_raw)
        eD = np.exp(-D * ctx.time)

        f0 = ctx.gam ** -1 * (ctx.beta - D) * (1 - eD) / (1 - G * eD)
        w = (1 - G * eD) * (1 - G * eD)
        g = -ctx.beta + ctx.gam * f0
        del eD, D, G

        K = _build_kernel_derivative_ops(ctx.grid, anchor_H=0.5, order=order)
        # f_n = h_n + w(t)^-1 ∫_0^t exp(-D (t - s)) w(s) g(s) h_n(s) ds, with the exponential
        # integrated exactly against the piecewise-linear w g h_n: stable for any |D| dt, and
        # |D| grows linearly in |u|.
        exp_weights = exp_kernel_weights(D_raw, ctx.grid.t)

        def solve_linearized(h_n: Field) -> Field:
            phi = np.asarray((w * g * h_n).values, dtype=np.complex128)
            return h_n + ctx.grid.field(exp_kernel_cumulative(phi, *exp_weights), has_time_axis=True) / w

        return AnchorExpansionOps(
            anchor_H=0.5,
            f0=f0,
            K=K,
            solve_linearized=solve_linearized,
            log_phi_anchor=log_phi_anchor,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _build_anchor_ops(
            self,
            ctx: RiccatiContext,
            order: int,
            base: str,
            anchor: float,
            pade_order: str = "22",
            root_pade_degree: int = 9,
            root_pade_series_order: int | None = None,
            root_pade_tail_match: bool = True,
            direct_riccati_solver: str = "auto",
    ) -> AnchorExpansionOps:
        anchor_tol = 1e-10
        if base == "auto":
            if abs(anchor - 0.5) < anchor_tol:
                return self._build_exact_heston_anchor_ops(ctx, order)
            return self._build_approx_anchor_ops(
                ctx, order, "direct_riccati", anchor,
                direct_riccati_solver="implicit_pi",
            )
        if base == "exact_heston":
            if abs(anchor - 0.5) >= anchor_tol:
                raise ValueError(
                    f"base='exact_heston' requires anchor=0.5, got anchor={anchor!r}"
                )
            return self._build_exact_heston_anchor_ops(ctx, order)
        if abs(anchor - 0.5) < anchor_tol:
            warnings.warn(
                f"base={base!r} at anchor H = 1/2 is approximate; use 'exact_heston' (what 'auto' picks "
                "there), which is exact.  Root-Pade loses accuracy at large u there, and expansions from "
                "an approximate base can blow up on coarse time grids.",
                stacklevel=2,
            )
        return self._build_approx_anchor_ops(
            ctx, order, base, anchor,
            pade_order=pade_order,
            root_pade_degree=root_pade_degree,
            root_pade_series_order=root_pade_series_order,
            root_pade_tail_match=root_pade_tail_match,
            direct_riccati_solver=direct_riccati_solver,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _build_approx_anchor_ops(
            self,
            ctx: RiccatiContext,
            order: int,
            base: str,
            anchor: float,
            pade_order: str = "22",
            root_pade_degree: int = 9,
            root_pade_series_order: int | None = None,
            root_pade_tail_match: bool = True,
            direct_riccati_solver: str = "auto",
    ) -> AnchorExpansionOps:
        if base == "pade":
            from .pade import rough_heston_pade_f0
            f0_values = rough_heston_pade_f0(
                u=ctx.u,
                t=ctx.grid.t,
                param=self.param,
                pade_order=pade_order,
                H_anchor=anchor,
            )
        elif base == "direct_riccati":
            # Solve psi = ∫_0^t (t-s)^(alpha-1) F(psi(s)) ds
            # with F(x) = c - beta*x + 0.5*gam*x^2.
            alpha_anchor = anchor + 0.5
            lam_raw = self.param.lam
            nu_raw = self.param.nu
            rho = self.param.rho
            _c_np = np.asarray(-0.5 / ctx.alpha * ctx.u * (ctx.u + 1j), dtype=np.complex128)
            _beta_np = np.asarray((lam_raw - rho * nu_raw * 1j * ctx.u) / ctx.alpha, dtype=np.complex128)
            _gam = float(nu_raw ** 2 / ctx.alpha)

            # "auto" is implicit-pi: it solves each step's quadratic exactly and
            # runs on any grid.  Adams (opt-in) is explicit in F, needs a
            # uniform grid, and can return NaN for H <= 0 or on coarse grids.
            _use_implicit_pi = direct_riccati_solver != "adams"

            if _use_implicit_pi:
                from .integration import ProductIntegrationRiccatiSolver
                solver = ProductIntegrationRiccatiSolver(
                    ctx.grid, alpha_anchor, _c_np, _beta_np, _gam,
                )
                f0_field = solver.solve()
            else:
                from .integration import AdamsFractionalRiccatiSolver
                solver = AdamsFractionalRiccatiSolver(
                    ctx.grid, alpha_anchor, _riccati_rhs, normalized=False,
                    F_args=(jnp.asarray(_c_np), jnp.asarray(_beta_np), 0.5 * _gam),
                )
                f0_field = solver.solve()

            f0_values = np.asarray(f0_field.values)
        elif base == "root_pade":
            from .pade import rough_heston_root_pade_f0
            f0_values = rough_heston_root_pade_f0(u=ctx.u, t=ctx.grid.t, param=self.param, degree=root_pade_degree,
                                                  H=anchor, tail_match=root_pade_tail_match)
        else:
            raise ValueError(
                f"base must be 'pade', 'root_pade', or 'direct_riccati', got {base!r}"
            )

        f0 = ctx.grid.field(f0_values, has_time_axis=True)
        K = _build_kernel_derivative_ops(ctx.grid, anchor_H=anchor, order=order)

        if order == 0:
            def solve_linearized(h_n: Field) -> Field:
                raise NotImplementedError("order >= 1 requires order >= 1 to be specified")

            return AnchorExpansionOps(anchor_H=anchor, f0=f0, K=K, solve_linearized=solve_linearized)

        Fx0 = -ctx.beta + ctx.gam * f0

        anchor_tol = 1e-10
        if abs(anchor - 0.5) < anchor_tol:
            # anchor=0.5 → K[0] is kernel='regular' (no weights_matrix).
            # Solve x = rhs + ∫_0^t Fx0(s)*x(s) ds by a forward left-rectangle
            # scan that avoids the dense Omega matrix. Uses actual panel widths
            # so nonuniform (graded) grids are handled correctly.
            _dt_arr = jnp.asarray(np.append(np.diff(ctx.grid.t), 0.0), dtype=jnp.float64)
            _Fx0 = jnp.asarray(Fx0.values, dtype=jnp.complex128)

            def solve_linearized(rhs: Field) -> Field:
                R = jnp.asarray(ctx.grid.field(rhs).values, dtype=jnp.complex128)
                return ctx.grid.field(_left_rectangle_scan(R, _dt_arr, _Fx0), has_time_axis=True)
        else:
            from .integration import LinearVolterraSolver
            solver = LinearVolterraSolver(ctx.grid, K[0], scale=1.0)

            def solve_linearized(rhs: Field) -> Field:
                return solver.solve(rhs, Fx0)

        return AnchorExpansionOps(anchor_H=anchor, f0=f0, K=K, solve_linearized=solve_linearized)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _run_expansion_recursion(
            self,
            ctx: RiccatiContext,
            ops: AnchorExpansionOps,
            target_H: float,
            order: int,
            return_lists: bool,
    ):
        f_list = [ops.f0]
        F_list = [ctx.F_ricc(ops.f0)]
        expansion = ops.f0

        for n in range(1, order + 1):
            h_n = ctx.grid.field(0.0)

            for k in range(1, n + 1):
                m = n - k
                h_n += comb(n, k) * (ops.K[k] @ F_list[m])

            if n >= 2:
                T_n = ctx.grid.field(0.0)
                for p in range(1, n):
                    T_n += comb(n, p) * (f_list[p] * f_list[n - p])
                h_n += ops.K[0] @ (0.5 * ctx.gam * T_n)

            f_n = ops.solve_linearized(h_n)
            f_list.append(f_n)
            cross = ctx.grid.field(0.0)

            for p in range(0, n + 1):
                cross += comb(n, p) * (f_list[p] * f_list[n - p])

            F_n = -ctx.beta * f_n + 0.5 * ctx.gam * cross
            F_list.append(F_n)
            expansion += (target_H - ops.anchor_H) ** n / factorial(n) * f_n

        f_out = f_list if return_lists else None
        F_out = F_list if return_lists else None
        return expansion, f_out, F_out

    def log_charfun_from_field(self, state: HestonExpansionState, field) -> np.ndarray:
        C = self._log_cf_integral(state.grid, state.F_ricc, field, state.target_H, state.anchor_H)
        return 1j * state.u * self.riskfree * state.grid.T + C

    def _compute_kappa_path_reflected(self, grid: TimeGrid, H: float):
        """Reflected variance path: evaluated at T-t instead of t."""
        t_ref = grid.field(np.maximum(grid.T - grid.t, 0.0), has_time_axis=True)
        return self.param.V0 + (self.param.theta / gamma(H + 1.5)) * np.power(t_ref, H + 0.5)

    def _log_cf_integral(self, grid: TimeGrid, F_ricc, field, H: float, anchor: float):
        """Integral part of the log CF of the Riccati solution ``field`` at target H (see log_cf_integrand)."""
        form = self.log_cf_integrand
        if form == "auto":
            form = "rhs" if abs(anchor - 0.5) < 1e-10 and abs(H - anchor) > 1e-10 else "solution"
        if form == "rhs":
            return Integrate(grid, start=0, stop="T") @ (
                    F_ricc(field) * self._compute_kappa_path_reflected(grid, H)
            )
        al = H + 0.5
        h = grid.field(field)
        if al < 1.0:
            frac = (Integrate(grid, kernel="frac", order=-al, start=0, stop="T") @ h) / gamma(1.0 - al)
        else:  # I^0 is the identity
            frac = h.values[..., -1]
        return self.param.theta * (Integrate(grid, start=0, stop="T") @ h) + self.param.V0 * frac

    def _log_charfun_raw(self, ctx: RiccatiContext, expansion_field, H: float) -> np.ndarray:
        """Log charfun given a pre-built context and the expansion field at target H."""
        C = self._log_cf_integral(ctx.grid, ctx.F_ricc, expansion_field, H, ctx.anchor_H)
        return 1j * ctx.u * self.riskfree * ctx.grid.T + C

    def _compute_expansion_for_H(self, f_list: list, anchor_H: float, H: float):
        """Taylor combination: f0 + (H−a)*f1/1! + (H−a)²*f2/2! + …"""
        result = f_list[0]
        for n in range(1, len(f_list)):
            result = result + (H - anchor_H) ** n / factorial(n) * f_list[n]
        return result

    def _build_h_independent_fields(
            self,
            arg: np.ndarray,
            maturity: float,
            time_steps: int,
            order: int,
            anchor_eff: float,
            base_eff: str,
            pade_order_eff: str,
            rpd_eff: int,
            rpso_eff,
            rptm_eff: bool = True,
            time_grid: str = "uniform",
            grid_power: float | None = None,
            direct_riccati_solver: str = "auto",
    ) -> tuple:
        """Build context, f_list and anchor ops independent of H (anchor_H used as placeholder for kappa_path).

        Callers must pass the target H to _log_charfun_raw.  While ``self._field_cache`` is a
        dict (during one ``eur_call_put`` call with ``accuracy_check``), results are reused,
        and a lower order is served from the leading coefficients of a higher one.
        """
        key = None
        if self._field_cache is not None:
            key = (np.asarray(arg, dtype=np.complex128).tobytes(), maturity, time_steps, anchor_eff,
                   base_eff, pade_order_eff, rpd_eff, rpso_eff, rptm_eff, time_grid, grid_power,
                   direct_riccati_solver)
            hit = self._field_cache.get(key)
            if hit is not None and len(hit[1]) > order:
                return hit[0], hit[1][:order + 1], hit[2]
        ctx = self._build_convention_context(
            arg, maturity, time_steps,
            anchor_H=anchor_eff, target_H=anchor_eff,
            time_grid=time_grid, grid_power=grid_power,
        )
        ops = self._build_anchor_ops(
            ctx, order, base_eff, anchor_eff,
            pade_order=pade_order_eff,
            root_pade_degree=rpd_eff,
            root_pade_series_order=rpso_eff,
            root_pade_tail_match=rptm_eff,
            direct_riccati_solver=direct_riccati_solver,
        )
        _, f_list, _ = self._run_expansion_recursion(
            ctx=ctx, ops=ops, target_H=anchor_eff, order=order, return_lists=True,
        )
        if key is not None:
            self._field_cache[key] = (ctx, f_list, ops)
        return ctx, f_list, ops

    def _expansion_fields(self, arg, maturity: float, time_steps: int, order: int, time_grid: str,
                          grid_power: float | None) -> tuple:
        """(ctx, f_list) of the H-expansion around param.H with this model's base settings."""
        ctx, f_list, _ = self._build_h_independent_fields(
            arg, maturity, time_steps, order + self._extra_order, self.param.H, self.base_solution, self.pade_order,
            self.root_pade_degree, self.root_pade_series_order, self.root_pade_tail_match,
            time_grid=time_grid, grid_power=grid_power, direct_riccati_solver=self.direct_riccati_solver,
        )
        return ctx, f_list[:order + 1]

    def _log_hN(self, ctx: RiccatiContext, f_list: list, anchor_H: float, H: float) -> np.ndarray:
        """log(phi_N / phi_0) on ctx's nodes: the H-expansion correction applied on top of phi_0."""
        exp_n = self._compute_expansion_for_H(f_list, anchor_H, H)
        return np.asarray(self._log_charfun_raw(ctx, exp_n, H) - self._log_charfun_raw(ctx, f_list[0], H))

    def log_charfun(
            self,
            arg: np.ndarray,
            maturity: float = 1.0,
            order: int = 0,
            time_steps: int = 400,
            H=None,
            time_grid: Literal["uniform", "graded", "symmetric_graded"] = "graded",
            grid_power: float | None = None,
    ) -> np.ndarray:
        """Log characteristic function E[exp(i u log(S_T/S_0))].

        Parameters
        ----------
        arg : np.ndarray
            Complex Fourier argument, shape ``(N_u,)``.
        maturity : float
            Option maturity T.
        order : int
            Taylor expansion order in H around the anchor ``self.param.H``.
        time_steps : int
            Number of time-grid nodes.  With the exact-Heston base the log CF
            at H = ½ is in closed form and does not use the grid.
        H : float, array-like, or None
            Target Hurst exponent(s).  ``None`` → ``self.param.H``.
            Passing an array evaluates all H values in a single Volterra solve.
        time_grid : {"uniform", "graded", "symmetric_graded"}
            Time discretisation; see :meth:`expansion_state`.
        grid_power : float or None
            Grading exponent; ``None`` auto-selects.

        Returns
        -------
        np.ndarray
            Shape ``(N_u, N_H)``, one column per target H; ``N_H = 1`` for a
            scalar or ``None`` H.
        """
        anchor_eff = self.param.H
        base_eff = self.base_solution
        po_eff = self.pade_order
        rpd_eff = self.root_pade_degree
        rpso_eff = self.root_pade_series_order
        rptm_eff = self.root_pade_tail_match

        if H is None:
            H_arr = np.array([self.param.H])
        else:
            H_arr = np.atleast_1d(np.asarray(H, dtype=float))

        key = None
        if self._field_cache is not None:   # the accuracy check reprices with the same phi0
            key = ("log_charfun", np.asarray(arg, dtype=np.complex128).tobytes(), maturity, order, time_steps,
                   H_arr.tobytes(), time_grid, grid_power, self.log_cf_integrand)
            if key in self._field_cache:
                return self._field_cache[key]
        ctx, f_list, ops = self._build_h_independent_fields(
            arg, maturity, time_steps, order,
            anchor_eff, base_eff, po_eff, rpd_eff, rpso_eff, rptm_eff,
            time_grid=time_grid, grid_power=grid_power,
            direct_riccati_solver=self.direct_riccati_solver,
        )
        cols = []
        for H_i in H_arr:
            if ops.log_phi_anchor is not None and H_i == anchor_eff:
                # At the anchor the expansion is f0, whose log CF the base knows in closed form.
                log_phi_i = ops.log_phi_anchor
            else:
                exp_i = self._compute_expansion_for_H(f_list, anchor_eff, float(H_i))
                log_phi_i = self._log_charfun_raw(ctx, exp_i, float(H_i))
            cols.append(np.asarray(log_phi_i).reshape(-1, 1))
        out = np.concatenate(cols, axis=1)  # (n_u, n_H)
        if key is not None:
            self._field_cache[key] = out
        return out

    def charfun(
            self,
            arg: np.ndarray,
            maturity: float = 1.0,
            order: int = 0,
            time_steps: int = 400,
            H=None,
            time_grid: Literal["uniform", "graded", "symmetric_graded"] = "graded",
            grid_power: float | None = None,
    ) -> np.ndarray:
        """Characteristic function.  See log_charfun for the H parameter."""
        return np.exp(self.log_charfun(
            arg=arg, maturity=maturity, order=order, time_steps=time_steps,
            H=H, time_grid=time_grid, grid_power=grid_power,
        ))

    def eur_call_put(
            self,
            # --- option ---
            K,
            maturity: float,
            S_0: float = 1.0,
            call: bool = True,
            digital: bool = False,
            implied_vol: bool = False,
            payoff_selection: str = "otm",
            # --- fourier ---
            R: float = 0.5,
            integration: Literal["panel", "adaptive_panel"] = "panel",
            L: float | None = None,
            panels=None,
            m_cheb: int = 16,
            q_weight: int = 128,
            time_steps_phi0: int | None = None,
            # --- expansion ---
            expansion_order: int = 0,
            time_steps: int = 400,
            H=None,
            # --- adaptive ---
            adaptive_metric: Literal["price", "iv"] = "iv",
            adaptive_tol: float = 1e-4,
            adaptive_abs_tol: float = 1e-8,
            adaptive_rel_tol: float = 1e-6,
            adaptive_L_candidates: list | None = None,
            adaptive_max_iter: int = 8,
            adaptive_verbose: bool = False,
            # --- time grid ---
            time_grid: Literal["uniform", "graded", "symmetric_graded"] = "graded",
            grid_power: float | None = None,
            # --- accuracy check ---
            accuracy_check: bool = True,
            accuracy_tol: float = 1e-4,
    ) -> np.ndarray:
        """Price European calls or puts via Fourier quadrature on the Heston measure.

        Parameters
        ----------
        K : array-like
            Strike prices.
        maturity : float
            Option maturity T > 0.
        S_0 : float
            Initial spot price.
        call : bool
            ``True`` returns call prices, ``False`` put prices.  With
            ``payoff_selection="otm"`` the out-of-the-money option is priced at
            each strike and converted by put–call parity; implied vols do not
            depend on ``call``.
        digital : bool
            Price digital (cash-or-nothing) payoffs instead of vanilla.
        implied_vol : bool
            If ``True``, return Black–Scholes implied volatilities instead of
            option prices.  Vanillas only.
        payoff_selection : {"otm", "as_requested"}
            ``"otm"`` prices calls for K ≥ F and puts for K < F, then applies
            put–call parity.  ``"as_requested"`` uses ``call`` for all strikes.
        R : float
            Fourier contour height.  The default ``R = 0.5`` prices every
            strike as a call and converts to puts by parity; for vanillas this
            is the Lewis contour.  It needs only E[S_T^0.5], which is always
            finite.  Other values use damped contours at ``+|R|`` for calls
            and ``-|R|`` for puts, which need E[S_T^|R|] and E[S_T^-|R|] to be
            finite.  Below ``|R| = 1`` vanilla calls get the residue S_0 of the
            payoff's pole at 1 added, as in the Lewis formula (half of it at
            1), and ``R = 0`` runs through the pole at 0, which adds half of
            its residue.  Near a pole the integrand has a peak of width
            |R - pole| at u = 0, which the panels resolve at 1e-2 from the
            pole (prices off by 4e-9) but not at 1e-3 (up to 2e-3 S_0); on
            the pole they are accurate.
        L : float or None
            Upper truncation of the Fourier integral.  ``None`` picks the point
            where |phi| has decayed to about 1e-10, from the model parameters,
            the maturity and every H priced (up to 256000).
        panels : array-like or None
            Explicit Chebyshev panel breakpoints.  Overrides ``L`` when given.
        m_cheb : int
            Chebyshev nodes per panel.
        q_weight : int
            Number of nodes for the quadrature weight pre-computation.
        time_steps_phi0 : int or None
            Time-grid nodes for the order-0 (phi0) solve.  ``None`` falls
            back to ``time_steps``.  With the exact-Heston base, phi0 at
            H = ½ is in closed form and does not use the grid.
        expansion_order : int
            Taylor expansion order N in H around ``self.param.H``.
        time_steps : int
            Time-grid nodes for the expansion solve.
        H : float, array-like, or None
            Target Hurst exponent(s).  ``None`` or a scalar uses ``self.param.H``
            or the given value and returns shape ``(N_K,)``.
            An array reuses the expensive Volterra fields across all H values
            and returns shape ``(N_K, N_H)``.
        integration : {"panel", "adaptive_panel"}
            ``"panel"`` uses fixed panels.  ``"adaptive_panel"`` iteratively
            widens ``L`` until prices or implied vols stabilise.
        adaptive_metric : {"price", "iv"}
            Convergence criterion for adaptive integration: the change of the
            out-of-the-money prices or of their implied vols.  Digitals always
            use ``"price"``.
        adaptive_tol : float
            Tolerance of the ``"iv"`` criterion, in implied vol.
        adaptive_abs_tol, adaptive_rel_tol : float
            Absolute and relative tolerances of the ``"price"`` criterion.
        adaptive_L_candidates : list or None
            Explicit sequence of ``L`` values to try.  ``None`` starts at the
            default ``L`` and widens through the panel-template endpoints.
        adaptive_max_iter : int
            Maximum number of adaptive refinement iterations.  When no two
            successive ``L`` agree, the prices at the last are returned with a
            warning.
        adaptive_verbose : bool
            Print convergence information at each adaptive step.
        time_grid : {"uniform", "graded", "symmetric_graded"}
            Time-node distribution; see :meth:`expansion_state`.
        grid_power : float or None
            Grading exponent; ``None`` auto-selects.
        accuracy_check : bool
            If ``True`` (the default), estimate the approximation error and emit
            a ``UserWarning`` when it exceeds ``accuracy_tol`` at some strike.
            For ``expansion_order = N >= 1`` the estimate is the change of the
            implied vols from order N to N+1 (the next term, computed along with
            the main pass) at the lowest and the highest target H.  For the
            Padé bases away from H₀ = ½ it is the difference between the two
            ``log_cf_integrand`` formulas at the anchor, which is about the
            error of ``"rhs"``; for root-Padé with ``"solution"``, which is
            several times more accurate, a third of it.  The returned values do
            not change.  Vanillas only; strikes whose out-of-the-money price is
            below 1e-6 S_0 are skipped.  The check adds about 10–35 % run time;
            pass ``False`` in loops that do not need it.
        accuracy_tol : float
            Warning threshold of the check, in implied volatility (1e-4 = 1 bp).

        Returns
        -------
        np.ndarray
            Option prices (or implied vols) of shape ``(N_K,)`` for scalar H,
            or ``(N_K, N_H)`` when H is an array.
        """
        _pricing_kw = dict(
            K=K, maturity=maturity, S_0=S_0, call=call, digital=digital,
            implied_vol=implied_vol, payoff_selection=payoff_selection,
            R=R, L=L, panels=panels, m_cheb=m_cheb, q_weight=q_weight,
            time_steps_phi0=time_steps_phi0, expansion_order=expansion_order,
            time_steps=time_steps, H=H,
            time_grid=time_grid, grid_power=grid_power,
        )
        if integration not in ("panel", "adaptive_panel"):
            raise ValueError(f"integration must be 'panel' or 'adaptive_panel', got {integration!r}")
        if digital and implied_vol:
            raise ValueError("implied_vol=True is for vanillas: the inversion uses the vanilla Black-Scholes price")

        def price(**kw):
            if integration == "panel":
                return self._eur_call_put_measure_quad_fixed_panel(**kw)[0]
            return self._eur_call_put_measure_quad_adaptive_panel(
                adaptive_metric=adaptive_metric,
                adaptive_tol=adaptive_tol,
                adaptive_abs_tol=adaptive_abs_tol,
                adaptive_rel_tol=adaptive_rel_tol,
                adaptive_L_candidates=adaptive_L_candidates,
                adaptive_max_iter=adaptive_max_iter,
                adaptive_verbose=adaptive_verbose,
                **kw,
            )

        if not accuracy_check or digital:
            return price(**_pricing_kw)
        # The checks reprice from the same expansion fields, so cache them for this call;
        # the expansion check needs one order more, which the main pass computes along.
        H_arr = np.atleast_1d(np.asarray(self.param.H if H is None else H, dtype=float))
        check_expansion = expansion_order >= 1 and bool(np.any(np.abs(H_arr - self.param.H) > 1e-12))
        self._field_cache = {}
        self._extra_order = int(check_expansion)
        try:
            result = price(**_pricing_kw)
            self._extra_order = 0
            self._warn_if_inaccurate(price, _pricing_kw, result, accuracy_tol, check_expansion)
        finally:
            self._field_cache = None
            self._extra_order = 0
        return result

    def _warn_if_inaccurate(self, price, kw: dict, result, tol: float, check_expansion: bool) -> None:
        """Warn when a cheap a-posteriori error estimate exceeds ``tol`` (implied vol) at some strike.

        The H-expansion: implied vols at orders N and N+1, since the next term
        (H - H0)^(N+1)/(N+1)! f_(N+1) estimates the truncation error of order N; it grows with
        |H - H0|, so the lowest and the highest target H bound it.  A Padé base: implied vols at
        the anchor with the two log_cf_integrand formulas.  Both are exact for the exact h, and
        "rhs" amplifies the error of an approximate h by about nu|u|, so they differ by about the
        error of "rhs".  Root-Padé with "solution" is typically 3-10 times more accurate than that
        difference, so a third of it is used there.  Strikes whose out-of-the-money price is below
        1e-6 S_0 are skipped.
        """
        from roughprix.fourier import _bs_price

        N = kw["expansion_order"]
        H_arr = np.atleast_1d(np.asarray(self.param.H if kw["H"] is None else kw["H"], dtype=float))
        iv_kw = dict(kw, implied_vol=True)
        main_iv = result if kw["implied_vol"] else None
        form = self.log_cf_integrand
        checks = []   # (pricing kw, its implied vols if known, the same with what changes, its form,
        #               share of the difference that estimates the error, what, advice)
        if check_expansion:
            # The next term grows like |H - H0|^(N+1), so the extreme targets are enough.  The
            # default Fourier cutoff depends only on them (the decay rate it uses is log-concave
            # in H), so these passes run on the main pass's nodes and reuse its cached fields.
            ext = [int(np.argmin(H_arr)), int(np.argmax(H_arr))]
            sub = H_arr.size > 2
            kw_a = dict(iv_kw, H=H_arr[ext]) if sub else iv_kw
            iv_a = np.asarray(main_iv)[:, ext] if sub and main_iv is not None else main_iv
            checks.append((kw_a, iv_a, dict(kw_a, expansion_order=N + 1), form, 1.0,
                           f"H-expansion, orders {N} and {N + 1}",
                           "raise expansion_order or anchor closer to the target H"))
        if self.base_solution in ("root_pade", "pade") and abs(self.param.H - 0.5) > 1e-10:
            at_main = N == 0 and H_arr.size == 1 and abs(H_arr[0] - self.param.H) < 1e-12
            kw_a = iv_kw if at_main else dict(iv_kw, H=None, expansion_order=0)
            share = 1 / 3 if self.base_solution == "root_pade" and form != "rhs" else 1.0
            checks.append((kw_a, main_iv if at_main else None, kw_a, "solution" if form == "rhs" else "rhs", share,
                           f"{self.base_solution} base at H = {self.param.H:g}, from the two log_cf_integrand formulas",
                           "base_solution='direct_riccati' is accurate"))
        K = np.asarray(kw["K"], dtype=float).reshape(-1)
        T, S_0 = kw["maturity"], kw["S_0"]
        F = S_0 * np.exp(self.riskfree * T)
        for kw_a, iv_a, kw_b, form_b, share, what, advice in checks:
            iv = np.asarray(price(**kw_a) if iv_a is None else iv_a, dtype=float)
            self.log_cf_integrand = form_b
            try:
                alt = np.asarray(price(**kw_b), dtype=float)
            finally:
                self.log_cf_integrand = form
            vol = np.where(np.isfinite(iv), iv, alt)   # where one pass failed, the other sets the price level
            Kb = K[:, None] if vol.ndim == 2 else K
            with np.errstate(invalid="ignore"):
                otm = np.where(Kb >= F, _bs_price(S_0, Kb, vol, T, self.riskfree, call=True),
                               _bs_price(S_0, Kb, vol, T, self.riskfree, call=False))
            diff = np.abs(alt - iv)[otm >= 1e-6 * S_0]
            est = share * np.max(np.where(np.isfinite(diff), diff, np.inf), initial=0.0)
            if est > tol:
                size = (f"up to {est * 1e4:.1f} bp in implied vol" if np.isfinite(est)
                        else "unbounded: non-finite implied vols")
                warnings.warn(f"{what}: estimated error {size} (accuracy_tol {tol * 1e4:.1f} bp); {advice}.",
                              stacklevel=3)

    def _eur_call_put_measure_quad_fixed_panel(
            self,
            # --- option ---
            K,
            maturity: float,
            S_0: float = 1.0,
            call: bool = True,
            digital: bool = False,
            implied_vol: bool = False,
            payoff_selection: str = "otm",
            # --- fourier ---
            R: float = 0.5,
            L: float | None = None,
            panels=None,
            m_cheb: int = 16,
            q_weight: int = 128,
            time_steps_phi0: int | None = None,
            # --- expansion ---
            expansion_order: int = 0,
            time_steps: int = 400,
            H=None,
            # --- incremental cache (supplied by adaptive loop) ---
            _phi0_cache: dict | None = None,
            _log_hN_cache: dict | None = None,
            # --- time grid ---
            time_grid: Literal["uniform", "graded", "symmetric_graded"] = "graded",
            grid_power: float | None = None,
    ) -> tuple:
        from .base_solution import (
            _as_strict_panels,
            _heuristic_measure_L, _heuristic_measure_panels,
        )
        from roughprix.fourier import (
            _iv_eur, build_panel_quadrature,
            measure_weights_from_phi, eur_call_put_from_weights, damped_contour_residues,
            lewis_weights_from_phi, eur_call_from_lewis_weights,
        )

        # --- resolve expansion base params from stored model values ---
        anchor_eff = self.param.H
        base_eff = self.base_solution
        po_eff = self.pade_order
        rpd_eff = self.root_pade_degree
        rpso_eff = self.root_pade_series_order
        rptm_eff = self.root_pade_tail_match

        if time_steps_phi0 is None:
            time_steps_phi0 = time_steps

        # --- resolve H ---
        scalar_H = H is None or np.ndim(H) == 0
        if H is None:
            H_arr = np.array([self.param.H])
        else:
            H_arr = np.atleast_1d(np.asarray(H, dtype=float))

        # R = 0.5 prices every strike as a call and gets puts by parity: on the
        # Lewis contour for vanillas, on the damped contour +0.5 for digitals.
        lewis_mode = (not digital) and np.isclose(R, 0.5)
        digital_parity = digital and np.isclose(R, 0.5)

        K = np.asarray(K, dtype=float).reshape(-1)
        D = np.exp(-self.riskfree * maturity)
        F = S_0 / D

        if panels is None:
            if L is None:
                L = _heuristic_measure_L(maturity, self.param, np.append(H_arr, self.param.H))
            panels_resolved = _heuristic_measure_panels(L)
        else:
            panels_resolved = _as_strict_panels(panels, L=L)
            L = float(panels_resolved[-1])

        if payoff_selection not in ("as_requested", "otm"):
            raise ValueError("payoff_selection must be 'as_requested' or 'otm'")
        if lewis_mode or digital_parity:
            # Price all strikes on a single contour as calls.
            call_mask = np.ones(len(K), dtype=bool)
            put_mask = np.zeros(len(K), dtype=bool)
            masks = [(call_mask, True)]
        elif payoff_selection == "as_requested":
            call_mask = np.ones(len(K), dtype=bool)
            put_mask = np.zeros(len(K), dtype=bool)
            masks = [(call_mask, call)]
        else:  # "otm"
            put_mask = K < F
            call_mask = ~put_mask
            masks = []
            if np.any(call_mask):
                masks.append((call_mask, True))
            if np.any(put_mask):
                masks.append((put_mask, False))

        quad = build_panel_quadrature(panels_resolved, m_cheb, q_weight)

        # For multiple H: precompute log phi0 for every H and the H-independent hN
        # fields per R-sign (or Lewis contour).
        _log_phi0: dict = {}  # R_sub / "lewis" -> log phi0 at u_quad, one column per H
        _hN_fields: dict = {}  # R_sub / "lewis" -> (ctx_n, f_list_n)
        if len(H_arr) > 1:
            if lewis_mode:
                _log_phi0["lewis"] = self.log_charfun(
                    quad.u_quad - 0.5j, maturity=maturity, order=0,
                    time_steps=time_steps_phi0, H=H_arr,
                    time_grid=time_grid, grid_power=grid_power,
                )
                _hN_fields["lewis"] = self._expansion_fields(
                    quad.u_nodes - 0.5j, maturity, time_steps, expansion_order, time_grid, grid_power)
            else:
                for mask, subset_call in masks:
                    R_sub = abs(R) if subset_call else -abs(R)
                    if R_sub not in _log_phi0:
                        _log_phi0[R_sub] = self.log_charfun(
                            -1j * (R_sub - 1j * quad.u_quad), maturity=maturity, order=0,
                            time_steps=time_steps_phi0, H=H_arr,
                            time_grid=time_grid, grid_power=grid_power,
                        )
                    if R_sub not in _hN_fields:
                        _hN_fields[R_sub] = self._expansion_fields(
                            -1j * (R_sub - 1j * quad.u_nodes), maturity, time_steps, expansion_order,
                            time_grid, grid_power)

        # ----------------------------------------------------------------
        # Evaluate for each H value
        # ----------------------------------------------------------------
        all_prices = []
        all_raw_otm_prices = []

        for i_H, H_i in enumerate(H_arr):
            H_i = float(H_i)
            otm_prices_i = np.zeros(len(K))
            otm_is_call_i = np.full(len(K), call)

            if len(H_arr) <= 1:
                if lewis_mode:
                    # Lewis: single contour arg = u - 0.5j, keyed as "lewis".
                    _lewis_key = (H_i, "lewis")
                    if _phi0_cache is not None and _lewis_key in _phi0_cache:
                        phi0_lewis = _phi0_cache[_lewis_key]
                    else:
                        log_phi0 = self.log_charfun(
                            quad.u_quad - 0.5j, maturity=maturity, order=0,
                            time_steps=time_steps_phi0, H=H_i,
                            time_grid=time_grid, grid_power=grid_power,
                        )
                        phi0_lewis = jnp.exp(jnp.asarray(log_phi0[:, 0], dtype=jnp.complex128))

                    if expansion_order == 0:
                        log_hN_lewis = np.zeros(len(quad.u_nodes))
                    elif _log_hN_cache is not None and _lewis_key in _log_hN_cache:
                        log_hN_lewis = _log_hN_cache[_lewis_key]
                    else:
                        ctx_n, f_list_n = self._expansion_fields(
                            quad.u_nodes - 0.5j, maturity, time_steps, expansion_order, time_grid, grid_power)
                        log_hN_lewis = self._log_hN(ctx_n, f_list_n, anchor_eff, H_i)
                else:
                    # --------------------------------------------------------
                    # Scalar H: pre-compute phi0 and log_hN for all R_subs
                    # in batched expansion_state calls.
                    #
                    # Optimisations:
                    #   • order=0  → hN=1 exactly; skip all hN evaluation.
                    #   • All R_subs batched into one log_charfun call for phi0.
                    #   • order>0  → all R_subs batched into one expansion_state
                    #                for hN (separate call, different order).
                    # --------------------------------------------------------
                    all_R_subs = list(dict.fromkeys(
                        abs(R) if sc else -abs(R) for _, sc in masks
                    ))
                    phi0_per_R: dict = {}
                    log_hN_per_R: dict = {}

                    R_need_phi0: list = []
                    R_need_hN: list = []
                    for R_sub in all_R_subs:
                        key = (H_i, R_sub)
                        if _phi0_cache is not None and key in _phi0_cache:
                            phi0_per_R[R_sub] = _phi0_cache[key]
                        else:
                            R_need_phi0.append(R_sub)
                        if expansion_order == 0:
                            log_hN_per_R[R_sub] = np.zeros(len(quad.u_nodes))
                        elif _log_hN_cache is not None and key in _log_hN_cache:
                            log_hN_per_R[R_sub] = _log_hN_cache[key]
                        else:
                            R_need_hN.append(R_sub)

                    # Batched phi0 at u_quad for all R_subs that need it
                    if R_need_phi0:
                        n_q = len(quad.u_quad)
                        arg_b = np.concatenate(
                            [-1j * (Rs - 1j * quad.u_quad) for Rs in R_need_phi0]
                        )
                        lp0 = self.log_charfun(
                            arg_b, maturity=maturity, order=0,
                            time_steps=time_steps_phi0, H=H_i,
                            time_grid=time_grid, grid_power=grid_power,
                        )[:, 0]
                        for i_r, R_sub in enumerate(R_need_phi0):
                            phi0_per_R[R_sub] = jnp.exp(jnp.asarray(
                                lp0[i_r * n_q:(i_r + 1) * n_q], dtype=jnp.complex128,
                            ))

                    # Batched log_hN at u_nodes for all R_subs that need it (order>0)
                    if R_need_hN:
                        n_n = len(quad.u_nodes)
                        arg_b = np.concatenate(
                            [-1j * (Rs - 1j * quad.u_nodes) for Rs in R_need_hN]
                        )
                        ctx_b, f_list_b = self._expansion_fields(
                            arg_b, maturity, time_steps, expansion_order, time_grid, grid_power)
                        log_hN_b = self._log_hN(ctx_b, f_list_b, anchor_eff, H_i)
                        for i_r, R_sub in enumerate(R_need_hN):
                            log_hN_per_R[R_sub] = log_hN_b[i_r * n_n:(i_r + 1) * n_n]

            # ---- Per-mask computation -----------------------------------
            _hN_cache_i: dict = {}  # used only by the multi-H path

            for mask, subset_call in masks:
                K_sub = K[mask]

                if lewis_mode:
                    # --- Lewis: single contour, all strikes as calls ---
                    if len(H_arr) > 1:
                        phi0_sub = jnp.exp(jnp.asarray(_log_phi0["lewis"][:, i_H], dtype=jnp.complex128))
                    else:
                        phi0_sub = phi0_lewis

                    w_jnp = lewis_weights_from_phi(phi0_sub, quad, K_sub, S_0)

                    if len(H_arr) > 1:
                        cache_key = ("lewis", quad.u_nodes.tobytes())
                        if cache_key not in _hN_cache_i:
                            log_hN_sub = self._log_hN(*_hN_fields["lewis"], anchor_eff, H_i)
                            hN_sub = np.exp(log_hN_sub)
                            _hN_cache_i[cache_key] = (log_hN_sub, hN_sub)
                        else:
                            log_hN_sub, hN_sub = _hN_cache_i[cache_key]
                    else:
                        log_hN_sub = log_hN_lewis
                        hN_sub = np.exp(log_hN_lewis)

                    sub_prices = np.asarray(
                        eur_call_from_lewis_weights(w_jnp, jnp.asarray(hN_sub), S_0, K_sub, D)
                    )
                    otm_prices_i[mask] = sub_prices
                    otm_is_call_i[mask] = True

                else:
                    R_sub = abs(R) if subset_call else -abs(R)

                    # --- phi0 ---
                    if len(H_arr) > 1:
                        phi0_sub = jnp.exp(jnp.asarray(_log_phi0[R_sub][:, i_H], dtype=jnp.complex128))
                    else:
                        phi0_sub = phi0_per_R[R_sub]

                    w_jnp = measure_weights_from_phi(
                        phi0_sub, quad, K_sub, R_sub, call=subset_call, digital=digital, S_0=S_0,
                    )

                    # --- log_hN ---
                    if len(H_arr) > 1:
                        cache_key = (float(R_sub), quad.u_nodes.tobytes())
                        if cache_key not in _hN_cache_i:
                            log_hN_sub = self._log_hN(*_hN_fields[R_sub], anchor_eff, H_i)
                            hN_sub = np.exp(log_hN_sub)
                            _hN_cache_i[cache_key] = (log_hN_sub, hN_sub)
                        else:
                            log_hN_sub, hN_sub = _hN_cache_i[cache_key]
                    else:
                        log_hN_sub = log_hN_per_R[R_sub]
                        hN_sub = np.exp(log_hN_sub)

                    sub_prices = np.asarray(
                        eur_call_put_from_weights(w_jnp, jnp.asarray(hN_sub), D)
                    ) + damped_contour_residues(K_sub, R_sub, S_0, D, call=subset_call, digital=digital)
                    otm_prices_i[mask] = sub_prices
                    otm_is_call_i[mask] = subset_call

            # capture raw OTM prices before parity correction and IV inversion
            all_raw_otm_prices.append(otm_prices_i.copy())

            # Put-call parity: C - P = S_0 - K D for vanillas, C + P = D for
            # (cash-or-nothing) digitals.
            if digital:
                def _put_to_call(p, k):
                    return D - p

                def _call_to_put(c, k):
                    return D - c
            else:
                def _put_to_call(p, k):
                    return p + S_0 - k * D

                def _call_to_put(c, k):
                    return c - S_0 + k * D

            if (lewis_mode or digital_parity) and payoff_selection == "as_requested" and not call:
                # All strikes were priced as calls; convert to the requested puts.
                prices_i = _call_to_put(otm_prices_i, K)
            elif payoff_selection == "as_requested":
                prices_i = otm_prices_i
            else:
                prices_i = np.empty_like(K)
                if call:
                    prices_i[call_mask] = otm_prices_i[call_mask]
                    prices_i[put_mask] = _put_to_call(otm_prices_i[put_mask], K[put_mask])
                else:
                    prices_i[put_mask] = otm_prices_i[put_mask]
                    prices_i[call_mask] = _call_to_put(otm_prices_i[call_mask], K[call_mask])

            if implied_vol:
                if payoff_selection == "as_requested":
                    payoff = "call" if call else "put"
                    result_i = _iv_eur(S_0=S_0, K=K, T=maturity,
                                       price=prices_i, payoff=payoff, r=self.riskfree)
                else:
                    ivs_i = np.empty_like(K)
                    if np.any(call_mask):
                        ivs_i[call_mask] = _iv_eur(
                            S_0=S_0, K=K[call_mask], T=maturity,
                            price=otm_prices_i[call_mask], payoff="call", r=self.riskfree,
                        )
                    if np.any(put_mask):
                        ivs_i[put_mask] = _iv_eur(
                            S_0=S_0, K=K[put_mask], T=maturity,
                            price=otm_prices_i[put_mask], payoff="put", r=self.riskfree,
                        )
                    result_i = ivs_i
            else:
                result_i = prices_i

            all_prices.append(result_i)

        # ----------------------------------------------------------------
        # Assemble output
        # ----------------------------------------------------------------
        raw_otm_prices = all_raw_otm_prices[0] if scalar_H else np.stack(all_raw_otm_prices, axis=-1)
        result = all_prices[0] if scalar_H else np.stack(all_prices, axis=-1)
        return result, raw_otm_prices, call_mask, put_mask

    def _eur_call_put_measure_quad_adaptive_panel(
            self,
            # --- adaptive params only; all pricing params arrive via **pricing_kw ---
            adaptive_metric: Literal["price", "iv"] = "iv",
            adaptive_tol: float = 1e-4,
            adaptive_abs_tol: float = 1e-8,
            adaptive_rel_tol: float = 1e-6,
            adaptive_L_candidates: list | None = None,
            adaptive_max_iter: int = 8,
            adaptive_verbose: bool = False,
            **pricing_kw,
    ) -> np.ndarray:
        from .base_solution import (
            _heuristic_measure_panels,
            _nested_measure_L_candidates, _validate_nested_measure_L_candidates,
        )
        from roughprix.fourier import (
            build_panel_quadrature, adaptive_check_values, adaptive_converged,
        )

        # Extract what the adaptive loop needs from pricing_kw.
        K = pricing_kw["K"]
        maturity = pricing_kw["maturity"]
        S_0 = pricing_kw.get("S_0", 1.0)
        call = pricing_kw.get("call", True)
        digital = pricing_kw.get("digital", False)
        payoff_selection = pricing_kw.get("payoff_selection", "otm")
        R = pricing_kw.get("R", 0.5)
        implied_vol = pricing_kw.get("implied_vol", False)
        expansion_order = pricing_kw.get("expansion_order", 0)
        time_steps = pricing_kw.get("time_steps", 400)
        time_steps_phi0 = pricing_kw.get("time_steps_phi0")
        time_grid = pricing_kw.get("time_grid", "graded")
        grid_power = pricing_kw.get("grid_power")
        H = pricing_kw.get("H")

        if pricing_kw.get("panels") is not None:
            raise ValueError(
                "adaptive_panel expects panels=None; explicit panels imply a fixed grid. "
                "Use integration='panel' to pass explicit panels."
            )
        if digital:   # digital prices have no vanilla implied vol to compare
            adaptive_metric = "price"

        K_arr = np.asarray(K, dtype=float).reshape(-1)
        D = np.exp(-self.riskfree * maturity)
        F = S_0 / D

        scalar_H = H is None or np.ndim(H) == 0
        H_arr_eff = np.array([self.param.H]) if H is None else np.atleast_1d(np.asarray(H, dtype=float))
        # Incremental caching only applies to the scalar-H path; multi-H uses
        # _build_h_independent_fields which can't be split across panels.
        use_incremental_cache = len(H_arr_eff) <= 1

        m_cheb = pricing_kw.get("m_cheb", 16)
        q_weight = pricing_kw.get("q_weight", 128)
        ts_phi0_eff = time_steps if time_steps_phi0 is None else time_steps_phi0

        lewis_mode = (not digital) and np.isclose(R, 0.5)
        digital_parity = digital and np.isclose(R, 0.5)

        # Pre-determine which contours will be used (mirrors mask logic in fixed-panel helper).
        if lewis_mode:
            R_subs = ["lewis"]
        elif digital_parity:
            R_subs = [abs(R)]
        elif payoff_selection == "as_requested":
            R_subs = [abs(R) if call else -abs(R)]
        else:
            R_subs = []
            if np.any(K_arr >= F):
                R_subs.append(abs(R))
            if np.any(K_arr < F):
                R_subs.append(-abs(R))
            R_subs = list(dict.fromkeys(R_subs))  # R = 0: one contour, as 0.0 == -0.0

        # Build L candidates — must be exact _MEASURE_PANEL_TEMPLATE endpoints to ensure
        # consecutive grids are nested (prefix property required by incremental cache).
        if adaptive_L_candidates is None:
            adaptive_L_candidates_used = _nested_measure_L_candidates(
                maturity, self.param, np.append(H_arr_eff, self.param.H)
            )
        else:
            adaptive_L_candidates_used = _validate_nested_measure_L_candidates(
                adaptive_L_candidates
            )

        L_seq = adaptive_L_candidates_used[:adaptive_max_iter]
        if not L_seq:
            raise ValueError("adaptive_panel needs at least one L candidate and adaptive_max_iter >= 1")

        # Incremental log-value cache: (H_i, R_sub) → concatenated log arrays
        _phi0_log_cache: dict = {}
        _log_hN_log_cache: dict = {}
        prev_n_quad = 0
        prev_n_nodes = 0
        prev_quad = None

        prev_check = None
        last_result = None

        for i, L_try in enumerate(L_seq):
            phi0_cache_iter: dict | None = None
            log_hN_cache_iter: dict | None = None

            if use_incremental_cache:
                panels_try = _heuristic_measure_panels(float(L_try))
                quad_try = build_panel_quadrature(panels_try, m_cheb, q_weight)

                if prev_quad is not None and not (
                        np.allclose(prev_quad.u_quad, quad_try.u_quad[:len(prev_quad.u_quad)])
                        and np.allclose(prev_quad.u_nodes, quad_try.u_nodes[:len(prev_quad.u_nodes)])
                ):
                    raise RuntimeError(
                        "Internal adaptive_panel cache error: quadrature grid is not nested. "
                        "Use nested L candidates or disable incremental caching."
                    )
                prev_quad = quad_try

                u_quad_new = quad_try.u_quad[prev_n_quad:]
                u_nodes_new = quad_try.u_nodes[prev_n_nodes:]

                phi0_cache_iter = {}
                log_hN_cache_iter = {}

                for H_i in H_arr_eff:
                    H_i = float(H_i)
                    for R_sub in R_subs:
                        key = (H_i, R_sub)  # R_sub is "lewis" or a float

                        # --- phi0 at new quadrature points ---
                        if len(u_quad_new) > 0:
                            if lewis_mode:
                                arg_new = u_quad_new - 0.5j
                            else:
                                arg_new = -1j * (R_sub - 1j * u_quad_new)
                            log_phi0_new = self.log_charfun(
                                arg_new, maturity=maturity, order=0,
                                time_steps=ts_phi0_eff, H=H_i,
                                time_grid=time_grid, grid_power=grid_power,
                            )[:, 0]
                            if key in _phi0_log_cache:
                                log_phi0_full = np.concatenate([_phi0_log_cache[key], log_phi0_new])
                            else:
                                log_phi0_full = log_phi0_new
                            _phi0_log_cache[key] = log_phi0_full
                            phi0_cache_iter[key] = jnp.exp(
                                jnp.asarray(log_phi0_full, dtype=jnp.complex128)
                            )

                        # --- log_hN at new Chebyshev nodes (order=0 → hN=1, skip) ---
                        if expansion_order > 0 and len(u_nodes_new) > 0:
                            if lewis_mode:
                                arg_n_new = u_nodes_new - 0.5j
                            else:
                                arg_n_new = -1j * (R_sub - 1j * u_nodes_new)
                            log_hN_new = self._log_hN(*self._expansion_fields(
                                arg_n_new, maturity, time_steps, expansion_order, time_grid, grid_power,
                            ), self.param.H, H_i)
                            if key in _log_hN_log_cache:
                                log_hN_full = np.concatenate([_log_hN_log_cache[key], log_hN_new])
                            else:
                                log_hN_full = log_hN_new
                            _log_hN_log_cache[key] = log_hN_full
                            log_hN_cache_iter[key] = log_hN_full

                prev_n_quad = len(quad_try.u_quad)
                prev_n_nodes = len(quad_try.u_nodes)

            result_try, raw_otm_try, call_mask_try, put_mask_try = (
                self._eur_call_put_measure_quad_fixed_panel(
                    L=float(L_try),
                    _phi0_cache=phi0_cache_iter,
                    _log_hN_cache=log_hN_cache_iter,
                    **{k: v for k, v in pricing_kw.items() if k not in ("panels", "L")},
                )
            )
            check = adaptive_check_values(
                result_try, raw_otm_try, call_mask_try, put_mask_try,
                implied_vol, adaptive_metric, K_arr, maturity, S_0,
                r=self.riskfree, scalar_H=scalar_H,
            )

            if prev_check is not None:
                if adaptive_verbose:
                    err = float(np.max(np.abs(check - prev_check)))
                    print(f"[adaptive_panel] iter={i}, L={L_try}, err={err:.3e}")

                if adaptive_converged(
                    prev_check, check, adaptive_metric,
                    adaptive_tol, adaptive_abs_tol, adaptive_rel_tol,
                ):
                    return result_try
            else:
                if adaptive_verbose:
                    print(f"[adaptive_panel] iter={i}, L={L_try} (first)")

            prev_check = check
            last_result = result_try

        warnings.warn(
            f"adaptive_panel did not converge: no two successive L up to {L_seq[-1]:g} agreed within "
            f"the tolerance; returning the prices at L = {L_seq[-1]:g} (adaptive_verbose=True shows "
            "the differences).",
            stacklevel=4,
        )
        return last_result
