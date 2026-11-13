"""Tests for rough_heston_pade.py: h_pade22, h_pade33, h_pade44 and rough_heston_pade_f0."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from roughprix.heston.pade import (
    h_pade22, h_pade33, h_pade44, rough_heston_pade_f0,
    rough_heston_root_pade_f0,
)
from roughprix.heston.rough_heston import RoughHestonParam


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_param():
    return RoughHestonParam()  # lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=0.4


@pytest.fixture
def u_typical():
    return np.array([0.5 + 0.1j, 1.0 + 0.5j, 2.0 + 1.0j, 5.0 + 2.0j])


@pytest.fixture
def t_grid():
    return np.linspace(0.0, 2.0, 21)


@pytest.fixture
def rp_param():
    return RoughHestonParam(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=0.1)


@pytest.fixture
def rp_u():
    return np.array([0.5 + 0.1j, 1.0 - 0.3j, 3.0 + 0.5j])


@pytest.fixture
def rp_t():
    return np.linspace(0.0, 0.5, 11)


# ---------------------------------------------------------------------------
# A. h_pade22 direct
# ---------------------------------------------------------------------------

class TestHPade22Shape:
    def test_output_shape(self, u_typical, t_grid):
        result = h_pade22(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (len(u_typical), len(t_grid))

    def test_output_shape_single_u(self, t_grid):
        result = h_pade22(np.array([1.0 + 0.5j]), t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (1, len(t_grid))

    def test_output_shape_single_t(self, u_typical):
        result = h_pade22(u_typical, np.array([1.0]), H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (len(u_typical), 1)

    def test_output_shape_scalar_inputs(self):
        result = h_pade22(np.array([1.0 + 0.5j]), np.array([1.0]), H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (1, 1)


class TestHPade22Dtype:
    def test_output_is_complex(self, u_typical, t_grid):
        result = h_pade22(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert np.iscomplexobj(result)
        assert result.dtype == np.complex128


class TestHPade22AtZero:
    def test_zero_at_t_zero(self, u_typical):
        tau = np.array([0.0])
        result = h_pade22(u_typical, tau, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        np.testing.assert_array_equal(result[:, 0], 0.0 + 0.0j)

    def test_zero_at_t_zero_in_grid(self, u_typical, t_grid):
        result = h_pade22(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        # t_grid[0] == 0.0
        np.testing.assert_array_equal(result[:, 0], 0.0 + 0.0j)


class TestHPade22Finite:
    @pytest.mark.parametrize("u_val", [
        0.5 + 0.1j,
        1.0 + 0.5j,
        2.0 + 1.0j,
        5.0 + 2.0j,
        10.0 + 3.0j,
    ])
    def test_no_nan_or_inf(self, u_val, t_grid):
        result = h_pade22(np.array([u_val]), t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert np.all(np.isfinite(result)), f"Non-finite values for u={u_val}"

    def test_vectorized_multiple_u_and_t(self):
        u = np.array([0.5 + 0.2j, 1.5 + 0.8j, 3.0 + 1.5j])
        t = np.linspace(0.0, 1.0, 11)
        result = h_pade22(u, t, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (3, 11)
        assert np.all(np.isfinite(result))


class TestHPade22Parameters:
    @pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.5])
    def test_various_rho(self, u_typical, t_grid, rho):
        result = h_pade22(u_typical, t_grid, H=0.0, rho=rho, nu=0.3, lam=0.3)
        assert result.shape == (len(u_typical), len(t_grid))
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("nu", [0.1, 0.3, 0.5, 1.0])
    def test_various_nu(self, u_typical, t_grid, nu):
        result = h_pade22(u_typical, t_grid, H=0.0, rho=-0.7, nu=nu, lam=0.3)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("lam", [0.1, 0.3, 0.5, 1.0])
    def test_various_lam(self, u_typical, t_grid, lam):
        result = h_pade22(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=lam)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# B. rough_heston_pade_f0 convenience wrapper
# ---------------------------------------------------------------------------

class TestRoughHestonPadeF0:
    def test_shape(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param)
        assert result.shape == (len(u_typical), len(t_grid))

    def test_complex_dtype(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param)
        assert np.iscomplexobj(result)

    def test_zero_at_t_zero(self, default_param, u_typical):
        result = rough_heston_pade_f0(u_typical, np.array([0.0]), default_param)
        np.testing.assert_array_equal(result[:, 0], 0.0 + 0.0j)

    def test_no_nan_or_inf(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param)
        assert np.all(np.isfinite(result))

    def test_unsupported_pade_order_raises(self, default_param, u_typical, t_grid):
        with pytest.raises(NotImplementedError, match="Padé order"):
            rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="55")

    def test_nonzero_H_anchor_runs(self, default_param, u_typical, t_grid):
        """rough_heston_pade_f0 supports any H_anchor via h_padeXX."""
        result = rough_heston_pade_f0(u_typical, t_grid, default_param, H_anchor=0.5)
        assert result.shape == (len(u_typical), len(t_grid))
        assert np.all(np.isfinite(result))

    def test_default_pade_order_is_22(self, default_param, u_typical, t_grid):
        # Should not raise
        result = rough_heston_pade_f0(u_typical, t_grid, default_param)
        assert result.shape == (len(u_typical), len(t_grid))

    def test_matches_h_pade22_directly(self, default_param, u_typical, t_grid):
        result_f0 = rough_heston_pade_f0(u_typical, t_grid, default_param)
        result_direct = h_pade22(
            u_typical, t_grid,
            H=0.0,
            rho=default_param.rho,
            nu=default_param.nu,
            lam=default_param.lam,
        )
        np.testing.assert_array_equal(result_f0, result_direct)


# ---------------------------------------------------------------------------
# C. h_pade44 direct
# ---------------------------------------------------------------------------

class TestHPade44Shape:
    def test_output_shape(self, u_typical, t_grid):
        result = h_pade44(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (len(u_typical), len(t_grid))

    def test_output_shape_single_u(self, t_grid):
        result = h_pade44(np.array([1.0 + 0.5j]), t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (1, len(t_grid))

    def test_output_shape_single_t(self, u_typical):
        result = h_pade44(u_typical, np.array([1.0]), H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (len(u_typical), 1)


class TestHPade44Dtype:
    def test_output_is_complex(self, u_typical, t_grid):
        result = h_pade44(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert np.iscomplexobj(result)
        assert result.dtype == np.complex128


class TestHPade44AtZero:
    def test_zero_at_t_zero(self, u_typical):
        tau = np.array([0.0])
        result = h_pade44(u_typical, tau, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        np.testing.assert_array_almost_equal(np.abs(result[:, 0]), 0.0, decimal=14)

    def test_zero_at_t_zero_in_grid(self, u_typical, t_grid):
        result = h_pade44(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        np.testing.assert_array_almost_equal(np.abs(result[:, 0]), 0.0, decimal=14)


class TestHPade44Finite:
    @pytest.mark.parametrize("u_val", [
        0.5 + 0.1j,
        1.0 + 0.5j,
        2.0 + 1.0j,
        5.0 + 2.0j,
        10.0 + 3.0j,
    ])
    def test_no_nan_or_inf(self, u_val, t_grid):
        result = h_pade44(np.array([u_val]), t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert np.all(np.isfinite(result)), f"Non-finite values for u={u_val}"

    def test_vectorized_multiple_u_and_t(self):
        u = np.array([0.5 + 0.2j, 1.5 + 0.8j, 3.0 + 1.5j])
        t = np.linspace(0.0, 1.0, 11)
        result = h_pade44(u, t, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (3, 11)
        assert np.all(np.isfinite(result))


class TestHPade44NotIdenticalToPade22:
    def test_pade44_differs_from_pade22(self, u_typical, t_grid):
        r22 = h_pade22(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        r44 = h_pade44(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert not np.allclose(r22, r44), "h_pade44 must differ from h_pade22"


# ---------------------------------------------------------------------------
# D. rough_heston_pade_f0 with pade_order="44"
# ---------------------------------------------------------------------------

class TestRoughHestonPadeF0Pade44:
    def test_shape(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="44")
        assert result.shape == (len(u_typical), len(t_grid))

    def test_complex_dtype(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="44")
        assert np.iscomplexobj(result)

    def test_zero_at_t_zero(self, default_param, u_typical):
        result = rough_heston_pade_f0(u_typical, np.array([0.0]), default_param, pade_order="44")
        np.testing.assert_array_almost_equal(np.abs(result[:, 0]), 0.0, decimal=14)

    def test_no_nan_or_inf(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="44")
        assert np.all(np.isfinite(result))

    def test_matches_h_pade44_directly(self, default_param, u_typical, t_grid):
        result_f0 = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="44")
        result_direct = h_pade44(
            u_typical, t_grid,
            H=0.0,
            rho=default_param.rho,
            nu=default_param.nu,
            lam=default_param.lam,
        )
        np.testing.assert_array_equal(result_f0, result_direct)

    def test_unsupported_order_bad_raises(self, default_param, u_typical, t_grid):
        with pytest.raises(NotImplementedError, match="Padé order"):
            rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="bad")


# ---------------------------------------------------------------------------
# E. h_pade33 direct
# ---------------------------------------------------------------------------

class TestHPade33Shape:
    def test_output_shape(self, u_typical, t_grid):
        result = h_pade33(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (len(u_typical), len(t_grid))

    def test_output_shape_single_u(self, t_grid):
        result = h_pade33(np.array([1.0 + 0.5j]), t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (1, len(t_grid))

    def test_output_shape_single_t(self, u_typical):
        result = h_pade33(u_typical, np.array([1.0]), H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (len(u_typical), 1)


class TestHPade33Dtype:
    def test_output_is_complex(self, u_typical, t_grid):
        result = h_pade33(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert np.iscomplexobj(result)
        assert result.dtype == np.complex128


class TestHPade33AtZero:
    def test_zero_at_t_zero(self, u_typical):
        tau = np.array([0.0])
        result = h_pade33(u_typical, tau, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        np.testing.assert_array_almost_equal(np.abs(result[:, 0]), 0.0, decimal=14)

    def test_zero_at_t_zero_in_grid(self, u_typical, t_grid):
        result = h_pade33(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        np.testing.assert_array_almost_equal(np.abs(result[:, 0]), 0.0, decimal=14)


class TestHPade33Finite:
    @pytest.mark.parametrize("u_val", [
        0.5 + 0.1j,
        1.0 + 0.5j,
        2.0 + 1.0j,
        5.0 + 2.0j,
        10.0 + 3.0j,
    ])
    def test_no_nan_or_inf(self, u_val, t_grid):
        result = h_pade33(np.array([u_val]), t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert np.all(np.isfinite(result)), f"Non-finite values for u={u_val}"

    def test_vectorized_multiple_u_and_t(self):
        u = np.array([0.5 + 0.2j, 1.5 + 0.8j, 3.0 + 1.5j])
        t = np.linspace(0.0, 1.0, 11)
        result = h_pade33(u, t, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (3, 11)
        assert np.all(np.isfinite(result))

    def test_regression_representative_inputs(self):
        u = np.array([0.2 + 0.1j, 1.0 - 0.5j, -10.0 - 1.2j])
        t = np.array([0.0, 0.01, 0.25, 1.0])
        result = h_pade33(u, t, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert result.shape == (3, 4)
        assert np.all(np.isfinite(result))
        # t=0 column must be zero
        np.testing.assert_array_almost_equal(np.abs(result[:, 0]), 0.0, decimal=14)


class TestHPade33NotIdentical:
    def test_pade33_differs_from_pade22(self, u_typical, t_grid):
        r22 = h_pade22(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        r33 = h_pade33(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert not np.allclose(r22, r33), "h_pade33 must differ from h_pade22"

    def test_pade33_differs_from_pade44(self, u_typical, t_grid):
        r33 = h_pade33(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        r44 = h_pade44(u_typical, t_grid, H=0.0, rho=-0.7, nu=0.3, lam=0.3)
        assert not np.allclose(r33, r44), "h_pade33 must differ from h_pade44"


# ---------------------------------------------------------------------------
# F. rough_heston_pade_f0 with pade_order="33"
# ---------------------------------------------------------------------------

class TestRoughHestonPadeF0Pade33:
    def test_shape(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="33")
        assert result.shape == (len(u_typical), len(t_grid))

    def test_complex_dtype(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="33")
        assert np.iscomplexobj(result)

    def test_zero_at_t_zero(self, default_param, u_typical):
        result = rough_heston_pade_f0(u_typical, np.array([0.0]), default_param, pade_order="33")
        np.testing.assert_array_almost_equal(np.abs(result[:, 0]), 0.0, decimal=14)

    def test_no_nan_or_inf(self, default_param, u_typical, t_grid):
        result = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="33")
        assert np.all(np.isfinite(result))

    def test_matches_h_pade33_directly(self, default_param, u_typical, t_grid):
        result_f0 = rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="33")
        result_direct = h_pade33(
            u_typical, t_grid,
            H=0.0,
            rho=default_param.rho,
            nu=default_param.nu,
            lam=default_param.lam,
        )
        np.testing.assert_array_equal(result_f0, result_direct)

    def test_unsupported_55_raises(self, default_param, u_typical, t_grid):
        with pytest.raises(NotImplementedError, match="Padé order"):  # noqa: keep remainder
            rough_heston_pade_f0(u_typical, t_grid, default_param, pade_order="55")


# ---------------------------------------------------------------------------
# G. driver_scale tests
# ---------------------------------------------------------------------------

_KW_BASE = dict(H=0.0, rho=-0.7, nu=0.3, lam=0.3)
_U_DS = np.array([0.2 + 0.1j, 1.0 - 0.5j, 3.0 + 0.8j])
_TAU_DS = np.linspace(0.0, 1.0, 21)





class TestRootPadeShape:
    def test_shape_default(self, rp_param, rp_u, rp_t):
        out = rough_heston_root_pade_f0(rp_u, rp_t, rp_param)
        assert out.shape == (len(rp_u), len(rp_t))

    def test_shape_single_u(self, rp_param, rp_t):
        out = rough_heston_root_pade_f0(np.array([1.0 + 0.5j]), rp_t, rp_param)
        assert out.shape == (1, len(rp_t))

    def test_shape_single_t(self, rp_param, rp_u):
        out = rough_heston_root_pade_f0(rp_u, np.array([1.0]), rp_param)
        assert out.shape == (len(rp_u), 1)

    def test_dtype_complex(self, rp_param, rp_u, rp_t):
        out = rough_heston_root_pade_f0(rp_u, rp_t, rp_param)
        assert np.iscomplexobj(out)


class TestRootPadeAtZero:
    def test_zero_at_t0(self, rp_param, rp_u):
        t = np.array([0.0, 0.5, 1.0])
        out = rough_heston_root_pade_f0(rp_u, t, rp_param)
        np.testing.assert_array_equal(out[:, 0], 0.0)

    def test_zero_at_t0_only_zero(self, rp_param, rp_u):
        out = rough_heston_root_pade_f0(rp_u, np.array([0.0]), rp_param)
        np.testing.assert_array_equal(out, 0.0)


class TestRootPadeFinite:
    def test_all_finite_default(self, rp_param, rp_u, rp_t):
        out = rough_heston_root_pade_f0(rp_u, rp_t, rp_param)
        assert np.all(np.isfinite(out)), "Non-finite values in root_pade output"

    def test_all_finite_various_degrees(self, rp_param, rp_u, rp_t):
        for degree in [4, 5, 6, 7]:
            out = rough_heston_root_pade_f0(rp_u, rp_t, rp_param, degree=degree)
            assert np.all(np.isfinite(out)), f"Non-finite values for degree={degree}"

    def test_all_finite_moderate_u(self, rp_param):
        u_mod = np.linspace(0.1, 10.0, 20) + 0.5j
        out = rough_heston_root_pade_f0(u_mod, np.linspace(0.0, 0.5, 11), rp_param)
        assert np.all(np.isfinite(out))


class TestRootPadeHAnchorError:
    def test_raises_for_out_of_range_h_anchor(self, rp_param, rp_u, rp_t):
        with pytest.raises(ValueError):
            rough_heston_root_pade_f0(rp_u, rp_t, rp_param, H=0.6)

    def test_raises_for_lower_bound_h_anchor(self, rp_param, rp_u, rp_t):
        with pytest.raises(ValueError):
            rough_heston_root_pade_f0(rp_u, rp_t, rp_param, H=-0.5)




class TestRootPadeVsDirectRiccati:
    """root_pade and direct_riccati must agree qualitatively on moderate contours."""

    def test_relative_mode_agreement(self):
        from roughprix.heston.rough_heston import RoughHeston, RoughHestonParam

        m = RoughHeston(
            RoughHestonParam(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=0.1),
            riskfree=0.0,
        )
        arg = np.array([0.5 + 0.1j, 1.0 - 0.3j])
        kw = dict(maturity=0.5, order=0, time_steps=100, return_lists=True)

        st_rp = m.expansion_state(arg, base="root_pade", **kw)  # no gamma_mode
        st_dr = m.expansion_state(arg, base="direct_riccati", **kw)

        f0_rp = np.array(st_rp.f0.values)
        f0_dr = np.array(st_dr.f0.values)

        rel_diff = np.max(np.abs(f0_rp - f0_dr) / np.maximum(np.abs(f0_dr), 1e-12))
        assert rel_diff < 0.3, (
            f"root_pade and direct_riccati differ too much: {rel_diff:.3f}"
        )


class TestRootPadeWiring:
    """root_pade is accepted by expansion_state and produces finite results."""

    def test_expansion_state_runs(self):
        from roughprix.heston.rough_heston import RoughHeston, RoughHestonParam

        m = RoughHeston(RoughHestonParam(), riskfree=0.0)
        arg = np.array([0.5 + 0.1j, 5.0 + 0.5j, 20.0 + 0.5j])
        state = m.expansion_state(
            arg, maturity=0.5, order=0, time_steps=50, base="root_pade",
        )
        f0 = np.array(state.f0.values)
        assert f0.shape[0] == len(arg)
        assert np.all(np.isfinite(f0))

    def test_validation_rejects_unknown(self):
        from roughprix.heston.rough_heston import RoughHeston, RoughHestonParam

        m = RoughHeston(RoughHestonParam(), riskfree=0.0)
        with pytest.raises(ValueError):
            m.expansion_state(
                np.array([1.0 + 0.5j]), maturity=1.0, order=0, base="bad_method",
            )

    def test_log_charfun_runs(self):
        from roughprix.heston.rough_heston import RoughHeston, RoughHestonParam

        m = RoughHeston(RoughHestonParam(), riskfree=0.0, base_solution="root_pade")
        arg = np.array([0.5 + 0.1j, 5.0 + 0.5j])
        log_cf = m.log_charfun(
            arg, maturity=0.5, order=0, time_steps=50,
        )
        assert log_cf.shape == (len(arg), 1)
        assert np.all(np.isfinite(log_cf))

    def test_frozen_mode_runs(self):
        from roughprix.heston.rough_heston import RoughHeston, RoughHestonParam

        m = RoughHeston(
            RoughHestonParam(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=0.2),
            riskfree=0.0,
        )
        arg = np.array([1.0 - 0.3j, 5.0 + 0.5j])
        state = m.expansion_state(
            arg, maturity=0.5, order=0, time_steps=50, base="root_pade",
        )
        f0 = np.array(state.f0.values)
        assert np.all(np.isfinite(f0))

    def test_various_pade_degrees_run(self):
        from roughprix.heston.rough_heston import RoughHeston, RoughHestonParam

        m = RoughHeston(RoughHestonParam(), riskfree=0.0)
        arg = np.array([1.0 + 0.5j, 3.0 + 0.5j])
        for degree in [4, 5, 6, 7]:
            state = m.expansion_state(
                arg, maturity=0.5, order=0, time_steps=50, base="root_pade",
                root_pade_degree=degree,
            )
            f0 = np.array(state.f0.values)
            assert np.all(np.isfinite(f0)), f"Non-finite for degree={degree}"


# ===========================================================================
# Section K: rough_heston_root_pade_f0 — general H_anchor
# ===========================================================================

class TestRootPadeHAnchorGeneral:
    """Tests for generalised H_anchor support in rough_heston_root_pade_f0."""

    @pytest.fixture
    def param(self):
        return RoughHestonParam(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=0.1)

    @pytest.fixture
    def u_small(self):
        return np.array([0.5 + 0.1j, 1.0 - 0.3j, 3.0 + 0.5j])

    @pytest.fixture
    def t_short(self):
        return np.linspace(0.0, 0.5, 11)

    def test_h_anchor_zero_matches_default(self, param, u_small, t_short):
        """H_anchor=0.0 must give the same output as the default call."""
        out_default = rough_heston_root_pade_f0(u_small, t_short, param)
        out_explicit = rough_heston_root_pade_f0(u_small, t_short, param, H=0.0)
        np.testing.assert_array_equal(out_default, out_explicit)

    def test_h_anchor_025_finite(self, param, u_small, t_short):
        """H_anchor=0.25 (alpha=0.75) must return all finite values."""
        out = rough_heston_root_pade_f0(u_small, t_short, param, H=0.25)
        assert out.shape == (len(u_small), len(t_short))
        assert np.iscomplexobj(out)
        assert np.all(np.isfinite(out)), "Non-finite values for H_anchor=0.25"

    def test_h_anchor_05_finite(self, param, u_small, t_short):
        """H_anchor=0.5 (alpha=1.0, classical Heston) must return all finite values."""
        out = rough_heston_root_pade_f0(u_small, t_short, param, H=0.5)
        assert out.shape == (len(u_small), len(t_short))
        assert np.all(np.isfinite(out)), "Non-finite values for H_anchor=0.5"

    def test_h_anchor_05_close_to_pade44(self, param, u_small):
        """H_anchor=0.5 root_pade should agree with h_pade44(H=0.5) for small maturity."""
        from roughprix.heston.pade import h_pade44
        t_tiny = np.linspace(0.0, 0.1, 6)
        out_rp = rough_heston_root_pade_f0(u_small, t_tiny, param, H=0.5)
        out_p4 = h_pade44(
            a=u_small,
            tau=t_tiny,
            H=0.5,
            rho=param.rho,
            nu=param.nu,
            lam=param.lam,
        )
        # Skip t=0 (both are zero there)
        rel_diff = np.max(
            np.abs(out_rp[:, 1:] - out_p4[:, 1:])
            / np.maximum(np.abs(out_p4[:, 1:]), 1e-12)
        )
        assert rel_diff < 0.5, (
            f"H_anchor=0.5 root_pade deviates too much from h_pade44: {rel_diff:.3f}"
        )

    def test_h_anchor_zero_at_t0(self, param, u_small):
        """h(t=0) == 0 for all H_anchor values."""
        for h_anc in [0.0, 0.25, 0.5]:
            out = rough_heston_root_pade_f0(u_small, np.array([0.0]), param, H=h_anc)
            np.testing.assert_array_equal(
                out, 0.0, err_msg=f"Non-zero at t=0 for H_anchor={h_anc}"
            )

    def test_diagnostics_report_h_anchor_and_alpha(self, param, u_small, t_short):
        """rough_heston_root_pade_f0 must return finite arrays for various H anchors."""
        for h_anc in [0.0, 0.25, 0.5]:
            result = rough_heston_root_pade_f0(u_small, t_short, param, H=h_anc)
            assert isinstance(result, np.ndarray)
            assert np.all(np.isfinite(result))


    def test_different_h_anchors_give_different_outputs(self, param, u_small):
        """Different H_anchor values must produce genuinely different results (t > 0)."""
        t_pos = np.array([0.1, 0.5])
        out_0 = rough_heston_root_pade_f0(u_small, t_pos, param, H=0.0)
        out_q = rough_heston_root_pade_f0(u_small, t_pos, param, H=0.25)
        out_h = rough_heston_root_pade_f0(u_small, t_pos, param, H=0.5)
        assert not np.allclose(out_0, out_q), "H_anchor=0.0 and 0.25 unexpectedly equal"
        assert not np.allclose(out_q, out_h), "H_anchor=0.25 and 0.5 unexpectedly equal"


# ===========================================================================
# Section L: a = 0 and a = -i, where c = -a(a+i)/2 vanishes
# ===========================================================================
# h = 0 solves the Riccati equation exactly at these two nodes, which gives
# phi(0) = 1 and phi(-i) = E[S_T / S_0] = e^{rT}.  The Padé coefficients are
# 0/0 there, so the approximants return h = 0 at these nodes directly and
# solve for the other nodes of the batch alone.

_C0_SETS = {
    "A": dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02),
    # Steep skew without mean reversion; here aa = 0 at a = 0 as well.
    "C": dict(lam=0.0, theta=0.0, nu=0.5, rho=-0.9, V0=0.04),
}
_C0_T = np.linspace(0.0, 1.0, 11)
_C0_METHODS = {
    "pade22": lambda a, p: h_pade22(a, _C0_T, H=p.H, rho=p.rho, nu=p.nu, lam=p.lam),
    "pade33": lambda a, p: h_pade33(a, _C0_T, H=p.H, rho=p.rho, nu=p.nu, lam=p.lam),
    "pade44": lambda a, p: h_pade44(a, _C0_T, H=p.H, rho=p.rho, nu=p.nu, lam=p.lam),
    "root_pade": lambda a, p: rough_heston_root_pade_f0(a, _C0_T, p, H=p.H, tail_match=False),
    "root_pade_tail": lambda a, p: rough_heston_root_pade_f0(a, _C0_T, p, H=p.H, tail_match=True),
}


@pytest.mark.parametrize("set_name", sorted(_C0_SETS))
@pytest.mark.parametrize("method", sorted(_C0_METHODS))
def test_pade_zero_where_c_vanishes(method, set_name):
    h_fn = _C0_METHODS[method]
    param = RoughHestonParam(**_C0_SETS[set_name], H=0.1)
    # Lewis and damped-contour nodes, with a = 0 and a = -i in between.
    a = np.array([0.5 - 0.5j, 0.0, 3.0 - 0.5j, -1j, -2.0 - 1.2j])
    trivial = np.array([False, True, False, True, False])

    h = h_fn(a, param)
    assert h.shape == (len(a), len(_C0_T))
    np.testing.assert_array_equal(h[trivial], 0.0)
    # The other nodes come out as if a = 0 and a = -i were not in the batch.
    h_regular = h_fn(a[~trivial], param)
    assert np.all(np.isfinite(h_regular))
    np.testing.assert_array_equal(h[~trivial], h_regular)
    np.testing.assert_array_equal(h_fn(a[trivial], param), np.zeros((2, len(_C0_T))))


@pytest.mark.parametrize("set_name", sorted(_C0_SETS))
@pytest.mark.parametrize("base_kw", [
    dict(base_solution="root_pade"),
    dict(base_solution="pade", pade_order="22"),
    dict(base_solution="pade", pade_order="33"),
    dict(base_solution="pade", pade_order="44"),
    dict(base_solution="direct_riccati"),
], ids=["root_pade", "pade22", "pade33", "pade44", "direct_riccati"])
@pytest.mark.parametrize("order", [0, 2])
def test_charfun_at_zero_and_minus_i(order, base_kw, set_name):
    """phi(0) = 1 and phi(-i) = e^{rT}, also through the H-expansion."""
    from roughprix.heston.rough_heston import RoughHeston

    rate, T = 0.03, 1.0
    m = RoughHeston(RoughHestonParam(**_C0_SETS[set_name], H=0.1), riskfree=rate, **base_kw)
    log_phi = m.log_charfun(np.array([0.0, -1j]), maturity=T, order=order,
                            time_steps=50, H=0.05)
    np.testing.assert_array_equal(log_phi.ravel(), [0.0, rate * T])


@pytest.mark.parametrize("set_name", sorted(_C0_SETS))
def test_digital_calls_on_contour_through_minus_i(set_name):
    """With R = 1 the call contour passes through a = -i at u = 0.

    u = 0 is a Chebyshev node, where h_N is evaluated when expansion_order >= 1.
    Digital calls are valid on any R > 0, so R = 1 must agree with R = 1.2.
    """
    from roughprix.heston.rough_heston import RoughHeston

    m = RoughHeston(RoughHestonParam(**_C0_SETS[set_name], H=0.1), riskfree=0.03)
    kw = dict(K=np.array([0.8, 0.9, 1.0, 1.1, 1.2]), maturity=1.0, digital=True,
              payoff_selection="as_requested", expansion_order=1, H=0.15, time_steps=100)
    got = np.asarray(m.eur_call_put(R=1.0, **kw))
    np.testing.assert_allclose(got, np.asarray(m.eur_call_put(R=1.2, **kw)), atol=1e-8, rtol=0)


# ===========================================================================
# Section M: root-Padé tail matching is on by default
# ===========================================================================
# The root variable E decays like C / x for large x = nu*aa*t^alpha, with
# C = r_+ / (r_+ - r_-) / Gamma(1 - alpha).  The tail constraint replaces the
# highest short-time equation of the [m/(m+1)] approximant by p_m = C q_(m+1),
# so the approximant has the exact leading tail.  The reference is implicit-pi
# on a graded grid, priced with the same Fourier settings, so the difference is
# the root-Padé error.

_TAIL_CASES = {
    "A_H-0.3_T1": (dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=-0.3), 1.0),
    "A_H-0.2_T1w": (dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=-0.2), 1 / 52),
    "B_H0.3_T2": (dict(lam=1.0, theta=0.025, nu=0.4, rho=-0.65, V0=0.025, H=0.3), 2.0),
    "C_H0.45_T1": (dict(lam=0.0, theta=0.0, nu=0.5, rho=-0.9, V0=0.04, H=0.45), 1.0),
}


@pytest.mark.parametrize("case", list(_TAIL_CASES))
def test_default_root_pade_smile(case):
    from roughprix.heston.rough_heston import RoughHeston

    p, T = _TAIL_CASES[case]
    param = RoughHestonParam(**p)
    K = np.exp(np.array([-3, -2, -1, 0, 1, 2, 3]) * np.sqrt(p["V0"] * T))
    iv = RoughHeston(param, 0.0, base_solution="root_pade").eur_call_put(
        K=K, maturity=T, implied_vol=True)
    ref = RoughHeston(param, 0.0, base_solution="direct_riccati").eur_call_put(
        K=K, maturity=T, implied_vol=True, time_grid="graded", time_steps=400)
    assert np.max(np.abs(iv - ref)) * 1e4 < 3.0


def test_default_root_pade_degree_is_9():
    """The default degree is 9 (Hager & Kreher 2026 use 7)."""
    from roughprix.heston.rough_heston import RoughHeston

    p = RoughHestonParam(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02, H=0.1)
    a = np.array([0.5 - 0.5j, 5.0 - 0.5j, 50.0 - 0.5j])
    t = np.linspace(0.0, 1.0, 11)
    np.testing.assert_array_equal(rough_heston_root_pade_f0(a, t, p, H=p.H),
                                  rough_heston_root_pade_f0(a, t, p, degree=9, H=p.H))
    np.testing.assert_array_equal(
        RoughHeston(p, 0.0, base_solution="root_pade").log_charfun(a),
        RoughHeston(p, 0.0, base_solution="root_pade", root_pade_degree=9).log_charfun(a))


# ===========================================================================
# Section N: root-Padé with the anchor at H = 1/2
# ===========================================================================
# At alpha = 1 the exact E is exp(-x), which has no 1/x tail: there the tail
# constraint, with C = 0 (1/Gamma(0) = 0), sets the 1/x coefficient to 0.  A free
# 1/x tail in E would make the log-CF integral grow like log u.  Selecting
# root-Padé at H = 1/2 warns that exact Heston is exact there.

_HALF_SETS = {
    "A": dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02),
    # Positive correlation.
    "D": dict(lam=0.1, theta=0.004, nu=0.5, rho=0.5, V0=0.03),
}


@pytest.mark.parametrize("set_name, T", [("D", 0.5), ("D", 1.0), ("A", 2.0)])
def test_root_pade_at_half_matches_exact_heston(set_name, T):
    from roughprix.heston.rough_heston import RoughHeston

    p = _HALF_SETS[set_name]
    param = RoughHestonParam(**p, H=0.5)
    K = np.exp(np.array([-3, -2, -1, 0, 1, 2, 3]) * np.sqrt(p["V0"] * T))
    with pytest.warns(UserWarning, match="exact_heston"):
        iv = RoughHeston(param, 0.0, base_solution="root_pade").eur_call_put(
            K=K, maturity=T, implied_vol=True)
    exact = RoughHeston(param, 0.0).eur_call_put(K=K, maturity=T, implied_vol=True)
    np.testing.assert_allclose(iv, exact, rtol=0, atol=1e-4)  # 1 bp


def test_no_root_pade_warning_from_auto_or_below_half():
    from roughprix.heston.rough_heston import RoughHeston

    K = np.array([0.9, 1.0, 1.1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RoughHeston(RoughHestonParam(**_HALF_SETS["A"], H=0.5), 0.0).eur_call_put(K=K, maturity=1.0)
        RoughHeston(RoughHestonParam(**_HALF_SETS["A"], H=0.49), 0.0,
                    base_solution="root_pade").eur_call_put(K=K, maturity=1.0)
    assert not [w for w in caught if "root_pade" in str(w.message)]


# ===========================================================================
# Section O: pade44 is Gatheral's [4/4]
# ===========================================================================
# The [4/4] approximant in y = tau^alpha matches four short-time coefficients
# b1..b4 and the four long-time coefficients of h ~ g0 + g1/y + g2/y^2 + g3/y^3,
# where g2 and g3 carry the nonlinear terms of Gatheral's formula.  g3 carries
# Gamma(1 - 2 alpha), which has a pole at H = 0; the approximant has a finite
# limit there.

_PADE44_SETS = {
    "A": dict(lam=0.3, theta=0.006, nu=0.3, rho=-0.7, V0=0.02),
    "B": dict(lam=1.0, theta=0.025, nu=0.4, rho=-0.65, V0=0.025),
    "C": dict(lam=0.0, theta=0.0, nu=0.5, rho=-0.9, V0=0.04),
}


def _pade44_kw(set_name):
    p = _PADE44_SETS[set_name]
    return dict(rho=p["rho"], nu=p["nu"], lam=p["lam"])


@pytest.mark.parametrize("H", [0.3, 0.1, -0.2, -0.4])
@pytest.mark.parametrize("set_name", ["A", "C"])
def test_pade44_long_time_coefficients(set_name, H):
    """h ~ g0 + g1/y + g2/y^2 + g3/y^3 for large y = tau^alpha, with the nonlinear g2 and g3."""
    from scipy.special import gamma

    kw = _pade44_kw(set_name)
    rho, nu, lam = kw["rho"], kw["nu"], kw["lam"]
    al = H + 0.5
    a = np.array([0.5 - 0.5j, 2.0 - 0.5j, 10.0 - 0.5j, -3.0 - 1.2j])
    D = np.sqrt((lam - 1j * rho * nu * a) ** 2 + nu ** 2 * a * (a + 1j))
    g0 = (lam - 1j * rho * nu * a - D) / nu ** 2  # stable root
    g1 = -g0 / (D * gamma(1 - al))
    g2 = -(gamma(1 - al) / gamma(1 - 2 * al) * g1 - 0.5 * nu ** 2 * g1 ** 2) / D
    g3 = -(gamma(1 - 2 * al) / gamma(1 - 3 * al) * g2 - nu ** 2 * g1 * g2) / D

    def h_at(y):
        return h_pade44(a, np.array([y ** (1 / al)]), H=H, **kw)[:, 0]

    # The next term of the expansion contributes O(1/y) to each extracted coefficient.
    y = 1e4
    np.testing.assert_allclose((h_at(y) - g0 - g1 / y) * y ** 2, g2, rtol=2e-3)
    y = 1e3
    np.testing.assert_allclose((h_at(y) - g0 - g1 / y - g2 / y ** 2) * y ** 3, g3, rtol=0.1)


@pytest.mark.parametrize("set_name", ["A", "C"])
def test_pade44_at_h_zero_is_the_limit(set_name):
    kw = _pade44_kw(set_name)
    a = np.array([0.5 - 0.5j, 2.0 - 0.5j, 10.0 - 0.5j, 200.0 - 0.5j, -3.0 - 1.2j])
    tau = np.geomspace(1e-4, 5.0, 30)
    h0 = h_pade44(a, tau, H=0.0, **kw)
    assert np.all(np.isfinite(h0))
    for H in (1e-8, -1e-8):
        np.testing.assert_allclose(h_pade44(a, tau, H=H, **kw), h0, rtol=1e-6)


_PADE44_SMILES = {
    "A_H-0.4_T3m": ("A", -0.4, 0.25),
    "B_H-0.2_T1w": ("B", -0.2, 1 / 52),
    "B_H-0.2_T1": ("B", -0.2, 1.0),
}


@pytest.mark.parametrize("case", list(_PADE44_SMILES))
def test_pade44_smile(case):
    from roughprix.heston.rough_heston import RoughHeston

    set_name, H, T = _PADE44_SMILES[case]
    p = _PADE44_SETS[set_name]
    param = RoughHestonParam(**p, H=H)
    K = np.exp(np.array([-2, -1, 0, 1, 2]) * np.sqrt(p["V0"] * T))
    iv = RoughHeston(param, 0.0, base_solution="pade", pade_order="44").eur_call_put(
        K=K, maturity=T, implied_vol=True, accuracy_check=False)
    ref = RoughHeston(param, 0.0, base_solution="direct_riccati").eur_call_put(
        K=K, maturity=T, implied_vol=True, time_grid="graded", time_steps=400)
    assert np.max(np.abs(iv - ref)) * 1e4 < 4.0
