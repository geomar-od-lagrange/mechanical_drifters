from drogued_drifters.drifter import (
    DroguedDrifter,
    drogue_added_mass,
    buoy_added_mass,
    drogue_drag_coeff,
    buoy_drag_coeff,
)
from drogued_drifters.lagrange_model import M_func, F_func

import numpy as np


def test_drogued_drifter_instantiation():
    dd = DroguedDrifter()


def test_MF_callable():
    assert callable(M_func)
    assert callable(F_func)


def test_MF_evaluates():
    dd = DroguedDrifter()

    t = 0.0
    currents = dd.get_uv(t=t, z_d=0.0, y_b=0.0, x_b=0.0)

    M, F = dd._eval_M_F(
        t,
        x=0.0,
        y=0.0,
        theta=0.3,
        phi=0.1,
        xd=0.0,
        yd=0.0,
        thetad=0.0,
        phid=0.0,
        currents=currents,
    )

    assert len(M.squeeze().shape) == 2, "M not 2dim"
    assert len(F.squeeze().shape) == 1, "F not 1dim"


def test_no_drift_for_zero_currents():
    def _getuv_zero(*, t, z_d, y_b, x_b):
        return 0.0, 0.0, 0.0, 0.0

    dd = DroguedDrifter(get_uv=_getuv_zero)

    ds = dd.get_final_drift(t_span=(0.0, 30.0), t_eval=(0, 30.0))

    np.testing.assert_almost_equal(float(ds.xd.isel(time=-1)), 0.0, decimal=1)
    np.testing.assert_almost_equal(float(ds.yd.isel(time=-1)), 0.0, decimal=1)


def test_no_drift_for_theta_pi_zero_currents():
    """Drogue hangs straight down (theta=pi), no currents: should stay at rest."""

    def _getuv_zero(*, t, z_d, y_b, x_b):
        return 0.0, 0.0, 0.0, 0.0

    dd = DroguedDrifter(get_uv=_getuv_zero)

    ds = dd.get_final_drift(t_span=(0.0, 30.0), theta=np.pi, t_eval=(0, 30.0))

    np.testing.assert_almost_equal(float(ds.xd.isel(time=-1)), 0.0, decimal=1)
    np.testing.assert_almost_equal(float(ds.yd.isel(time=-1)), 0.0, decimal=1)


def test_parameterization_matches_table1():
    """Check that parameterization functions reproduce Callies et al. values."""
    rho = 1025.0
    # Drogue: cross of two plates, w_d=0.5m, h_d=0.5m
    m_tilde_d = drogue_added_mass(rho=rho, w_d=0.5, h_d=0.5)
    np.testing.assert_almost_equal(m_tilde_d, 101.0, decimal=0)

    k_d = drogue_drag_coeff(rho=rho, w_d=0.5, h_d=0.5)
    np.testing.assert_almost_equal(k_d, 154.0, decimal=-1)

    # Buoy: cylinder, d_b=0.1m, h_b=0.24m
    m_tilde_b = buoy_added_mass(rho=rho, d_b=0.1, h_b=0.24)
    np.testing.assert_almost_equal(m_tilde_b, 1.9, decimal=1)

    k_b = buoy_drag_coeff(rho=rho, d_b=0.1, h_b=0.24)
    np.testing.assert_almost_equal(k_b, 12.0, decimal=0)


def test_steady_state_independent_of_added_mass():
    """Added mass only affects acceleration, not steady-state drift."""

    def _getuv_sheared(*, t, z_d, y_b, x_b):
        factor = np.exp(-abs(z_d) / 2.0)
        return 1.0, 0.0, factor, 0.0

    dd_with = DroguedDrifter(
        m_tilde_d=101.0,
        m_tilde_b=1.9,
        get_uv=_getuv_sheared,
    )
    dd_without = DroguedDrifter(
        m_tilde_d=0.0,
        m_tilde_b=0.0,
        get_uv=_getuv_sheared,
    )

    ds_with = dd_with.get_final_drift(t_span=(0.0, 600.0))
    ds_without = dd_without.get_final_drift(t_span=(0.0, 600.0))

    np.testing.assert_almost_equal(
        float(ds_with.xd.isel(time=-1)), float(ds_without.xd.isel(time=-1)), decimal=1
    )
    np.testing.assert_almost_equal(
        float(ds_with.yd.isel(time=-1)), float(ds_without.yd.isel(time=-1)), decimal=1
    )


def test_get_full_solution_returns_xarray():
    """get_full_solution returns an xarray Dataset with named variables."""
    dd = DroguedDrifter()
    ds = dd.get_full_solution(t_span=(0, 10), t_eval=[0, 5, 10])

    assert "time" in ds.coords
    for var in ["x", "y", "theta", "phi", "xd", "yd", "thetad", "phid"]:
        assert var in ds, f"missing variable {var}"
    assert len(ds.time) == 3

    # arithmetic preserves xarray type (needed for .plot())
    import xarray as xr

    theta_deg = ds.theta * 180 / np.pi
    assert isinstance(theta_deg, xr.DataArray)
    assert "time" in theta_deg.coords
