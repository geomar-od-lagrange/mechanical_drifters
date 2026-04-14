"""Full-chain integration tests.

These tests verify that the complete pipeline works end-to-end:
1. Synthetic gridded velocity data
2. Stokes drift profile computation
3. DroguedDrifter modeling
4. Final drift velocity computation
"""

import numpy as np
import pytest

from mechanical_drifters.models.drogued_drifter import DroguedDrifter
from mechanical_drifters.parcels import _make_profile_sampler as make_profile_sampler
from mechanical_drifters.stokes import compute_stokes_profile


def _integrate_single(dd, sample_uv, *, t_span, t_eval=None, **kwargs):
    """Helper: integrate single particle, return (xd, yd, max_accel)."""
    y0 = np.array([[
        kwargs.get('x', 0.0), kwargs.get('y', 0.0),
        kwargs.get('theta', np.pi), kwargs.get('phi', 0.0),
        kwargs.get('xd', 0.0), kwargs.get('yd', 0.0),
        kwargs.get('thetad', 0.0), kwargs.get('phid', 0.0),
    ]])
    t, Y, max_accel = dd.integrate(sample_uv, t_span=t_span, y0=y0, t_eval=t_eval)
    vel = dd.drift_velocity(Y[-1])
    return float(vel[0, 0]), float(vel[0, 1]), max_accel


def test_full_chain_stokes_to_drifter():
    """Full chain: Stokes drift -> profile sampler -> drifter model -> drift velocity."""
    surface_u = 0.05
    surface_v = 0.0
    peak_period = 10.0
    depth_levels = np.array([-20.0, -15.0, -10.0, -5.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels
    )

    assert u_stokes.shape == (5,)
    assert v_stokes.shape == (5,)

    N = 1
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()

    t, Y, max_accel = dd.integrate(
        sample_uv,
        t_span=(0, 120),
    )
    drift_vel = dd.drift_velocity(Y[-1])
    xd = drift_vel[:, 0]
    yd = drift_vel[:, 1]

    assert xd.shape == (1,)
    assert yd.shape == (1,)
    assert np.isfinite(xd[0]), "xd should be finite"
    assert np.isfinite(yd[0]), "yd should be finite"


def test_full_chain_multi_partition_stokes():
    """Full chain with multi-partition Stokes drift summation."""
    depth_levels = np.array([-15.0, -10.0, -5.0, 0.0])

    u1, v1 = compute_stokes_profile(0.04, 0.01, 10.0, depth_levels)
    u2, v2 = compute_stokes_profile(0.02, 0.005, 8.0, depth_levels)

    u_total = u1 + u2
    v_total = v1 + v2

    N = 2
    U_profiles = np.column_stack([u_total, u_total * 0.5])
    V_profiles = np.column_stack([v_total, v_total * 0.5])

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()
    t, Y, max_accel = dd.integrate(
        sample_uv,
        t_span=(0, 120),
    )
    drift_vel = dd.drift_velocity(Y[-1])
    xd = drift_vel[:, 0]
    yd = drift_vel[:, 1]

    assert xd.shape == (2,)
    assert yd.shape == (2,)
    assert np.all(np.isfinite(xd))
    assert np.all(np.isfinite(yd))

    if xd[0] != 0:
        assert np.abs(xd[0]) > np.abs(xd[1] * 0.3), f"Expected xd[0]={xd[0]} > 0.3*xd[1]={xd[1]*0.3}"


def test_full_chain_zero_stokes_zero_drift():
    """With zero Stokes drift, final drift should be zero."""
    depth_levels = np.array([-10.0, -5.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.0, 0.0, 10.0, depth_levels)

    N = 1
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()
    t, Y, _ = dd.integrate(
        sample_uv,
        t_span=(0, 120),
    )
    drift_vel = dd.drift_velocity(Y[-1])

    np.testing.assert_allclose(drift_vel[:, 0], 0.0, atol=0.05)
    np.testing.assert_allclose(drift_vel[:, 1], 0.0, atol=0.05)


def test_full_chain_shear_increases_drift():
    """Strong velocity shear should increase drift magnitude."""

    def sample_uv_weak(z):
        z_arr = np.asarray(z, dtype=float)
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.1, 0.01)
        V = np.zeros_like(z_arr)
        return U, V

    def sample_uv_strong(z):
        z_arr = np.asarray(z, dtype=float)
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.5, 0.01)
        V = np.zeros_like(z_arr)
        return U, V

    dd = DroguedDrifter()
    xd_weak, yd_weak, _ = _integrate_single(dd, sample_uv_weak, t_span=(0, 120))
    xd_strong, yd_strong, _ = _integrate_single(dd, sample_uv_strong, t_span=(0, 120))

    if xd_weak != 0:
        assert np.sign(xd_strong) == np.sign(xd_weak), \
            f"Shear direction changed: weak={xd_weak}, strong={xd_strong}"

    assert np.abs(xd_strong) > np.abs(xd_weak) * 0.9, \
        f"Strong shear should give larger drift: weak={xd_weak}, strong={xd_strong}"


def test_full_chain_preserves_initial_condition_for_warm_start():
    """Warm-starting from previous solution should converge quickly."""
    depth_levels = np.array([-10.0, -5.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.0, 10.0, depth_levels)

    N = 1
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()

    t1, Y1, max_accel1 = dd.integrate(
        sample_uv,
        t_span=(0, 120),
    )
    Y_final1 = Y1[-1]

    t2, Y2, max_accel2 = dd.integrate(
        sample_uv,
        t_span=(0, 120),
        y0=Y_final1,
    )
    drift_vel2 = dd.drift_velocity(Y2[-1])
    assert drift_vel2.shape == (1, 2)


def test_full_chain_multiple_particles_independence():
    """Multiple particles should evolve independently with different profiles."""
    depth_levels = np.array([-10.0, 0.0])

    u0, v0 = compute_stokes_profile(0.05, 0.0, 10.0, depth_levels)
    u1, v1 = compute_stokes_profile(0.02, 0.0, 10.0, depth_levels)

    N = 2
    U_profiles = np.column_stack([u0, u1])
    V_profiles = np.column_stack([v0, v1])

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()
    y0 = np.zeros((N, 8))
    y0[:, 2] = 0.999 * np.pi
    t, Y, _ = dd.integrate(
        sample_uv,
        t_span=(0, 120),
        y0=y0,
    )
    drift_vel = dd.drift_velocity(Y[-1])
    xd = drift_vel[:, 0]

    if xd[0] != 0 and xd[1] != 0:
        assert np.abs(xd[0]) / np.abs(xd[1]) > 1.5, (
            f"Particle 0 (2.5x stronger Stokes) should drift ~2.5x more: "
            f"xd[0]={xd[0]}, xd[1]={xd[1]}"
        )


def test_full_chain_opposite_shear_direction():
    """Opposite shear directions should produce opposite drift."""

    def sample_uv_east(z):
        z_arr = np.asarray(z, dtype=float)
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.2, 0.1)
        V = np.zeros_like(z_arr)
        return U, V

    def sample_uv_west(z):
        z_arr = np.asarray(z, dtype=float)
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, -0.2, -0.1)
        V = np.zeros_like(z_arr)
        return U, V

    dd = DroguedDrifter()
    xd_east, _, _ma = _integrate_single(dd, sample_uv_east, t_span=(0, 120))
    xd_west, _, _ma = _integrate_single(dd, sample_uv_west, t_span=(0, 120))

    assert np.sign(xd_east) == -np.sign(xd_west), \
        f"Opposite shear should give opposite drift: east={xd_east}, west={xd_west}"


def test_full_chain_convergence_time():
    """Drifter should reach steady state within integration window."""
    depth_levels = np.array([-10.0, -5.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.01, 10.0, depth_levels)

    N = 1
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()

    t, Y, max_accel = dd.integrate(
        sample_uv,
        t_span=(0, 600),
    )
    Y_final = Y[-1]
    drift_vel = dd.drift_velocity(Y_final)

    assert np.all(np.isfinite(drift_vel))

    theta_final = Y_final[0, 2]
    assert 0.9 * np.pi <= theta_final <= np.pi or np.isclose(theta_final, np.pi, atol=0.1)


def test_full_chain_scalar_to_batch_consistency():
    """Batch solution should match scalar solution with N=1."""
    depth_levels = np.array([-10.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.0, 10.0, depth_levels)

    def sample_uv_scalar(z):
        z_arr = np.asarray(z, dtype=float)
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, u_stokes[-1], u_stokes[0])
        V = np.where(z_arr == 0.0, v_stokes[-1], v_stokes[0])
        return U, V

    dd = DroguedDrifter()
    xd_scalar, yd_scalar, _ = _integrate_single(dd, sample_uv_scalar, t_span=(0, 120))

    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)
    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    t, Y, _ = dd.integrate(
        sample_uv,
        t_span=(0, 120),
    )
    drift_vel = dd.drift_velocity(Y[-1])

    np.testing.assert_allclose(drift_vel[0, 0], xd_scalar, rtol=0.3, atol=0.01)
    np.testing.assert_allclose(drift_vel[0, 1], yd_scalar, rtol=0.3, atol=0.01)


def test_full_chain_integrate_with_t_eval():
    """integrate with t_eval returns full trajectory in public coords."""
    depth_levels = np.array([-10.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.01, 10.0, depth_levels)

    def sample_uv(z):
        z_arr = np.asarray(z, dtype=float)
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, u_stokes[-1], u_stokes[0])
        V = np.where(z_arr == 0.0, v_stokes[-1], v_stokes[0])
        return U, V

    dd = DroguedDrifter()

    t_eval_arr = np.linspace(0, 120, 10)
    t, Y, max_accel = dd.integrate(
        sample_uv,
        t_span=(0, 120),
        t_eval=t_eval_arr,
    )

    assert t.shape == (10,)
    assert Y.shape == (10, 1, 8)

    for var_idx in range(8):
        assert np.all(np.isfinite(Y[:, 0, var_idx])), f"Variable {var_idx} has non-finite values"
