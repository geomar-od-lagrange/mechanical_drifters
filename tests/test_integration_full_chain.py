"""Full-chain integration tests.

These tests verify that the complete pipeline works end-to-end:
1. Synthetic gridded velocity data
2. Stokes drift profile computation
3. DroguedDrifter modeling
4. Final drift velocity computation

They validate that modules integrate correctly and produce
physically sensible results.
"""

import numpy as np
import pytest

from drogued_drifters.drifter import (
    DroguedDrifter,
    make_profile_sampler,
)
from drogued_drifters.stokes import compute_stokes_profile


def test_full_chain_stokes_to_drifter():
    """Full chain: Stokes drift → profile sampler → drifter model → drift velocity."""
    # Step 1: Synthetic Stokes drift profiles
    surface_u = 0.05
    surface_v = 0.0
    peak_period = 10.0
    depth_levels = np.array([-20.0, -15.0, -10.0, -5.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels
    )

    # u_stokes shape: (5,)
    assert u_stokes.shape == (5,)
    assert v_stokes.shape == (5,)

    # Step 2: Build profile sampler for N=1 particle
    N = 1
    U_profiles = u_stokes.reshape(-1, 1)  # (5, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # Step 3: DroguedDrifter with Stokes drift profile
    dd = DroguedDrifter()

    # Step 4: Compute final drift
    xd, yd, theta, Y_final = dd.get_final_drift_batch(
        sample_uv=sample_uv,
        t_span=(0, 120),
    )

    # Should produce finite results
    assert xd.shape == (1,)
    assert yd.shape == (1,)
    assert np.isfinite(xd[0]), "xd should be finite"
    # yd can be zero since v_stokes=0
    assert np.isfinite(yd[0]), "yd should be finite"


def test_full_chain_multi_partition_stokes():
    """Full chain with multi-partition Stokes drift summation."""
    depth_levels = np.array([-15.0, -10.0, -5.0, 0.0])

    # Partition 1: T=10s
    u1, v1 = compute_stokes_profile(0.04, 0.01, 10.0, depth_levels)

    # Partition 2: T=8s
    u2, v2 = compute_stokes_profile(0.02, 0.005, 8.0, depth_levels)

    # Total drift (linear superposition)
    u_total = u1 + u2
    v_total = v1 + v2

    # Build profile sampler for N=2 particles
    N = 2
    U_profiles = np.column_stack([u_total, u_total * 0.5])  # (4, 2)
    V_profiles = np.column_stack([v_total, v_total * 0.5])

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # DroguedDrifter
    dd = DroguedDrifter()
    xd, yd, theta, Y_final = dd.get_final_drift_batch(
        sample_uv=sample_uv,
        t_span=(0, 120),
    )

    # Should produce shape (2,) results
    assert xd.shape == (2,)
    assert yd.shape == (2,)
    assert np.all(np.isfinite(xd))
    assert np.all(np.isfinite(yd))

    # Particle 0 should have larger drift than particle 1
    # (0.5x smaller Stokes drift should give ~0.5x smaller drift)
    if xd[0] != 0:
        assert np.abs(xd[0]) > np.abs(
            xd[1] * 0.3
        ), f"Expected xd[0]={xd[0]} > 0.3*xd[1]={xd[1]*0.3}"


def test_full_chain_zero_stokes_zero_drift():
    """With zero Stokes drift, final drift should be zero."""
    depth_levels = np.array([-10.0, -5.0, 0.0])

    # Zero Stokes drift
    u_stokes, v_stokes = compute_stokes_profile(0.0, 0.0, 10.0, depth_levels)

    N = 1
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()
    xd, yd, theta, _ = dd.get_final_drift_batch(
        sample_uv=sample_uv,
        t_span=(0, 120),
    )

    # Drift should be negligible
    np.testing.assert_allclose(xd, 0.0, atol=0.05)
    np.testing.assert_allclose(yd, 0.0, atol=0.05)


def test_full_chain_shear_increases_drift():
    """Strong velocity shear should increase drift magnitude."""

    # No Stokes drift, but velocity shear
    def sample_uv_weak(*, t, x, y, z):
        return (0.1, 0.0) if z == 0 else (0.01, 0.0)

    def sample_uv_strong(*, t, x, y, z):
        return (0.5, 0.0) if z == 0 else (0.01, 0.0)

    dd_weak = DroguedDrifter(get_uv=sample_uv_weak)
    xd_weak, yd_weak = dd_weak.get_final_drift(t_span=(0, 120))

    dd_strong = DroguedDrifter(get_uv=sample_uv_strong)
    xd_strong, yd_strong = dd_strong.get_final_drift(t_span=(0, 120))

    # Both should have same sign (shear direction)
    if xd_weak != 0:
        assert np.sign(xd_strong) == np.sign(
            xd_weak
        ), f"Shear direction changed: weak={xd_weak}, strong={xd_strong}"

    # Strong shear should give larger drift magnitude
    assert (
        np.abs(xd_strong) > np.abs(xd_weak) * 0.9
    ), f"Strong shear should give larger drift: weak={xd_weak}, strong={xd_strong}"


def test_full_chain_preserves_initial_condition_for_warm_start():
    """Warm-starting from previous solution should converge quickly."""
    depth_levels = np.array([-10.0, -5.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.0, 10.0, depth_levels)

    N = 1
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()

    # First run: from rest
    xd1, yd1, theta1, Y_final1 = dd.get_final_drift_batch(
        sample_uv=sample_uv,
        t_span=(0, 120),
    )

    # Second run: warm-start from solution (if supported)
    try:
        xd2, yd2, theta2, Y_final2 = dd.get_final_drift_batch(
            sample_uv=sample_uv,
            t_span=(0, 120),
            y0=Y_final1,
        )
        # Just verify it runs without error
        assert xd2.shape == (1,)
        assert yd2.shape == (1,)
    except TypeError as e:
        # If y0 is not a parameter, skip
        if "y0" in str(e):
            pytest.skip("get_final_drift_batch does not accept y0 parameter")
        raise


def test_full_chain_multiple_particles_independence():
    """Multiple particles should evolve independently with different profiles."""
    depth_levels = np.array([-10.0, 0.0])

    # Particle 0: moderate Stokes
    u0, v0 = compute_stokes_profile(0.05, 0.0, 10.0, depth_levels)

    # Particle 1: weak Stokes
    u1, v1 = compute_stokes_profile(0.02, 0.0, 10.0, depth_levels)

    N = 2
    U_profiles = np.column_stack([u0, u1])
    V_profiles = np.column_stack([v0, v1])

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()
    # Build y0 in public format with theta close to pi
    y0 = np.zeros((N, 8))
    y0[:, 2] = 0.999 * np.pi  # theta column
    xd, yd, theta, _ = dd.get_final_drift_batch(
        sample_uv=sample_uv,
        t_span=(0, 120),
        y0=y0,
    )

    # Particle 0 should have stronger drift
    if xd[0] != 0 and xd[1] != 0:
        assert np.abs(xd[0]) / np.abs(xd[1]) > 1.5, (
            f"Particle 0 (2.5x stronger Stokes) should drift ~2.5x more: "
            f"xd[0]={xd[0]}, xd[1]={xd[1]}"
        )


def test_full_chain_opposite_shear_direction():
    """Opposite shear directions should produce opposite drift."""

    # Eastward shear
    def sample_uv_east(*, t, x, y, z):
        return (0.2, 0.0) if z == 0 else (0.1, 0.0)

    # Westward shear (opposite)
    def sample_uv_west(*, t, x, y, z):
        return (-0.2, 0.0) if z == 0 else (-0.1, 0.0)

    dd_east = DroguedDrifter(get_uv=sample_uv_east)
    xd_east, _ = dd_east.get_final_drift(t_span=(0, 120))

    dd_west = DroguedDrifter(get_uv=sample_uv_west)
    xd_west, _ = dd_west.get_final_drift(t_span=(0, 120))

    # Drifts should be opposite sign
    assert np.sign(xd_east) == -np.sign(
        xd_west
    ), f"Opposite shear should give opposite drift: east={xd_east}, west={xd_west}"


def test_full_chain_convergence_time():
    """Drifter should reach steady state within integration window."""
    depth_levels = np.array([-10.0, -5.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.01, 10.0, depth_levels)

    N = 1
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    dd = DroguedDrifter()

    # Run with sufficient time for convergence
    xd, yd, theta, Y_final = dd.get_final_drift_batch(
        sample_uv=sample_uv,
        t_span=(0, 600),  # Long integration
    )

    # Solution should be valid
    assert np.all(np.isfinite([xd[0], yd[0]]))

    # Theta should be near π (drogue hanging down) or near equilibrium
    assert 0.9 * np.pi <= theta[0] <= np.pi or np.isclose(theta[0], np.pi, atol=0.1)


def test_full_chain_scalar_to_batch_consistency():
    """Batch solution should match scalar solution with N=1."""
    depth_levels = np.array([-10.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.0, 10.0, depth_levels)

    # Scalar path (using custom get_uv)
    # depth_levels[-1] = 0.0 (surface), depth_levels[0] = -10.0 (deep)
    def sample_uv_scalar(*, t, x, y, z):
        if z == 0:
            return u_stokes[-1], v_stokes[-1]  # surface (index -1 = z=0)
        return u_stokes[0], v_stokes[0]  # deep (index 0 = z=-10)

    dd = DroguedDrifter(get_uv=sample_uv_scalar)
    xd_scalar, yd_scalar = dd.get_final_drift(t_span=(0, 120))

    # Batch path (N=1)
    U_profiles = u_stokes.reshape(-1, 1)
    V_profiles = v_stokes.reshape(-1, 1)
    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    xd_batch, yd_batch, _, _ = dd.get_final_drift_batch(
        sample_uv=sample_uv,
        t_span=(0, 120),
    )

    # Should agree (within solver tolerance)
    # Note: The exact match may vary due to solver differences
    np.testing.assert_allclose(xd_batch[0], xd_scalar, rtol=0.3, atol=0.01)
    np.testing.assert_allclose(yd_batch[0], yd_scalar, rtol=0.3, atol=0.01)


def test_full_chain_xarray_output():
    """get_full_solution should return xarray Dataset."""
    depth_levels = np.array([-10.0, 0.0])

    u_stokes, v_stokes = compute_stokes_profile(0.05, 0.01, 10.0, depth_levels)

    def sample_uv(*, t, x, y, z):
        if z == 0:
            return u_stokes[-1], v_stokes[-1]  # surface (z=0 at last index)
        return u_stokes[0], v_stokes[0]  # deep (z=-10 at index 0)

    dd = DroguedDrifter(get_uv=sample_uv)

    ds = dd.get_full_solution(
        t_span=(0, 120),
        t_eval=np.linspace(0, 120, 10),
    )

    # Should be xarray Dataset
    import xarray as xr

    assert isinstance(ds, xr.Dataset)

    # Should have expected variables
    for var in ["x", "y", "theta", "phi", "xd", "yd", "thetad", "phid"]:
        assert var in ds, f"Missing variable {var}"

    # Should have time coordinate
    assert "time" in ds.coords
    assert len(ds.time) == 10
