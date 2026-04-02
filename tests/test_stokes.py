"""Tests for stokes.py: Stokes drift profile computation.

Tests cover:
- Analytical validation against exponential decay law
- Multi-partition summation
- Edge cases (zero period, long periods, large depths)
- Vectorization over spatial dimensions
- Surface boundary condition (z=0)
"""
import numpy as np
import pytest

from drogued_drifters.stokes import compute_stokes_profile


def test_stokes_single_partition_analytical():
    """Verify against analytical exponential decay for a single wave partition.

    For a single monochromatic wave, the Stokes drift decays as:
        u(z) = u_surface * exp(2*k*z)   (z <= 0, z-up convention)
    where k = omega^2 / g with omega = 2*pi/T.

    Test a known case: surface_u=0.1 m/s, T=10 s,
    depth_levels=[-15, -10, -5, 0] m (ascending, deepest first).
    """
    surface_u = 0.1
    surface_v = 0.0
    peak_period = 10.0
    depth_levels = np.array([-15.0, -10.0, -5.0, 0.0])
    g = 9.81

    omega = 2 * np.pi / peak_period
    k_expected = omega**2 / g

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels, g=g
    )

    # u_profile shape should be (4,) with decay at each depth
    assert u_profile.shape == (4,), f"Expected shape (4,), got {u_profile.shape}"

    # At z=0 (index 3), should recover surface value
    np.testing.assert_allclose(u_profile[3], surface_u, rtol=1e-10)

    # At z=-5 (index 2), should decay by exp(2*k*(-5)) = exp(-2*k*5)
    expected_u5 = surface_u * np.exp(2 * k_expected * (-5.0))
    np.testing.assert_allclose(u_profile[2], expected_u5, rtol=1e-10)

    # At z=-10 (index 1), should decay by exp(2*k*(-10)) = exp(-2*k*10)
    expected_u10 = surface_u * np.exp(2 * k_expected * (-10.0))
    np.testing.assert_allclose(u_profile[1], expected_u10, rtol=1e-10)

    # v should remain zero (no northward surface drift)
    np.testing.assert_allclose(v_profile, 0.0, atol=1e-15)


def test_stokes_zero_surface_drift():
    """Zero surface drift should give zero drift at all depths."""
    surface_u = 0.0
    surface_v = 0.0
    peak_period = 8.0
    depth_levels = np.array([-20.0, -10.0, -5.0, -2.0, 0.0])

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels
    )

    np.testing.assert_allclose(u_profile, 0.0, atol=1e-15)
    np.testing.assert_allclose(v_profile, 0.0, atol=1e-15)


def test_stokes_both_components():
    """Test with nonzero u and v components."""
    surface_u = 0.05
    surface_v = 0.03
    peak_period = 12.0
    depth_levels = np.array([-6.0, -3.0, 0.0])
    g = 9.81

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels, g=g
    )

    # Both should have same exponential decay with same k
    omega = 2 * np.pi / peak_period
    k = omega**2 / g

    # At z=0 (index 2), recover surface
    np.testing.assert_allclose(u_profile[2], surface_u, rtol=1e-10)
    np.testing.assert_allclose(v_profile[2], surface_v, rtol=1e-10)

    # Ratio u/v should be preserved at all depths
    for i in range(len(depth_levels)):
        if surface_v != 0:
            ratio_expected = surface_u / surface_v
            ratio_actual = u_profile[i] / v_profile[i]
            np.testing.assert_allclose(ratio_actual, ratio_expected, rtol=1e-10)


def test_stokes_long_wavelength_shallow_decay():
    """Long period (small k) gives shallow decay with depth.

    Very long period (e.g., T=30s, k~0.002/m) should show minimal decay
    over typical depths.
    """
    surface_u = 0.1
    surface_v = 0.0
    peak_period = 30.0  # Long wave
    depth_levels = np.array([-10.0, -5.0, 0.0])
    g = 9.81

    omega = 2 * np.pi / peak_period
    k = omega**2 / g

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels, g=g
    )

    # Decay factor at z=-10: exp(2*k*(-10)) = exp(-2*k*10)
    decay_z10 = np.exp(2 * k * (-10.0))
    expected_u10 = surface_u * decay_z10

    np.testing.assert_allclose(u_profile[0], expected_u10, rtol=1e-10)

    # For very long waves, decay should be slow (decay_z10 not << 1)
    assert decay_z10 > 0.5, f"Expected slow decay for long wave, got {decay_z10}"


def test_stokes_short_wavelength_fast_decay():
    """Short period (large k) gives rapid decay with depth.

    Short period (e.g., T=5s, k~0.008/m) should show substantial decay
    over modest depths.
    """
    surface_u = 0.1
    surface_v = 0.0
    peak_period = 5.0  # Short wave
    depth_levels = np.array([-10.0, -5.0, 0.0])
    g = 9.81

    omega = 2 * np.pi / peak_period
    k = omega**2 / g

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels, g=g
    )

    # Decay factor at z=-10: exp(2*k*(-10)) = exp(-2*k*10)
    decay_z10 = np.exp(2 * k * (-10.0))
    expected_u10 = surface_u * decay_z10

    np.testing.assert_allclose(u_profile[0], expected_u10, rtol=1e-10)

    # For short waves, decay should be rapid (decay_z10 << 1)
    assert decay_z10 < 0.1, f"Expected fast decay for short wave, got {decay_z10}"


def test_stokes_vectorized_spatial():
    """Test vectorization over spatial dimensions (y, x)."""
    # Scalar period, vectorized surface drift
    surface_u = np.array([[0.1, 0.2], [0.3, 0.4]])  # (2, 2)
    surface_v = np.array([[0.05, 0.1], [0.15, 0.2]])  # (2, 2)
    peak_period = 10.0
    depth_levels = np.array([-10.0, -5.0, 0.0])

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels
    )

    # Output should be (D, 2, 2) = (3, 2, 2)
    assert u_profile.shape == (3, 2, 2), f"Expected (3, 2, 2), got {u_profile.shape}"
    assert v_profile.shape == (3, 2, 2), f"Expected (3, 2, 2), got {v_profile.shape}"

    # At z=0 (index 2), should recover surface
    np.testing.assert_allclose(u_profile[2], surface_u, rtol=1e-10)
    np.testing.assert_allclose(v_profile[2], surface_v, rtol=1e-10)


def test_stokes_vectorized_period_and_spatial():
    """Test vectorization over both period and spatial dimensions."""
    # Spatial variation (2, 2) with period variation (2, 2)
    surface_u = np.array([[0.1, 0.2], [0.3, 0.4]])
    surface_v = np.array([[0.0, 0.05], [0.1, 0.15]])
    peak_period = np.array([[10.0, 12.0], [8.0, 15.0]])
    depth_levels = np.array([-5.0, 0.0])

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels
    )

    # Output should be (2, 2, 2) = (D, y, x)
    assert u_profile.shape == (2, 2, 2), f"Expected (2, 2, 2), got {u_profile.shape}"

    # At z=0 (index 1), recover surface
    np.testing.assert_allclose(u_profile[1], surface_u, rtol=1e-10)


def test_stokes_multi_partition_summation():
    """Test summing multiple wave partitions manually.

    Should be able to call compute_stokes_profile multiple times and sum
    the results to get total drift from multiple partitions.
    """
    depth_levels = np.array([-10.0, -5.0, 0.0])

    # Partition 1: u_s=0.1, v_s=0.0, T=10s
    u1, v1 = compute_stokes_profile(0.1, 0.0, 10.0, depth_levels)

    # Partition 2: u_s=0.05, v_s=0.02, T=8s
    u2, v2 = compute_stokes_profile(0.05, 0.02, 8.0, depth_levels)

    # Total drift
    u_total = u1 + u2
    v_total = v1 + v2

    # Verify shapes
    assert u_total.shape == (3,)
    assert v_total.shape == (3,)

    # Verify surface (z=0, index 2) sums correctly
    np.testing.assert_allclose(u_total[2], 0.1 + 0.05, rtol=1e-10)
    np.testing.assert_allclose(v_total[2], 0.0 + 0.02, rtol=1e-10)


def test_stokes_array_input_types():
    """Test with various input array types (list, array, scalar)."""
    surface_u = 0.1
    surface_v = 0.0
    peak_period = 10.0
    depth_levels_list = [-10.0, -5.0, 0.0]

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels_list
    )

    assert u_profile.shape == (3,)
    assert v_profile.shape == (3,)
    # Surface is at index 2 (z=0)
    np.testing.assert_allclose(u_profile[2], 0.1, rtol=1e-10)


def test_stokes_positive_depth_invalid():
    """Depth levels should be non-positive in z-up convention.

    Positive depths are physically nonsensical (above the surface).
    The function should handle or reject them gracefully.
    """
    surface_u = 0.1
    surface_v = 0.0
    peak_period = 10.0
    depth_levels = np.array([-10.0, 5.0, 0.0])  # Positive 5.0 is invalid (above surface)

    # The function does not currently validate; test that shape is preserved.
    # With z-up, positive z means above the surface: exp(2*k*z) > 1 (amplification).
    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels
    )

    # Shape should still be (3,) regardless of validity of individual levels.
    assert u_profile.shape == (3,)


def test_stokes_very_large_depth():
    """Very large depths should see negligible Stokes drift.

    At |z| >> 1/(2k), the exponential decay should be extremely small.
    With z-up convention, depth_levels must be non-positive (deepest first).
    """
    surface_u = 0.1
    surface_v = 0.0
    peak_period = 10.0
    depth_levels = np.array([-500.0, -100.0, 0.0])  # Very deep (z-up: negative)
    g = 9.81

    omega = 2 * np.pi / peak_period
    k = omega**2 / g

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels, g=g
    )

    # At z=-500 (index 0), decay is exp(2*k*(-500)) = exp(-2*k*500) ~ 1e-30
    decay_large = np.exp(2 * k * (-500.0))
    expected_u_large = surface_u * decay_large

    np.testing.assert_allclose(u_profile[0], expected_u_large, rtol=1e-6)
    assert u_profile[0] < 1e-10, f"Drift at z=-500m should be negligible, got {u_profile[0]}"


def test_stokes_output_dtype_float():
    """Output arrays should be float dtype."""
    surface_u = 0.1
    surface_v = 0.0
    peak_period = 10.0
    depth_levels = np.array([-5.0, 0.0])

    u_profile, v_profile = compute_stokes_profile(
        surface_u, surface_v, peak_period, depth_levels
    )

    assert u_profile.dtype == np.float64 or u_profile.dtype == float
    assert v_profile.dtype == np.float64 or v_profile.dtype == float
