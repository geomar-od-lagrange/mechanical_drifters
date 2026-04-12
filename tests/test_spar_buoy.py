"""Tests for the SparBuoy model."""

import numpy as np
import pytest

from drogued_drifters.models.spar_buoy import SparBuoy, SparBuoyPhysics


def test_spar_buoy_instantiation():
    sb = SparBuoy()
    assert sb.physics.l == 15.0


def test_spar_buoy_custom_length():
    sb = SparBuoy(l=10.0)
    assert sb.physics.l == 10.0


def test_spar_buoy_physics_instance():
    physics = SparBuoyPhysics(l=20.0)
    sb = SparBuoy(physics)
    assert sb.physics.l == 20.0


def test_spar_buoy_uniform_flow():
    """Uniform flow: spar buoy should drift at the current speed."""
    sb = SparBuoy(l=10.0)

    U_const = 0.5

    def sample_uv(z):
        N = len(np.atleast_1d(z))
        return np.full(N, U_const), np.zeros(N)

    drift_vel, Y_final, max_accel = sb.steady_state_batch(sample_uv)

    np.testing.assert_allclose(drift_vel[0, 0], U_const, rtol=1e-10)
    np.testing.assert_allclose(drift_vel[0, 1], 0.0, atol=1e-14)
    assert max_accel == 0.0


def test_spar_buoy_sheared_flow():
    """Linearly sheared flow: drift = average of surface and bottom."""
    sb = SparBuoy(l=10.0)

    U_surface = 1.0
    U_bottom = 0.0

    def sample_uv(z):
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        # Linear profile: U=1 at z=0, U=0 at z=-10
        U = np.clip(1.0 + z_arr / 10.0, 0.0, 1.0)
        V = np.zeros_like(U)
        return U, V

    drift_vel, Y_final, max_accel = sb.steady_state_batch(sample_uv)

    # Average of surface (1.0) and bottom (0.0) = 0.5
    expected = 0.5 * (U_surface + U_bottom)
    np.testing.assert_allclose(drift_vel[0, 0], expected, rtol=1e-10)


def test_spar_buoy_multiple_particles():
    """Multiple particles with different velocity profiles."""
    sb = SparBuoy(l=5.0)
    N = 3

    def sample_uv(z):
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        # Different velocity for each particle
        U = np.array([0.1, 0.3, 0.5]) * np.ones_like(z_arr)
        V = np.array([0.05, 0.1, 0.0]) * np.ones_like(z_arr)
        return U, V

    drift_vel, Y_final, max_accel = sb.steady_state_batch(sample_uv)

    assert drift_vel.shape == (N, 2)
    np.testing.assert_allclose(drift_vel[:, 0], [0.1, 0.3, 0.5], rtol=1e-10)
    np.testing.assert_allclose(drift_vel[:, 1], [0.05, 0.1, 0.0], rtol=1e-10)


def test_spar_buoy_zero_flow():
    """Zero flow: zero drift."""
    sb = SparBuoy(l=10.0)

    def sample_uv(z):
        N = len(np.atleast_1d(z))
        return np.zeros(N), np.zeros(N)

    drift_vel, Y_final, max_accel = sb.steady_state_batch(sample_uv)

    np.testing.assert_allclose(drift_vel[0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(drift_vel[0, 1], 0.0, atol=1e-14)


def test_spar_buoy_max_depth():
    sb = SparBuoy(l=10.0)
    assert sb._max_depth(sb.physics) == 10.0


def test_spar_buoy_state_size():
    sb = SparBuoy()
    assert sb.state_size == 4


def test_spar_buoy_make_kernel_creates_callable():
    """make_kernel should return a callable."""
    sb = SparBuoy(l=5.0)
    kernel = sb.make_kernel()
    assert callable(kernel)


def test_spar_buoy_rhs_batch():
    """_rhs_batch should return depth-averaged velocities."""
    sb = SparBuoy(l=10.0)
    N = 2
    Y = np.zeros((N, 4))

    def sample_uv(z):
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        U = np.where(z_arr == 0, np.array([0.4, 0.6]), np.array([0.2, 0.3]))
        V = np.zeros_like(U)
        return U, V

    dY = sb._rhs_batch(Y, sample_uv)

    assert dY.shape == (N, 4)
    # dx/dt = average U
    np.testing.assert_allclose(dY[:, 0], [0.3, 0.45], rtol=1e-10)
    # dy/dt = 0
    np.testing.assert_allclose(dY[:, 1], 0.0, atol=1e-14)
    # accelerations = 0
    np.testing.assert_allclose(dY[:, 2], 0.0, atol=1e-14)
    np.testing.assert_allclose(dY[:, 3], 0.0, atol=1e-14)
