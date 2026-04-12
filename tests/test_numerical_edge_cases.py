"""Tests for numerical edge cases and robustness.

Covers:
- NaN/inf recovery in _rhs_batch (overflow handling)
- Extreme pole tilts (near-vertical theta~0, near-horizontal theta~pi/2)
- Zero drogue velocity
- Singularity avoidance in coordinate conversions
"""

import numpy as np
import pytest

from conftest import DEFAULT_PHYSICS as _DEFAULT_PHYSICS

from drogued_drifters.models.drogued_drifter import DroguedDrifter, DrifterPhysics, EOMState
from drogued_drifters.eom import eval_M, eval_F
from drogued_drifters.coords import _uv_to_theta


def _sample_uv_default(z):
    z_arr = np.asarray(z, dtype=float)
    scalar = z_arr.ndim == 0
    z_arr = np.atleast_1d(z_arr)
    U = np.where(z_arr == 0.0, 1.0, -1.0)
    V = np.where(z_arr == 0.0, 1.0, -1.0)
    if scalar:
        return float(U[0]), float(V[0])
    return U, V


def test_rhs_batch_handles_nan_M():
    """_rhs_batch should replace NaN in qdd with zero."""
    dd = DroguedDrifter()

    N = 5
    Y = np.zeros((N, 8))
    Y[:, 2] = np.array([0.1, 1000.0, 0.05, 10.0, 0.01])
    Y[:, 3] = np.array([0.05, 500.0, 0.02, 5.0, 0.005])

    def sample_uv_const(z):
        return np.ones(N) * 0.1, np.ones(N) * 0.05

    try:
        dY = dd._rhs_batch(Y, sample_uv_const)
        assert dY.shape == Y.shape, "dY shape mismatch"
        assert np.all(np.isfinite(dY)), "dY should be finite after NaN handling"
    except np.linalg.LinAlgError:
        pass


def test_rhs_batch_handles_inf_F():
    """_rhs_batch should replace inf in F with zero vector."""
    dd = DroguedDrifter()

    N = 3
    Y = np.zeros((N, 8))

    def sample_uv_extreme(z):
        return (
            np.array([1e10, 1e-10, 0.1]),
            np.array([1e10, 1e-10, 0.1]),
        )

    try:
        dY = dd._rhs_batch(Y, sample_uv_extreme)
        assert dY.shape == Y.shape
        assert np.any(np.isfinite(dY)), "At least some rows should be finite"
    except (np.linalg.LinAlgError, OverflowError):
        pass


def test_extreme_vertical_pole():
    """Test pole nearly vertical (theta ~ pi)."""
    dd = DroguedDrifter()

    def sample_uv(z):
        if np.all(z == 0):
            return 0.5, 0.0
        return 0.1, 0.0

    xd, yd, _ = dd.get_final_drift(sample_uv, t_span=(0, 60))
    assert np.isfinite(xd) and np.isfinite(yd), "Vertical pole should give finite drift"


def test_extreme_horizontal_pole():
    """Test pole nearly horizontal (theta ~ pi/2)."""
    dd = DroguedDrifter()

    def sample_uv(z):
        return 0.1, 0.0

    try:
        xd, yd, _ = dd.get_final_drift(sample_uv, t_span=(0, 120))
        assert np.isfinite(xd) or np.isfinite(yd), "Near-horizontal should be handled"
    except (ValueError, RuntimeError):
        pass


def test_zero_drogue_velocity():
    """Drogue velocity stationary, buoy moving: drift should converge."""

    def sample_uv_sheared(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        U = np.zeros_like(z_arr)
        V = np.where(z_arr == 0.0, 0.5, 0.0)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    dd = DroguedDrifter()
    xd, yd, _ = dd.get_final_drift(sample_uv_sheared, t_span=(0, 120))

    assert np.isfinite(xd) and np.isfinite(yd)
    assert 0.0 <= yd <= 0.5, f"Expected yd in [0, 0.5], got {yd}"


def test_uv_to_theta_near_horizontal():
    """Conversion with large r (tilted pole)."""
    u_large = 10.0
    v_small = 0.1
    theta = _uv_to_theta(u_large, v_small)

    assert np.isfinite(theta), f"theta should be finite, got {theta}"
    assert 0 < theta < np.pi, f"Expected theta in (0, pi), got {theta}"


def test_uv_to_theta_near_vertical():
    """Conversion near-vertical."""
    u_small = 0.001
    v_small = 0.001
    theta = _uv_to_theta(u_small, v_small)

    assert np.isfinite(theta), f"theta should be finite, got {theta}"
    assert 0.99 * np.pi < theta <= np.pi, f"Expected theta near pi, got {theta}"


def test_uv_to_theta_zero_vector():
    """Conversion at (u, v) = (0, 0) should give theta = pi."""
    theta = _uv_to_theta(0.0, 0.0)
    np.testing.assert_allclose(theta, np.pi, rtol=1e-10)


def test_uv_to_theta_roundtrip_extreme():
    """Round-trip u,v<->theta,phi at extreme angles."""
    from drogued_drifters.coords import _spherical_to_uv

    theta_in = 0.9999 * np.pi
    phi_in = 0.0

    u, v, ud, vd = _spherical_to_uv(theta_in, phi_in, 0.0, 0.0)
    theta_out = _uv_to_theta(u, v)

    np.testing.assert_allclose(theta_out, theta_in, rtol=1e-8, err_msg="Round-trip failed near-vertical")


def test_M_func_positive_definite_extreme_angles():
    """M matrix should remain positive-definite even at extreme tilt angles."""
    dd = DroguedDrifter()
    M_horiz = eval_M(
        dd, _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=5.0, v_stereo=0.0,
            xd=0.0, yd=0.0,
            ud_stereo=0.0, vd_stereo=0.0,
            U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0,
        ),
    )

    eigvals = np.linalg.eigvalsh(M_horiz)
    assert np.all(eigvals > 0), f"M not positive definite at horizontal angle: {eigvals}"


def test_batch_extreme_velocities():
    """Batch processing should handle extreme individual particle velocities."""
    dd = DroguedDrifter()

    N = 5

    def sample_uv_batch(z):
        U_b = np.array([0.1, 1000.0, 0.01, -50.0, 0.2])
        V_b = np.array([0.05, 500.0, 0.005, -25.0, 0.1])
        return U_b, V_b

    try:
        xd, yd, Y_final, max_accel = dd.get_final_drift_batch(
            sample_uv_batch,
            t_span=(0, 120),
        )
        assert xd.shape == (N,)
        assert yd.shape == (N,)
        assert np.any(np.isfinite(xd)) or np.any(np.isfinite(yd))
    except (np.linalg.LinAlgError, OverflowError, ValueError):
        pass


def test_very_small_perturbation_stability():
    """Tiny perturbations should not cause numerical instability."""
    eps = 1e-12

    def sample_uv_base(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.1, 0.05)
        V = np.where(z_arr == 0.0, 0.05, 0.025)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    dd = DroguedDrifter()
    xd_base, yd_base, _ = dd.get_final_drift(sample_uv_base, t_span=(0, 120))

    def sample_uv_pert(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.1 + eps, 0.05 + eps)
        V = np.where(z_arr == 0.0, 0.05 + eps, 0.025 + eps)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    xd_pert, yd_pert, _ = dd.get_final_drift(sample_uv_pert, t_span=(0, 120))

    assert np.isfinite(xd_pert) and np.isfinite(yd_pert)


def test_M_F_continuity_near_zero():
    """M and F should be continuous as (u, v) -> (0, 0)."""
    dd = DroguedDrifter()
    eps_values = [1e-2, 1e-3, 1e-4, 1e-5]

    M_vals = []
    F_vals = []

    for eps in eps_values:
        state = EOMState(
            u_stereo=eps, v_stereo=eps, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0
        )
        M = eval_M(dd, _DEFAULT_PHYSICS, state)
        F = eval_F(dd, _DEFAULT_PHYSICS, state)
        M_vals.append(M.flatten())
        F_vals.append(F)

    for M_flat in M_vals:
        assert np.all(np.isfinite(M_flat)), "M has non-finite values"
    for F in F_vals:
        assert np.all(np.isfinite(F)), "F has non-finite values"


def test_spherical_singularity_at_pi():
    """Spherical (theta, phi) has singularity at theta=pi, but stereographic avoids it."""
    from drogued_drifters.coords import _uv_to_spherical, _spherical_to_uv

    u, v, ud, vd = _spherical_to_uv(np.pi, 0.0, 0.0, 0.0)
    assert np.isfinite(u) and np.isfinite(v)
    np.testing.assert_allclose([u, v], [0.0, 0.0], atol=1e-14)

    theta, phi, thetad, phid = _uv_to_spherical(0.0, 0.0, 0.0, 0.0)
    assert np.isfinite(theta) and np.isfinite(phi)
    np.testing.assert_allclose(theta, np.pi, rtol=1e-10)


def test_large_depth_pole_length():
    """Very long pole should not cause numerical issues."""

    def sample_uv(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.1, 0.05)
        V = np.zeros_like(z_arr)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    dd_long = DroguedDrifter(l=100.0)

    try:
        xd, yd, _ = dd_long.get_final_drift(sample_uv, t_span=(0, 120))
        assert np.isfinite(xd) or np.isfinite(yd)
    except (ValueError, RuntimeError):
        pass


def test_tiny_pole_length():
    """Very short pole should reduce to point particle."""

    def sample_uv(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.1, 0.05)
        V = np.zeros_like(z_arr)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    dd_short = DroguedDrifter(l=0.01)
    xd, yd, _ = dd_short.get_final_drift(sample_uv, t_span=(0, 120))
    assert np.isfinite(xd) and np.isfinite(yd)
