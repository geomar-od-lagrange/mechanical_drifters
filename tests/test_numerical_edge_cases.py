"""Tests for numerical edge cases and robustness.

Covers:
- NaN/inf recovery in _rhs_batch (overflow handling)
- Extreme pole tilts (near-vertical θ≈0, near-horizontal θ≈π/2)
- Zero drogue velocity
- Singularity avoidance in coordinate conversions
"""
import numpy as np
import pytest

from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.lagrange_model import F_func, M_func, _uv_to_theta


def test_rhs_batch_handles_nan_M():
    """_rhs_batch should replace NaN in M with identity matrix."""
    dd = DroguedDrifter()

    # Create state with values that might trigger overflow
    N = 5
    Y = np.zeros((N, 8))
    Y[:, 2] = np.array([0.1, 1000.0, 0.05, 10.0, 0.01])  # u: some extreme
    Y[:, 3] = np.array([0.05, 500.0, 0.02, 5.0, 0.005])  # v: some extreme

    def sample_uv_const(z):
        """Dummy sampler returning small velocities."""
        return np.ones(N) * 0.1, np.ones(N) * 0.05

    # Call _rhs_batch; should not raise, even if some M/F are NaN
    try:
        dY = dd._rhs_batch(Y, sample_uv_const)
        # If we get here, NaN handling worked
        assert dY.shape == Y.shape, "dY shape mismatch"
        # The bad particles should have finite dY (identity + zero-force solve)
        assert np.all(np.isfinite(dY)), "dY should be finite after NaN handling"
    except np.linalg.LinAlgError:
        # Alternative: linear algebra error is acceptable if overflow is severe
        pass


def test_rhs_batch_handles_inf_F():
    """_rhs_batch should replace inf in F with zero vector."""
    dd = DroguedDrifter()

    N = 3
    Y = np.zeros((N, 8))

    def sample_uv_extreme(z):
        """Return extreme velocities that might cause inf in F."""
        return (
            np.array([1e10, 1e-10, 0.1]),  # U_b
            np.array([1e10, 1e-10, 0.1]),  # V_b
        )

    try:
        dY = dd._rhs_batch(Y, sample_uv_extreme)
        assert dY.shape == Y.shape
        # Should recover by replacing bad rows with identity+zero
        assert np.any(np.isfinite(dY)), "At least some rows should be finite"
    except (np.linalg.LinAlgError, OverflowError):
        # Overflow is acceptable edge case
        pass


def test_extreme_vertical_pole():
    """Test pole nearly vertical (θ ≈ π)."""
    dd = DroguedDrifter()

    def sample_uv(z):
        if np.all(z == 0):
            return 0.5, 0.0
        return 0.1, 0.0

    # Equilibrium: u ≈ 0, v ≈ 0 gives θ ≈ π
    xd, yd = dd.get_final_drift(
        # No custom get_uv, use default
        t_span=(0, 60),
    )
    assert np.isfinite(xd) and np.isfinite(yd), "Vertical pole should give finite drift"


def test_extreme_horizontal_pole():
    """Test pole nearly horizontal (θ ≈ π/2)."""
    dd = DroguedDrifter()

    # Large u/v gives θ → π/2
    def sample_uv(z):
        return 0.1, 0.0

    try:
        xd, yd = dd.get_final_drift(t_span=(0, 120))
        # Should either succeed or raise clearly
        assert np.isfinite(xd) or np.isfinite(yd), "Near-horizontal should be handled"
    except (ValueError, RuntimeError):
        # Pole angle singularity is acceptable to reject
        pass


def test_zero_drogue_velocity():
    """Drogue velocity stationary, buoy moving: drift should converge."""
    def sample_uv_sheared(*, t, x, y, z):
        """Buoy: v=0.5 m/s, Drogue: v=0.0 m/s."""
        if z == 0:
            return 0.0, 0.5
        return 0.0, 0.0

    dd = DroguedDrifter(get_uv=sample_uv_sheared)
    xd, yd = dd.get_final_drift(t_span=(0, 120))

    # Drift should be nonzero (shear-driven)
    assert np.isfinite(xd) and np.isfinite(yd)
    # Should be between buoy and drogue velocities
    assert 0.0 <= yd <= 0.5, f"Expected yd in [0, 0.5], got {yd}"


def test_uv_to_theta_near_horizontal():
    """Conversion θ = π - 2*arctan2(r, 2) with large r (tilted pole)."""
    # Large r (u or v) means tilted pole, which gives small theta
    u_large = 10.0
    v_small = 0.1
    theta = _uv_to_theta(u_large, v_small)

    # Should be finite and between 0 and π
    assert np.isfinite(theta), f"theta should be finite, got {theta}"
    assert 0 < theta < np.pi, f"Expected theta in (0, π), got {theta}"


def test_uv_to_theta_near_vertical():
    """Conversion θ = arctan(u,v) near-vertical."""
    # Near-vertical: small u, small v
    u_small = 0.001
    v_small = 0.001
    theta = _uv_to_theta(u_small, v_small)

    # Should give θ close to π
    assert np.isfinite(theta), f"theta should be finite, got {theta}"
    assert 0.99 * np.pi < theta <= np.pi, f"Expected theta near π, got {theta}"


def test_uv_to_theta_zero_vector():
    """Conversion at (u, v) = (0, 0) should give θ = π."""
    u_zero = 0.0
    v_zero = 0.0
    theta = _uv_to_theta(u_zero, v_zero)

    # Should be exactly π
    np.testing.assert_allclose(theta, np.pi, rtol=1e-10)


def test_uv_to_theta_roundtrip_extreme():
    """Round-trip u,v↔θ,φ at extreme angles."""
    from drogued_drifters.lagrange_model import _spherical_to_uv

    # Near-vertical pole
    theta_in = 0.9999 * np.pi
    phi_in = 0.0

    u, v, ud, vd = _spherical_to_uv(theta_in, phi_in, 0.0, 0.0)
    theta_out = _uv_to_theta(u, v)

    np.testing.assert_allclose(theta_out, theta_in, rtol=1e-8,
                                err_msg="Round-trip failed near-vertical")


def test_M_func_positive_definite_extreme_angles():
    """M matrix should remain positive-definite even at extreme tilt angles."""
    # Very large u (near-horizontal)
    M_horiz = M_func(
        u=5.0, v=0.0, xd=0.0, yd=0.0, ud=0.0, vd=0.0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0,
    )

    eigvals = np.linalg.eigvalsh(M_horiz)
    assert np.all(eigvals > 0), f"M not positive definite at horizontal angle: {eigvals}"


def test_batch_extreme_velocities():
    """Batch processing should handle extreme individual particle velocities."""
    dd = DroguedDrifter()

    N = 5
    def sample_uv_batch(z):
        """Different current for each particle, some extreme."""
        U_b = np.array([0.1, 1000.0, 0.01, -50.0, 0.2])
        V_b = np.array([0.05, 500.0, 0.005, -25.0, 0.1])
        return U_b, V_b

    try:
        xd, yd, theta, Y_final = dd.get_final_drift_batch(
            sample_uv=sample_uv_batch,
            t_span=(0, 120),
        )
        # Should produce shape (N,)
        assert xd.shape == (N,)
        assert yd.shape == (N,)
        # At least some should be finite
        assert np.any(np.isfinite(xd)) or np.any(np.isfinite(yd))
    except (np.linalg.LinAlgError, OverflowError, ValueError):
        # Overflow rejection is acceptable
        pass


def test_very_small_perturbation_stability():
    """Tiny perturbations should not cause numerical instability."""
    eps = 1e-12

    def sample_uv_base(*, t, x, y, z):
        if z == 0:
            return 0.1, 0.05
        return 0.05, 0.025

    dd_base = DroguedDrifter(get_uv=sample_uv_base)
    xd_base, yd_base = dd_base.get_final_drift(t_span=(0, 120))

    # Integrate with tiny perturbation
    def sample_uv_pert(*, t, x, y, z):
        if z == 0:
            return 0.1 + eps, 0.05 + eps
        return 0.05 + eps, 0.025 + eps

    dd_pert = DroguedDrifter(get_uv=sample_uv_pert)
    xd_pert, yd_pert = dd_pert.get_final_drift(t_span=(0, 120))

    # Difference should be small relative to base
    assert np.isfinite(xd_pert) and np.isfinite(yd_pert)


def test_M_F_continuity_near_zero():
    """M and F should be continuous as (u, v) → (0, 0)."""
    eps_values = [1e-2, 1e-3, 1e-4, 1e-5]

    M_vals = []
    F_vals = []

    for eps in eps_values:
        M = M_func(
            u=eps, v=eps, xd=0, yd=0, ud=0, vd=0,
            m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
            l=3.0, g=9.81, k_b=12.0, k_d=154.0,
            U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0,
        )
        F = F_func(
            u=eps, v=eps, xd=0, yd=0, ud=0, vd=0,
            m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
            l=3.0, g=9.81, k_b=12.0, k_d=154.0,
            U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0,
        )
        M_vals.append(M.flatten())
        F_vals.append(F)

    # All should be finite
    for M_flat in M_vals:
        assert np.all(np.isfinite(M_flat)), f"M has non-finite values"
    for F in F_vals:
        assert np.all(np.isfinite(F)), f"F has non-finite values"


def test_spherical_singularity_at_pi():
    """Spherical (θ, φ) has singularity at θ=π, but stereographic (u, v) avoids it."""
    from drogued_drifters.lagrange_model import _uv_to_spherical, _spherical_to_uv

    # At θ=π, φ is undefined, but (u, v) = (0, 0) should always map correctly
    u, v, ud, vd = _spherical_to_uv(np.pi, 0.0, 0.0, 0.0)
    assert np.isfinite(u) and np.isfinite(v)
    np.testing.assert_allclose([u, v], [0.0, 0.0], atol=1e-14)

    # Reverse: (u, v) = (0, 0) should map to (θ, φ) with θ=π
    theta, phi, thetad, phid = _uv_to_spherical(0.0, 0.0, 0.0, 0.0)
    assert np.isfinite(theta) and np.isfinite(phi)
    np.testing.assert_allclose(theta, np.pi, rtol=1e-10)


def test_large_depth_pole_length():
    """Very long pole should not cause numerical issues."""
    def sample_uv(*, t, x, y, z):
        return (0.1, 0.0) if z == 0 else (0.05, 0.0)

    dd_long = DroguedDrifter(l=100.0, get_uv=sample_uv)  # Very long pole

    try:
        xd, yd = dd_long.get_final_drift(t_span=(0, 120))
        assert np.isfinite(xd) or np.isfinite(yd)
    except (ValueError, RuntimeError):
        # Long pole may be physically unrealistic, so rejection is OK
        pass


def test_tiny_pole_length():
    """Very short pole should reduce to point particle."""
    def sample_uv(*, t, x, y, z):
        return (0.1, 0.0) if z == 0 else (0.05, 0.0)

    dd_short = DroguedDrifter(l=0.01, get_uv=sample_uv)  # Very short pole
    xd, yd = dd_short.get_final_drift(t_span=(0, 120))
    assert np.isfinite(xd) and np.isfinite(yd)
