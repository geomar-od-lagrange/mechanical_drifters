"""Tests for physical properties of the Lagrange model.

Validates:
- Mass matrix positive-definiteness (physical constraint)
- Drag force scaling (quadratic, not linear)
- Pole tilt depth mapping (_z_eff_batch)
"""

import numpy as np
import pytest

from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.lagrange_model import (
    DrifterPhysics,
    EOMState,
    F_func,
    M_func,
    _uv_to_theta,
)

_DEFAULT_PHYSICS = DrifterPhysics(
    m_b=1.0,
    m_d=2.7,
    m_hat_d=1.0,
    m_tilde_d=101.0,
    m_tilde_b=1.9,
    l=3.0,
    g=9.81,
    k_b=12.0,
    k_d=154.0,
)


def test_mass_matrix_nonsingular_at_equilibrium():
    """Mass matrix at (u, v) = (0, 0) should be nonsingular (no phi singularity).

    Tests that the stereographic coordinate system avoids the phi singularity
    that exists in spherical coordinates at theta=pi. The mass matrix should
    be well-conditioned at equilibrium.
    """
    M = M_func(
        _DEFAULT_PHYSICS,
        EOMState(u=0, v=0, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
    )
    # Should be well-conditioned (no near-zero eigenvalues)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(
        eigvals > 0
    ), f"Mass matrix not positive definite: eigenvalues = {eigvals}"


def test_mass_matrix_positive_definite_scalar():
    """Mass matrix must be positive definite at all test points.

    Positive-definiteness is a physical constraint: kinetic energy > 0 for
    any nonzero velocity.
    """
    test_points = [
        # Equilibrium
        dict(u=0, v=0, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # Small tilt
        dict(u=0.1, v=0.05, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # Large tilt
        dict(u=1.0, v=0.5, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # With nonzero velocities
        dict(
            u=0.2,
            v=-0.1,
            xd=0.1,
            yd=-0.05,
            ud=0.01,
            vd=-0.02,
            U_b=0.5,
            V_b=-0.3,
            U_d=0.2,
            V_d=0.1,
        ),
        # Extreme: near-horizontal pole
        dict(u=2.0, v=1.5, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
    ]

    for pt in test_points:
        M = M_func(
            _DEFAULT_PHYSICS,
            EOMState(
                u=pt["u"],
                v=pt["v"],
                xd=pt["xd"],
                yd=pt["yd"],
                ud=pt["ud"],
                vd=pt["vd"],
                U_b=pt["U_b"],
                V_b=pt["V_b"],
                U_d=pt["U_d"],
                V_d=pt["V_d"],
            ),
        )

        # Check positive-definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), (
            f"M not positive definite at {pt}. " f"Eigenvalues: {eigvals}"
        )


def test_mass_matrix_positive_definite_batch():
    """Mass matrix positive-definiteness for N particles."""
    N = 10
    rng = np.random.default_rng(42)
    u = rng.uniform(-1, 1, N)
    v = rng.uniform(-1, 1, N)
    xd = rng.uniform(-0.5, 0.5, N)
    yd = rng.uniform(-0.5, 0.5, N)
    ud = rng.uniform(-0.1, 0.1, N)
    vd = rng.uniform(-0.1, 0.1, N)
    U_b = rng.uniform(-0.5, 0.5, N)
    V_b = rng.uniform(-0.5, 0.5, N)
    U_d = rng.uniform(-0.5, 0.5, N)
    V_d = rng.uniform(-0.5, 0.5, N)

    M_batch = M_func(
        _DEFAULT_PHYSICS,
        EOMState(
            u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d
        ),
    )

    # M_batch has shape (N, 4, 4)
    assert M_batch.shape == (N, 4, 4)

    # Check each particle's M
    for i in range(N):
        eigvals = np.linalg.eigvalsh(M_batch[i])
        assert np.all(
            eigvals > 0
        ), f"M[{i}] not positive definite. Eigenvalues: {eigvals}"


def test_drag_force_quadratic_scaling():
    """Drag forces should scale as |v|*v (quadratic), not linear.

    At equilibrium with zero velocities, forces should be zero.
    If we perturb the current velocity, force should scale ~v^2, not ~v.

    Test: Compare drag at v0 and 2*v0. For quadratic drag:
        F(2*v) / F(v) ~ (2*v)^2 / v^2 = 4
    """

    # Setup: equilibrium pole (u=0, v=0) with varying buoy current
    def test_force_scaling(U_b_test):
        """Helper: compute F at given buoy current."""
        return F_func(
            _DEFAULT_PHYSICS,
            EOMState(
                u=0.0,
                v=0.0,  # Equilibrium orientation
                xd=0.0,
                yd=0.0,  # No buoy motion
                ud=0.0,
                vd=0.0,
                U_b=U_b_test,
                V_b=0.0,  # Vary buoy current
                U_d=0.0,
                V_d=0.0,  # Still drogue
            ),
        )

    # Test scaling for small perturbations in current
    eps = 0.1
    F1 = test_force_scaling(eps)
    F2 = test_force_scaling(2 * eps)

    # F ~ |v|*v, so F(2*v) / F(v) ~ 4 (approximately, for small v)
    ratio = np.linalg.norm(F2) / np.linalg.norm(F1)

    # Allow some tolerance (not exactly 4 due to nonlinearity in drag)
    assert 3.0 < ratio < 5.0, (
        f"Drag force scaling not quadratic. Ratio F(2*eps)/F(eps) = {ratio}, "
        f"expected ~4 for quadratic drag."
    )


def test_zero_velocity_zero_force():
    """With zero buoy/drogue currents, force should be zero at equilibrium.

    At (u, v)=(0, 0) with all velocities zero and zero currents:
    F should be zero (no forces acting).
    """
    F = F_func(
        _DEFAULT_PHYSICS,
        EOMState(
            u=0.0,
            v=0.0,
            xd=0.0,
            yd=0.0,
            ud=0.0,
            vd=0.0,
            U_b=0.0,
            V_b=0.0,
            U_d=0.0,
            V_d=0.0,
        ),
    )

    np.testing.assert_allclose(
        F,
        0.0,
        atol=1e-12,
        err_msg="Force should be zero at equilibrium with zero currents",
    )


def test_z_eff_batch_range():
    """Effective drogue vertical position should be in [-l, 0] where l is pole length."""
    dd = DroguedDrifter(l=3.0)

    N = 20
    rng = np.random.default_rng(43)
    u = rng.uniform(-2, 2, N)
    v = rng.uniform(-2, 2, N)

    z_eff = dd._z_eff_batch(u, v)

    # Should have shape (N,)
    assert z_eff.shape == (N,)

    # All values should be in [-3.0, 0] (z-up: non-positive, magnitude <= l)
    assert np.all(z_eff <= 0.0), f"z_eff has positive values: {z_eff[z_eff > 0]}"
    assert np.all(z_eff >= -3.0), f"z_eff exceeds pole length: {z_eff[z_eff < -3.0]}"


def test_z_eff_batch_equilibrium():
    """At equilibrium (u, v)=(0, 0), drogue should hang straight down: z_eff = -l."""
    dd = DroguedDrifter(l=3.0)

    N = 5
    u = np.zeros(N)
    v = np.zeros(N)

    z_eff = dd._z_eff_batch(u, v)

    np.testing.assert_allclose(
        z_eff,
        -3.0,
        rtol=1e-10,
        err_msg="At equilibrium, z_eff should equal -l = -3.0 (z-up)",
    )


def test_z_eff_batch_tilt_increases_depth():
    """Tilting pole away from vertical should make drogue shallower (less negative z_eff).

    At (u, v)=(0, 0), pole is vertical, z_eff = -l.
    As pole tilts (u, v increase), cos(theta) becomes less negative,
    so z_eff = l*cos(theta) increases toward 0 (shallower).
    """
    dd = DroguedDrifter(l=3.0)

    # Small tilt (pass as arrays to use batch mode)
    u_small = np.array([0.1])
    v_small = np.array([0.0])
    z_small = dd._z_eff_batch(u_small, v_small)[0]

    # Large tilt
    u_large = np.array([1.0])
    v_large = np.array([0.0])
    z_large = dd._z_eff_batch(u_large, v_large)[0]

    # Larger tilt should give shallower depth (larger z_eff, i.e. less negative)
    assert z_large > z_small, (
        f"Expected z_eff to increase (become shallower) with tilt, "
        f"but z({u_small[0]})={z_small} vs z({u_large[0]})={z_large}"
    )


def test_uv_to_theta_inversion():
    """Test that _uv_to_theta inverts the stereographic projection correctly."""
    # Test at several known angles
    test_angles = [np.pi, 0.9 * np.pi, 0.8 * np.pi, 0.7 * np.pi, np.pi / 2]

    for theta_expected in test_angles:
        # Convert theta to (u, v) via stereographic projection
        delta = np.pi - theta_expected
        u = 2 * np.tan(delta / 2)  # phi=0, so v=0
        v = 0.0

        # Convert back
        theta_actual = _uv_to_theta(u, v)

        np.testing.assert_allclose(
            theta_actual,
            theta_expected,
            rtol=1e-10,
            err_msg=f"Round-trip failed for theta={theta_expected}",
        )


def test_no_singularity_at_equilibrium():
    """Stereographic coordinates avoid phi singularity at theta=pi (equilibrium).

    The spherical parameterization (theta, phi) has a singularity at theta=pi
    (drogue hangs straight down), but stereographic (u, v) doesn't.
    Verify M is well-conditioned at (u, v)=(0, 0).
    """
    M = M_func(
        _DEFAULT_PHYSICS,
        EOMState(
            u=0.0,
            v=0.0,
            xd=0.0,
            yd=0.0,
            ud=0.0,
            vd=0.0,
            U_b=0.0,
            V_b=0.0,
            U_d=0.0,
            V_d=0.0,
        ),
    )

    # Compute condition number
    eigvals = np.linalg.eigvalsh(M)
    cond = np.max(eigvals) / np.min(eigvals)

    # Should not be ill-conditioned (would be ~infinity with singularity)
    assert cond < 1e6, (
        f"M is ill-conditioned at equilibrium: cond={cond}. "
        f"Suggests singularity issue. Eigenvalues: {eigvals}"
    )
