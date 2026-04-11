"""Tests for physical properties of the Lagrange model.

Validates:
- Mass matrix positive-definiteness (physical constraint)
- Drag force scaling (quadratic, not linear)
- Pole tilt depth mapping (_z_eff)
"""

import numpy as np
import pytest

from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.lagrange_model import (
    DrifterPhysics,
    EOMState,
    F_func,
    M_func,
    _qdd_func,
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


def test_packer_covers_all_struct_fields():
    """Every DrifterPhysics and EOMState field must appear in the lambda signature.

    The packer maps lambda parameter names to struct fields by name.
    This test catches: renamed struct fields, renamed sympy symbols,
    stale .srepr cache, or fields added to one side but not the other.
    """
    import inspect
    from drogued_drifters.lagrange_model import _get_eom_callables

    _qdd_raw, _M_raw, _F_raw, _args, _pack = _get_eom_callables()
    lambda_params = set(inspect.signature(_qdd_raw).parameters)
    struct_fields = set(DrifterPhysics._fields) | set(EOMState._fields)

    # Every struct field must appear in the lambda
    missing_from_lambda = struct_fields - lambda_params
    assert (
        not missing_from_lambda
    ), f"Struct fields not in lambda signature: {missing_from_lambda}"

    # Every lambda param must map to a struct field
    missing_from_structs = lambda_params - struct_fields
    assert (
        not missing_from_structs
    ), f"Lambda params not in any struct: {missing_from_structs}"

    # Lengths must match (no duplicates)
    assert len(lambda_params) == len(DrifterPhysics._fields) + len(EOMState._fields)


def test_packer_arg_order_matches_lambda():
    """pack_eom_args must produce values in the order the lambda expects.

    Assigns distinct integer values to each field, packs them, and verifies
    the result matches what the lambda signature demands.
    """
    import inspect
    from drogued_drifters.lagrange_model import _get_eom_callables

    _qdd_raw, _, _, _, pack = _get_eom_callables()
    lambda_params = list(inspect.signature(_qdd_raw).parameters)

    # Give each field a unique value
    physics = DrifterPhysics(
        **{f: float(i) for i, f in enumerate(DrifterPhysics._fields)}
    )
    state = EOMState(**{f: float(100 + i) for i, f in enumerate(EOMState._fields)})

    packed = pack(physics, state)
    field_to_val = {**physics._asdict(), **state._asdict()}
    expected = tuple(field_to_val[name] for name in lambda_params)

    assert (
        packed == expected
    ), f"Packer ordering mismatch.\n  Got:      {packed}\n  Expected: {expected}"


def test_mass_matrix_nonsingular_at_equilibrium():
    """Mass matrix at (u, v) = (0, 0) should be nonsingular (no phi singularity).

    Tests that the stereographic coordinate system avoids the phi singularity
    that exists in spherical coordinates at theta=pi. The mass matrix should
    be well-conditioned at equilibrium.
    """
    M = M_func(
        _DEFAULT_PHYSICS,
        EOMState(u_stereo=0, v_stereo=0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
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
        dict(u_stereo=0, v_stereo=0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # Small tilt
        dict(u_stereo=0.1, v_stereo=0.05, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # Large tilt
        dict(u_stereo=1.0, v_stereo=0.5, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # With nonzero velocities
        dict(
            u_stereo=0.2,
            v_stereo=-0.1,
            xd=0.1,
            yd=-0.05,
            ud_stereo=0.01,
            vd_stereo=-0.02,
            U_b=0.5,
            V_b=-0.3,
            U_d=0.2,
            V_d=0.1,
        ),
        # Extreme: near-horizontal pole
        dict(u_stereo=2.0, v_stereo=1.5, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
    ]

    for pt in test_points:
        M = M_func(
            _DEFAULT_PHYSICS,
            EOMState(
                u_stereo=pt["u_stereo"],
                v_stereo=pt["v_stereo"],
                xd=pt["xd"],
                yd=pt["yd"],
                ud_stereo=pt["ud_stereo"],
                vd_stereo=pt["vd_stereo"],
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
            u_stereo=u, v_stereo=v, xd=xd, yd=yd, ud_stereo=ud, vd_stereo=vd, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d
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

    # Setup: equilibrium pole (u_stereo=0, v_stereo=0) with varying buoy current
    def test_force_scaling(U_b_test):
        """Helper: compute F at given buoy current."""
        return F_func(
            _DEFAULT_PHYSICS,
            EOMState(
                u_stereo=0.0,
                v_stereo=0.0,  # Equilibrium orientation
                xd=0.0,
                yd=0.0,  # No buoy motion
                ud_stereo=0.0,
                vd_stereo=0.0,
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
            u_stereo=0.0,
            v_stereo=0.0,
            xd=0.0,
            yd=0.0,
            ud_stereo=0.0,
            vd_stereo=0.0,
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


def test_z_eff_range():
    """Effective drogue vertical position should be in [-l, 0] where l is pole length."""
    dd = DroguedDrifter(l=3.0)

    N = 20
    rng = np.random.default_rng(43)
    u = rng.uniform(-2, 2, N)
    v = rng.uniform(-2, 2, N)

    z_eff = dd._z_eff(u, v)

    # Should have shape (N,)
    assert z_eff.shape == (N,)

    # All values should be in [-3.0, 0] (z-up: non-positive, magnitude <= l)
    assert np.all(z_eff <= 0.0), f"z_eff has positive values: {z_eff[z_eff > 0]}"
    assert np.all(z_eff >= -3.0), f"z_eff exceeds pole length: {z_eff[z_eff < -3.0]}"


def test_z_eff_equilibrium():
    """At equilibrium (u, v)=(0, 0), drogue should hang straight down: z_eff = -l."""
    dd = DroguedDrifter(l=3.0)

    N = 5
    u = np.zeros(N)
    v = np.zeros(N)

    z_eff = dd._z_eff(u, v)

    np.testing.assert_allclose(
        z_eff,
        -3.0,
        rtol=1e-10,
        err_msg="At equilibrium, z_eff should equal -l = -3.0 (z-up)",
    )


def test_z_eff_tilt_increases_depth():
    """Tilting pole away from vertical should make drogue shallower (less negative z_eff).

    At (u, v)=(0, 0), pole is vertical, z_eff = -l.
    As pole tilts (u, v increase), cos(theta) becomes less negative,
    so z_eff = l*cos(theta) increases toward 0 (shallower).
    """
    dd = DroguedDrifter(l=3.0)

    # Small tilt (pass as arrays to use batch mode)
    u_small = np.array([0.1])
    v_small = np.array([0.0])
    z_small = dd._z_eff(u_small, v_small)[0]

    # Large tilt
    u_large = np.array([1.0])
    v_large = np.array([0.0])
    z_large = dd._z_eff(u_large, v_large)[0]

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
        u = 2 * np.tan(delta / 2)  # phi=0, so v_stereo=0
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
            u_stereo=0.0,
            v_stereo=0.0,
            xd=0.0,
            yd=0.0,
            ud_stereo=0.0,
            vd_stereo=0.0,
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


# ---------------------------------------------------------------------------
# qdd correctness: _qdd_func must match M_func + F_func + np.linalg.solve
# ---------------------------------------------------------------------------


def test_qdd_func_matches_M_F_solve_scalar():
    """_qdd_func must agree with M_func + F_func + np.linalg.solve at several points."""
    test_points = [
        dict(u_stereo=0, v_stereo=0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(
            u_stereo=0.1, v_stereo=0.05, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1
        ),
        dict(
            u_stereo=0.3,
            v_stereo=-0.2,
            xd=0.1,
            yd=-0.05,
            ud_stereo=0.01,
            vd_stereo=-0.02,
            U_b=0.5,
            V_b=-0.3,
            U_d=0.2,
            V_d=0.1,
        ),
        dict(
            u_stereo=2.0, v_stereo=0.0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=1.0, V_b=1.0, U_d=-1.0, V_d=-1.0
        ),
    ]

    for pt in test_points:
        state = EOMState(**pt)
        qdd = _qdd_func(_DEFAULT_PHYSICS, state)
        M = M_func(_DEFAULT_PHYSICS, state)
        F = F_func(_DEFAULT_PHYSICS, state)
        qdd_ref = np.linalg.solve(M, F)

        np.testing.assert_allclose(
            qdd,
            qdd_ref,
            atol=1e-12,
            err_msg=f"qdd mismatch at {pt}",
        )


def test_qdd_func_matches_M_F_solve_batch():
    """_qdd_func batch must agree with per-particle M_func + F_func + solve."""
    N = 10
    rng = np.random.default_rng(42)
    state = EOMState(
        u_stereo=rng.uniform(-1, 1, N),
        v_stereo=rng.uniform(-1, 1, N),
        xd=rng.uniform(-0.5, 0.5, N),
        yd=rng.uniform(-0.5, 0.5, N),
        ud_stereo=rng.uniform(-0.1, 0.1, N),
        vd_stereo=rng.uniform(-0.1, 0.1, N),
        U_b=rng.uniform(-0.5, 0.5, N),
        V_b=rng.uniform(-0.5, 0.5, N),
        U_d=rng.uniform(-0.5, 0.5, N),
        V_d=rng.uniform(-0.5, 0.5, N),
    )

    qdd_batch = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd_batch.shape == (N, 4)

    M_batch = M_func(_DEFAULT_PHYSICS, state)
    F_batch = F_func(_DEFAULT_PHYSICS, state)

    for i in range(N):
        qdd_ref = np.linalg.solve(M_batch[i], F_batch[i])
        np.testing.assert_allclose(
            qdd_batch[i],
            qdd_ref,
            atol=1e-12,
            err_msg=f"qdd mismatch at particle {i}",
        )


# ---------------------------------------------------------------------------
# Broadcasting contract tests
# ---------------------------------------------------------------------------


def test_lambdify_scalar_input():
    """Scalar args in, scalar results out (baseline)."""
    state = EOMState(
        u_stereo=0.1,
        v_stereo=0.05,
        xd=0.0,
        yd=0.0,
        ud_stereo=0.0,
        vd_stereo=0.0,
        U_b=0.5,
        V_b=-0.3,
        U_d=0.2,
        V_d=0.1,
    )
    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (4,), f"Expected (4,), got {qdd.shape}"
    assert np.all(np.isfinite(qdd))

    M = M_func(_DEFAULT_PHYSICS, state)
    assert M.shape == (4, 4)

    F = F_func(_DEFAULT_PHYSICS, state)
    assert F.shape == (4,)


def test_lambdify_batch_input():
    """(N,) arrays in, batch results out."""
    N = 20
    rng = np.random.default_rng(123)
    state = EOMState(
        u_stereo=rng.uniform(-1, 1, N),
        v_stereo=rng.uniform(-1, 1, N),
        xd=rng.uniform(-0.5, 0.5, N),
        yd=rng.uniform(-0.5, 0.5, N),
        ud_stereo=rng.uniform(-0.1, 0.1, N),
        vd_stereo=rng.uniform(-0.1, 0.1, N),
        U_b=rng.uniform(-0.5, 0.5, N),
        V_b=rng.uniform(-0.5, 0.5, N),
        U_d=rng.uniform(-0.5, 0.5, N),
        V_d=rng.uniform(-0.5, 0.5, N),
    )

    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (N, 4), f"Expected ({N}, 4), got {qdd.shape}"
    assert np.all(np.isfinite(qdd))

    M = M_func(_DEFAULT_PHYSICS, state)
    assert M.shape == (N, 4, 4)

    F = F_func(_DEFAULT_PHYSICS, state)
    assert F.shape == (N, 4)


def test_lambdify_batch_matches_scalar_loop():
    """Batch result must match scalar-per-point loop."""
    N = 10
    rng = np.random.default_rng(99)
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

    batch_state = EOMState(
        u_stereo=u, v_stereo=v, xd=xd, yd=yd, ud_stereo=ud, vd_stereo=vd, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d
    )
    qdd_batch = _qdd_func(_DEFAULT_PHYSICS, batch_state)

    for i in range(N):
        scalar_state = EOMState(
            u_stereo=u[i],
            v_stereo=v[i],
            xd=xd[i],
            yd=yd[i],
            ud_stereo=ud[i],
            vd_stereo=vd[i],
            U_b=U_b[i],
            V_b=V_b[i],
            U_d=U_d[i],
            V_d=V_d[i],
        )
        qdd_i = _qdd_func(_DEFAULT_PHYSICS, scalar_state)
        np.testing.assert_allclose(
            qdd_batch[i],
            qdd_i,
            atol=1e-14,
            err_msg=f"Batch vs scalar mismatch at particle {i}",
        )


def test_lambdify_mixed_scalar_array_broadcast():
    """Physics args are scalars, state args are (N,) arrays."""
    N = 5
    state = EOMState(
        u_stereo=np.full(N, 0.1),
        v_stereo=np.full(N, 0.05),
        xd=np.zeros(N),
        yd=np.zeros(N),
        ud_stereo=np.zeros(N),
        vd_stereo=np.zeros(N),
        U_b=np.full(N, 0.5),
        V_b=np.full(N, -0.3),
        U_d=np.full(N, 0.2),
        V_d=np.full(N, 0.1),
    )

    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (N, 4)

    # All particles have identical input, so all rows should be equal
    for i in range(1, N):
        np.testing.assert_allclose(qdd[i], qdd[0], atol=1e-14)


def test_lambdify_cse_preserves_broadcasting():
    """Compare lambdify(cse=True) vs lambdify(cse=False) on batch input.

    If CSE breaks broadcasting, this catches it.
    """
    import sympy as sp
    from drogued_drifters.lagrange_model import _load_or_derive

    _, _, qdd_exprs, args = _load_or_derive()

    qdd_cse = sp.lambdify(args, qdd_exprs, modules="numpy", cse=True)
    qdd_nocse = sp.lambdify(args, qdd_exprs, modules="numpy", cse=False)

    from drogued_drifters.lagrange_model import _build_packer

    pack = _build_packer(qdd_cse)

    N = 10
    rng = np.random.default_rng(77)
    state = EOMState(
        u_stereo=rng.uniform(-1, 1, N),
        v_stereo=rng.uniform(-1, 1, N),
        xd=rng.uniform(-0.5, 0.5, N),
        yd=rng.uniform(-0.5, 0.5, N),
        ud_stereo=rng.uniform(-0.1, 0.1, N),
        vd_stereo=rng.uniform(-0.1, 0.1, N),
        U_b=rng.uniform(-0.5, 0.5, N),
        V_b=rng.uniform(-0.5, 0.5, N),
        U_d=rng.uniform(-0.5, 0.5, N),
        V_d=rng.uniform(-0.5, 0.5, N),
    )

    packed = pack(_DEFAULT_PHYSICS, state)
    result_cse = np.column_stack(qdd_cse(*packed))
    result_nocse = np.column_stack(qdd_nocse(*packed))

    np.testing.assert_allclose(
        result_cse,
        result_nocse,
        atol=1e-12,
        err_msg="CSE vs no-CSE mismatch on batch input",
    )


def test_lambdify_batch_N1():
    """Single-element (N=1) arrays: common source of shape bugs."""
    state = EOMState(
        u_stereo=np.array([0.1]),
        v_stereo=np.array([0.05]),
        xd=np.array([0.0]),
        yd=np.array([0.0]),
        ud_stereo=np.array([0.0]),
        vd_stereo=np.array([0.0]),
        U_b=np.array([0.5]),
        V_b=np.array([-0.3]),
        U_d=np.array([0.2]),
        V_d=np.array([0.1]),
    )

    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (1, 4), f"Expected (1, 4), got {qdd.shape}"
    assert np.all(np.isfinite(qdd))

    M = M_func(_DEFAULT_PHYSICS, state)
    assert M.shape == (1, 4, 4)

    F = F_func(_DEFAULT_PHYSICS, state)
    assert F.shape == (1, 4)
