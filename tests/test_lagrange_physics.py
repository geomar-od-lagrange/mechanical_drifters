"""Tests for physical properties of the Lagrange model.

Validates:
- Mass matrix positive-definiteness (physical constraint)
- Drag force scaling (quadratic, not linear)
- Pole tilt depth mapping (_z_eff)
"""

import numpy as np
import pytest

from conftest import DEFAULT_PHYSICS as _DEFAULT_PHYSICS

from drogued_drifters.models.drogued_drifter import DroguedDrifter, DrifterPhysics, EOMState
from drogued_drifters.eom import eval_M, eval_F, _make_qdd_func
from drogued_drifters.coords import _uv_to_theta


dd_singleton = DroguedDrifter()
_qdd_func = _make_qdd_func(dd_singleton, "numpy")


def test_packer_covers_all_struct_fields():
    """Every DrifterPhysics and EOMState field must appear in the lambda signature."""
    import inspect
    from drogued_drifters.eom import _get_eom_callables

    dd = DroguedDrifter()
    _qdd_raw, _M_raw, _F_raw, _pack = _get_eom_callables(dd)
    lambda_params = set(inspect.signature(_qdd_raw).parameters)
    struct_fields = set(DrifterPhysics._fields) | set(EOMState._fields)

    missing_from_lambda = struct_fields - lambda_params
    assert (
        not missing_from_lambda
    ), f"Struct fields not in lambda signature: {missing_from_lambda}"

    missing_from_structs = lambda_params - struct_fields
    assert (
        not missing_from_structs
    ), f"Lambda params not in any struct: {missing_from_structs}"

    assert len(lambda_params) == len(DrifterPhysics._fields) + len(EOMState._fields)


def test_packer_arg_order_matches_lambda():
    """pack_eom_args must produce values in the order the lambda expects."""
    import inspect
    from drogued_drifters.eom import _get_eom_callables

    dd = DroguedDrifter()
    _qdd_raw, _, _, pack = _get_eom_callables(dd)
    lambda_params = list(inspect.signature(_qdd_raw).parameters)

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
    """Mass matrix at (u, v) = (0, 0) should be nonsingular."""
    dd = DroguedDrifter()
    M = eval_M(
        dd, _DEFAULT_PHYSICS,
        EOMState(u_stereo=0, v_stereo=0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
    )
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0), f"Mass matrix not positive definite: eigenvalues = {eigvals}"


def test_mass_matrix_positive_definite_scalar():
    """Mass matrix must be positive definite at all test points."""
    dd = DroguedDrifter()
    test_points = [
        dict(u_stereo=0, v_stereo=0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(u_stereo=0.1, v_stereo=0.05, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(u_stereo=1.0, v_stereo=0.5, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(
            u_stereo=0.2, v_stereo=-0.1,
            xd=0.1, yd=-0.05,
            ud_stereo=0.01, vd_stereo=-0.02,
            U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
        ),
        dict(u_stereo=2.0, v_stereo=1.5, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
    ]

    for pt in test_points:
        M = eval_M(dd, _DEFAULT_PHYSICS, EOMState(**pt))
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), (
            f"M not positive definite at {pt}. Eigenvalues: {eigvals}"
        )


def test_mass_matrix_positive_definite_batch():
    """Mass matrix positive-definiteness for N particles."""
    dd = DroguedDrifter()
    N = 10
    rng = np.random.default_rng(42)

    M_batch = eval_M(
        dd, _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=rng.uniform(-1, 1, N), v_stereo=rng.uniform(-1, 1, N),
            xd=rng.uniform(-0.5, 0.5, N), yd=rng.uniform(-0.5, 0.5, N),
            ud_stereo=rng.uniform(-0.1, 0.1, N), vd_stereo=rng.uniform(-0.1, 0.1, N),
            U_b=rng.uniform(-0.5, 0.5, N), V_b=rng.uniform(-0.5, 0.5, N),
            U_d=rng.uniform(-0.5, 0.5, N), V_d=rng.uniform(-0.5, 0.5, N),
        ),
    )

    assert M_batch.shape == (N, 4, 4)
    for i in range(N):
        eigvals = np.linalg.eigvalsh(M_batch[i])
        assert np.all(eigvals > 0), f"M[{i}] not positive definite. Eigenvalues: {eigvals}"


def test_drag_force_quadratic_scaling():
    """Drag forces should scale as |v|*v (quadratic), not linear."""
    dd = DroguedDrifter()

    def test_force_scaling(U_b_test):
        return eval_F(
            dd, _DEFAULT_PHYSICS,
            EOMState(
                u_stereo=0.0, v_stereo=0.0,
                xd=0.0, yd=0.0,
                ud_stereo=0.0, vd_stereo=0.0,
                U_b=U_b_test, V_b=0.0,
                U_d=0.0, V_d=0.0,
            ),
        )

    eps = 0.1
    F1 = test_force_scaling(eps)
    F2 = test_force_scaling(2 * eps)

    ratio = np.linalg.norm(F2) / np.linalg.norm(F1)
    assert 3.0 < ratio < 5.0, (
        f"Drag force scaling not quadratic. Ratio F(2*eps)/F(eps) = {ratio}, "
        f"expected ~4 for quadratic drag."
    )


def test_zero_velocity_zero_force():
    """With zero currents, force should be zero at equilibrium."""
    dd = DroguedDrifter()
    F = eval_F(
        dd, _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=0.0, v_stereo=0.0,
            xd=0.0, yd=0.0,
            ud_stereo=0.0, vd_stereo=0.0,
            U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0,
        ),
    )
    np.testing.assert_allclose(F, 0.0, atol=1e-12, err_msg="Force should be zero at equilibrium with zero currents")


def test_z_eff_range():
    """Effective drogue vertical position should be in [-l, 0]."""
    dd = DroguedDrifter(l=3.0)

    N = 20
    rng = np.random.default_rng(43)
    u = rng.uniform(-2, 2, N)
    v = rng.uniform(-2, 2, N)

    z_eff = dd._z_eff(u, v)

    assert z_eff.shape == (N,)
    assert np.all(z_eff <= 0.0), f"z_eff has positive values: {z_eff[z_eff > 0]}"
    assert np.all(z_eff >= -3.0), f"z_eff exceeds pole length: {z_eff[z_eff < -3.0]}"


def test_z_eff_equilibrium():
    """At equilibrium (u, v)=(0, 0), drogue should hang straight down: z_eff = -l."""
    dd = DroguedDrifter(l=3.0)

    N = 5
    u = np.zeros(N)
    v = np.zeros(N)

    z_eff = dd._z_eff(u, v)

    np.testing.assert_allclose(z_eff, -3.0, rtol=1e-10, err_msg="At equilibrium, z_eff should equal -l = -3.0")


def test_z_eff_tilt_increases_depth():
    """Tilting pole should make drogue shallower (less negative z_eff)."""
    dd = DroguedDrifter(l=3.0)

    u_small = np.array([0.1])
    v_small = np.array([0.0])
    z_small = dd._z_eff(u_small, v_small)[0]

    u_large = np.array([1.0])
    v_large = np.array([0.0])
    z_large = dd._z_eff(u_large, v_large)[0]

    assert z_large > z_small, (
        f"Expected z_eff to increase (become shallower) with tilt, "
        f"but z({u_small[0]})={z_small} vs z({u_large[0]})={z_large}"
    )


def test_uv_to_theta_inversion():
    """Test that _uv_to_theta inverts the stereographic projection correctly."""
    test_angles = [np.pi, 0.9 * np.pi, 0.8 * np.pi, 0.7 * np.pi, np.pi / 2]

    for theta_expected in test_angles:
        delta = np.pi - theta_expected
        u = 2 * np.tan(delta / 2)
        v = 0.0

        theta_actual = _uv_to_theta(u, v)

        np.testing.assert_allclose(
            theta_actual, theta_expected, rtol=1e-10,
            err_msg=f"Round-trip failed for theta={theta_expected}",
        )


def test_no_singularity_at_equilibrium():
    """Stereographic coordinates avoid phi singularity at theta=pi."""
    dd = DroguedDrifter()
    M = eval_M(
        dd, _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=0.0, v_stereo=0.0,
            xd=0.0, yd=0.0,
            ud_stereo=0.0, vd_stereo=0.0,
            U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0,
        ),
    )

    eigvals = np.linalg.eigvalsh(M)
    cond = np.max(eigvals) / np.min(eigvals)

    assert cond < 1e6, (
        f"M is ill-conditioned at equilibrium: cond={cond}. Eigenvalues: {eigvals}"
    )


# ---------------------------------------------------------------------------
# qdd correctness: _qdd_func must match eval_M + eval_F + np.linalg.solve
# ---------------------------------------------------------------------------


def test_qdd_func_matches_M_F_solve_scalar():
    """_qdd_func must agree with eval_M + eval_F + np.linalg.solve."""
    dd = DroguedDrifter()
    test_points = [
        dict(u_stereo=0, v_stereo=0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(
            u_stereo=0.1, v_stereo=0.05, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1
        ),
        dict(
            u_stereo=0.3, v_stereo=-0.2,
            xd=0.1, yd=-0.05,
            ud_stereo=0.01, vd_stereo=-0.02,
            U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
        ),
        dict(
            u_stereo=2.0, v_stereo=0.0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=1.0, V_b=1.0, U_d=-1.0, V_d=-1.0
        ),
    ]

    for pt in test_points:
        state = EOMState(**pt)
        qdd = _qdd_func(_DEFAULT_PHYSICS, state)
        M = eval_M(dd, _DEFAULT_PHYSICS, state)
        F = eval_F(dd, _DEFAULT_PHYSICS, state)
        qdd_ref = np.linalg.solve(M, F)

        np.testing.assert_allclose(qdd, qdd_ref, atol=1e-12, err_msg=f"qdd mismatch at {pt}")


def test_qdd_func_matches_M_F_solve_batch():
    """_qdd_func batch must agree with per-particle eval_M + eval_F + solve."""
    dd = DroguedDrifter()
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

    M_batch = eval_M(dd, _DEFAULT_PHYSICS, state)
    F_batch = eval_F(dd, _DEFAULT_PHYSICS, state)

    for i in range(N):
        qdd_ref = np.linalg.solve(M_batch[i], F_batch[i])
        np.testing.assert_allclose(qdd_batch[i], qdd_ref, atol=1e-12, err_msg=f"qdd mismatch at particle {i}")


# ---------------------------------------------------------------------------
# Broadcasting contract tests
# ---------------------------------------------------------------------------


def test_lambdify_scalar_input():
    """Scalar args in, scalar results out."""
    dd = DroguedDrifter()
    state = EOMState(
        u_stereo=0.1, v_stereo=0.05,
        xd=0.0, yd=0.0,
        ud_stereo=0.0, vd_stereo=0.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )
    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (4,), f"Expected (4,), got {qdd.shape}"
    assert np.all(np.isfinite(qdd))

    M = eval_M(dd, _DEFAULT_PHYSICS, state)
    assert M.shape == (4, 4)

    F = eval_F(dd, _DEFAULT_PHYSICS, state)
    assert F.shape == (4,)


def test_lambdify_batch_input():
    """(N,) arrays in, batch results out."""
    dd = DroguedDrifter()
    N = 20
    rng = np.random.default_rng(123)
    state = EOMState(
        u_stereo=rng.uniform(-1, 1, N), v_stereo=rng.uniform(-1, 1, N),
        xd=rng.uniform(-0.5, 0.5, N), yd=rng.uniform(-0.5, 0.5, N),
        ud_stereo=rng.uniform(-0.1, 0.1, N), vd_stereo=rng.uniform(-0.1, 0.1, N),
        U_b=rng.uniform(-0.5, 0.5, N), V_b=rng.uniform(-0.5, 0.5, N),
        U_d=rng.uniform(-0.5, 0.5, N), V_d=rng.uniform(-0.5, 0.5, N),
    )

    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (N, 4), f"Expected ({N}, 4), got {qdd.shape}"
    assert np.all(np.isfinite(qdd))

    M = eval_M(dd, _DEFAULT_PHYSICS, state)
    assert M.shape == (N, 4, 4)

    F = eval_F(dd, _DEFAULT_PHYSICS, state)
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
            u_stereo=u[i], v_stereo=v[i],
            xd=xd[i], yd=yd[i],
            ud_stereo=ud[i], vd_stereo=vd[i],
            U_b=U_b[i], V_b=V_b[i], U_d=U_d[i], V_d=V_d[i],
        )
        qdd_i = _qdd_func(_DEFAULT_PHYSICS, scalar_state)
        np.testing.assert_allclose(qdd_batch[i], qdd_i, atol=1e-14, err_msg=f"Batch vs scalar mismatch at particle {i}")


def test_lambdify_mixed_scalar_array_broadcast():
    """Physics args are scalars, state args are (N,) arrays."""
    N = 5
    state = EOMState(
        u_stereo=np.full(N, 0.1), v_stereo=np.full(N, 0.05),
        xd=np.zeros(N), yd=np.zeros(N),
        ud_stereo=np.zeros(N), vd_stereo=np.zeros(N),
        U_b=np.full(N, 0.5), V_b=np.full(N, -0.3),
        U_d=np.full(N, 0.2), V_d=np.full(N, 0.1),
    )

    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (N, 4)

    for i in range(1, N):
        np.testing.assert_allclose(qdd[i], qdd[0], atol=1e-14)


def test_lambdify_cse_preserves_broadcasting():
    """Compare lambdify(cse=True) vs lambdify(cse=False) on batch input."""
    import sympy as sp
    from drogued_drifters.eom import _load_or_derive, _build_packer

    dd = DroguedDrifter()
    _, _, qdd_exprs, args = _load_or_derive(dd)

    qdd_cse = sp.lambdify(args, qdd_exprs, modules="numpy", cse=True)
    qdd_nocse = sp.lambdify(args, qdd_exprs, modules="numpy", cse=False)

    pack = _build_packer(qdd_cse, DrifterPhysics, EOMState)

    N = 10
    rng = np.random.default_rng(77)
    state = EOMState(
        u_stereo=rng.uniform(-1, 1, N), v_stereo=rng.uniform(-1, 1, N),
        xd=rng.uniform(-0.5, 0.5, N), yd=rng.uniform(-0.5, 0.5, N),
        ud_stereo=rng.uniform(-0.1, 0.1, N), vd_stereo=rng.uniform(-0.1, 0.1, N),
        U_b=rng.uniform(-0.5, 0.5, N), V_b=rng.uniform(-0.5, 0.5, N),
        U_d=rng.uniform(-0.5, 0.5, N), V_d=rng.uniform(-0.5, 0.5, N),
    )

    packed = pack(_DEFAULT_PHYSICS, state)
    result_cse = np.column_stack(qdd_cse(*packed))
    result_nocse = np.column_stack(qdd_nocse(*packed))

    np.testing.assert_allclose(result_cse, result_nocse, atol=1e-12, err_msg="CSE vs no-CSE mismatch on batch input")


def test_lambdify_batch_N1():
    """Single-element (N=1) arrays."""
    dd = DroguedDrifter()
    state = EOMState(
        u_stereo=np.array([0.1]), v_stereo=np.array([0.05]),
        xd=np.array([0.0]), yd=np.array([0.0]),
        ud_stereo=np.array([0.0]), vd_stereo=np.array([0.0]),
        U_b=np.array([0.5]), V_b=np.array([-0.3]),
        U_d=np.array([0.2]), V_d=np.array([0.1]),
    )

    qdd = _qdd_func(_DEFAULT_PHYSICS, state)
    assert qdd.shape == (1, 4), f"Expected (1, 4), got {qdd.shape}"
    assert np.all(np.isfinite(qdd))

    M = eval_M(dd, _DEFAULT_PHYSICS, state)
    assert M.shape == (1, 4, 4)

    F = eval_F(dd, _DEFAULT_PHYSICS, state)
    assert F.shape == (1, 4)
