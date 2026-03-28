import numpy as np

from drogued_drifters._generated_eom import compute_F, compute_M
from drogued_drifters.drifter import (
    DroguedDrifter,
    buoy_added_mass,
    buoy_drag_coeff,
    drogue_added_mass,
    drogue_drag_coeff,
)


def test_drogued_drifter_instantiation():
    dd = DroguedDrifter()


def test_MF_callable():
    assert callable(compute_M)
    assert callable(compute_F)


def test_MF_evaluates():
    dd = DroguedDrifter()

    t = 0.0
    currents = dd.get_uv(t=t, z_d=0.0, y_b=0.0, x_b=0.0)

    # Use stereographic coordinates: u=0.1, v=0.05 (small tilt from vertical)
    M, F = dd._eval_M_F(
        t,
        x=0.0,
        y=0.0,
        u=0.1,
        v=0.05,
        xd=0.0,
        yd=0.0,
        ud=0.0,
        vd=0.0,
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


def test_mass_matrix_nonsingular_at_equilibrium():
    """Mass matrix at (u, v) = (0, 0) should be nonsingular (no phi singularity)."""
    M_elems = compute_M(
        0, 0, 0, 0, 0, 0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0, V_b=0, U_d=0, V_d=0,
    )
    M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems
    M = np.array([
        [M00, M01, M02, M03],
        [M01, M11, M12, M13],
        [M02, M12, M22, M23],
        [M03, M13, M23, M33],
    ], dtype=float)
    # Should be well-conditioned (no near-zero eigenvalues)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0), f"Mass matrix not positive definite: eigenvalues = {eigvals}"


# ---------------------------------------------------------------------------
# Tests for the batched RHS path
# ---------------------------------------------------------------------------


def _make_const_uv(U_b, V_b, U_d, V_d):
    """Return a get_uv callback that always returns the given constants."""

    def _getuv(*, t, z_d, y_b, x_b):
        return U_b, V_b, U_d, V_d

    return _getuv


def test_batch_matches_scalar():
    """get_final_drift_batch must agree with the scalar path for N=5 random conditions."""
    rng = np.random.default_rng(42)
    N = 5
    U_b = rng.uniform(-0.5, 0.5, N)
    V_b = rng.uniform(-0.5, 0.5, N)
    U_d = rng.uniform(-0.5, 0.5, N)
    V_d = rng.uniform(-0.5, 0.5, N)

    t_span = (0.0, 120.0)
    theta0 = 0.999 * np.pi

    # --- batch path ---
    dd_batch = DroguedDrifter()  # default (unused) get_uv; batch path uses arrays directly
    xd_batch, yd_batch, theta_batch, _ = dd_batch.get_final_drift_batch(
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d, t_span=t_span, theta0=theta0,
    )

    # --- scalar path (one call per particle) ---
    xd_scalar = np.empty(N)
    yd_scalar = np.empty(N)
    theta_scalar = np.empty(N)

    for i in range(N):
        dd_i = DroguedDrifter(
            get_uv=_make_const_uv(U_b[i], V_b[i], U_d[i], V_d[i]),
        )
        ds = dd_i.get_final_drift(t_span=t_span, theta=theta0)
        xd_scalar[i] = float(ds.xd.isel(time=-1))
        yd_scalar[i] = float(ds.yd.isel(time=-1))
        theta_scalar[i] = float(ds.theta.isel(time=-1))

    np.testing.assert_allclose(xd_batch, xd_scalar, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(yd_batch, yd_scalar, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(theta_batch, theta_scalar, atol=1e-2, rtol=1e-2)


def test_batch_zero_currents():
    """N=10 particles with zero currents: drift velocity and theta should be zero/pi."""
    N = 10
    zeros = np.zeros(N)
    dd = DroguedDrifter()

    xd, yd, theta, _ = dd.get_final_drift_batch(
        U_b=zeros, V_b=zeros, U_d=zeros, V_d=zeros, t_span=(0.0, 120.0),
    )

    np.testing.assert_allclose(xd, 0.0, atol=0.05)
    np.testing.assert_allclose(yd, 0.0, atol=0.05)
    np.testing.assert_allclose(theta, np.pi, atol=0.05)


def test_batch_uniform_currents():
    """N=10 particles all seeing the same currents should produce identical drift."""
    N = 10
    dd = DroguedDrifter()

    xd, yd, theta, _ = dd.get_final_drift_batch(
        U_b=np.full(N, 0.3),
        V_b=np.full(N, -0.1),
        U_d=np.full(N, 0.15),
        V_d=np.full(N, -0.05),
        t_span=(0.0, 120.0),
    )

    # All particles must agree with each other
    np.testing.assert_allclose(xd, xd[0], atol=1e-10)
    np.testing.assert_allclose(yd, yd[0], atol=1e-10)
    np.testing.assert_allclose(theta, theta[0], atol=1e-10)


def test_batch_opposite_shear():
    """Two particles with swapped buoy/drogue forcing should give different drifts."""
    dd = DroguedDrifter()

    xd, yd, theta, _ = dd.get_final_drift_batch(
        U_b=np.array([0.1, 0.0]),
        V_b=np.array([0.0, 0.0]),
        U_d=np.array([0.0, 0.1]),
        V_d=np.array([0.0, 0.0]),
        t_span=(0.0, 120.0),
    )

    # The drifter model is not symmetric in buoy vs drogue forcing,
    # so the two particles must produce different drift velocities.
    assert not np.allclose(xd[0], xd[1], atol=1e-4), (
        f"Expected different xd for swapped buoy/drogue forcing, got {xd}"
    )


def test_batch_drift_between_buoy_and_drogue():
    """Drift velocity should lie between the buoy and drogue current speeds."""
    N = 1
    dd = DroguedDrifter()

    U_b_val, U_d_val = 0.2, 0.1

    xd, yd, theta, _ = dd.get_final_drift_batch(
        U_b=np.array([U_b_val]),
        V_b=np.zeros(N),
        U_d=np.array([U_d_val]),
        V_d=np.zeros(N),
        t_span=(0.0, 120.0),
    )

    # The drifter cannot go faster than the fastest layer or slower than the slowest
    assert U_d_val <= xd[0] <= U_b_val, (
        f"Expected {U_d_val} <= xd={xd[0]:.6f} <= {U_b_val}"
    )
    # y-drift should be negligible (no V forcing)
    np.testing.assert_allclose(yd[0], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Generated numpy code vs lambdified sympy reference
# ---------------------------------------------------------------------------


def _make_kwargs(u, v, xd, yd, ud, vd, U_b, V_b, U_d, V_d):
    """Build the kwargs dict for both lambdified and generated functions."""
    return dict(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )


def _positional_from_kwargs(kw):
    """Convert kwargs to positional args for the generated functions."""
    return (
        kw["u"], kw["v"], kw["xd"], kw["yd"], kw["ud"], kw["vd"],
        kw["m_b"], kw["m_d"], kw["m_hat_d"], kw["m_tilde_d"], kw["m_tilde_b"],
        kw["l"], kw["g"], kw["k_b"], kw["k_d"],
        kw["U_b"], kw["V_b"], kw["U_d"], kw["V_d"],
    )


def test_generated_vs_lambdified():
    """Generated numpy code must agree with lambdified sympy to machine precision."""
    from drogued_drifters.lagrange_model import M_func, F_func

    test_points = [
        # equilibrium: u=v=0, all velocities zero, no currents
        dict(u=0, v=0, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # small tilt
        dict(u=0.1, v=0.05, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # nonzero velocities and currents
        dict(u=0.3, v=-0.2, xd=0.1, yd=-0.05, ud=0.01, vd=-0.02,
             U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1),
        # large tilt (theta ~ pi/2)
        dict(u=2.0, v=0.0, xd=0, yd=0, ud=0, vd=0, U_b=0, V_b=0, U_d=0, V_d=0),
        # symmetric case
        dict(u=0.5, v=0.5, xd=0.1, yd=0.1, ud=0.05, vd=0.05,
             U_b=1.0, V_b=1.0, U_d=-1.0, V_d=-1.0),
    ]

    for pt in test_points:
        kw = _make_kwargs(**pt)

        # Lambdified (from sympy)
        M_lbd = np.array(M_func(**kw), dtype=float)
        F_lbd = np.array(F_func(**kw), dtype=float).reshape(-1)

        # Generated numpy
        pos = _positional_from_kwargs(kw)
        M_elems = compute_M(*pos)
        F_elems = compute_F(*pos)

        M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems
        M_gen = np.array([
            [M00, M01, M02, M03],
            [M01, M11, M12, M13],
            [M02, M12, M22, M23],
            [M03, M13, M23, M33],
        ], dtype=float)
        F_gen = np.array(F_elems, dtype=float)

        np.testing.assert_allclose(
            M_gen, M_lbd, atol=1e-12, rtol=1e-12,
            err_msg=f"M mismatch at {pt}",
        )
        np.testing.assert_allclose(
            F_gen, F_lbd, atol=1e-12, rtol=1e-12,
            err_msg=f"F mismatch at {pt}",
        )


def test_generated_vectorized():
    """Generated functions must work on (N,) arrays and match scalar results."""
    N = 5
    rng = np.random.default_rng(123)
    u = rng.uniform(-1, 1, N)
    v = rng.uniform(-1, 1, N)
    xd = rng.uniform(-0.5, 0.5, N)
    yd = rng.uniform(-0.5, 0.5, N)
    ud = rng.uniform(-0.1, 0.1, N)
    vd = rng.uniform(-0.1, 0.1, N)
    params = dict(
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=rng.uniform(-0.5, 0.5, N),
        V_b=rng.uniform(-0.5, 0.5, N),
        U_d=rng.uniform(-0.5, 0.5, N),
        V_d=rng.uniform(-0.5, 0.5, N),
    )

    # Vectorized call
    M_vec = compute_M(u, v, xd, yd, ud, vd, **params)
    F_vec = compute_F(u, v, xd, yd, ud, vd, **params)

    # Scalar calls
    for i in range(N):
        p_i = {k: (val[i] if isinstance(val, np.ndarray) else val)
               for k, val in params.items()}
        M_i = compute_M(u[i], v[i], xd[i], yd[i], ud[i], vd[i], **p_i)
        F_i = compute_F(u[i], v[i], xd[i], yd[i], ud[i], vd[i], **p_i)

        for j, (m_vec_j, m_i_j) in enumerate(zip(M_vec, M_i)):
            np.testing.assert_allclose(
                np.asarray(m_vec_j)[i] if np.ndim(m_vec_j) > 0 else m_vec_j,
                m_i_j,
                atol=1e-14,
                err_msg=f"M element {j} mismatch at particle {i}",
            )
        for j, (f_vec_j, f_i_j) in enumerate(zip(F_vec, F_i)):
            np.testing.assert_allclose(
                np.asarray(f_vec_j)[i],
                f_i_j,
                atol=1e-14,
                err_msg=f"F element {j} mismatch at particle {i}",
            )


def test_generated_eom_freshness():
    """The generated file must be up to date with the sympy derivation."""
    from drogued_drifters._generated_eom import SYMPY_HASH
    from drogued_drifters.cli import _symbolic_hash
    from drogued_drifters.lagrange_model import _derive_symbolic

    M_sub, F_sub, _ = _derive_symbolic()
    expected_hash = _symbolic_hash(M_sub, F_sub)
    assert SYMPY_HASH == expected_hash, (
        f"Generated EOM file is stale (hash {SYMPY_HASH} != {expected_hash}). "
        "Re-run: pixi run generate-eom"
    )
