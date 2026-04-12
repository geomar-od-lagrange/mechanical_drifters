import numpy as np
import pytest

from drogued_drifters.models.drogued_drifter import (
    DroguedDrifter,
    buoy_horizontal_added_mass,
    buoy_horizontal_drag_coeff,
    drogue_horizontal_added_mass,
    drogue_horizontal_drag_coeff,
)
from drogued_drifters.models.drogued_drifter import (
    DrifterPhysics,
    EOMState,
)
from drogued_drifters.eom import (
    eval_M,
    eval_F,
    _make_qdd_func,
)


def _step_sampler(U_b, V_b, U_d, V_d):
    """Return a sample_uv callable: buoy velocity at z=0, drogue velocity otherwise."""
    U_b = np.asarray(U_b, dtype=float)
    V_b = np.asarray(V_b, dtype=float)
    U_d = np.asarray(U_d, dtype=float)
    V_d = np.asarray(V_d, dtype=float)

    def sample_uv(z):
        z_arr = np.asarray(z)
        if np.all(z_arr == 0):
            return U_b, V_b
        return U_d, V_d

    return sample_uv


def _sample_uv_zero(z):
    z_arr = np.asarray(z, dtype=float)
    scalar = z_arr.ndim == 0
    z_arr = np.atleast_1d(z_arr)
    U = np.zeros_like(z_arr)
    V = np.zeros_like(z_arr)
    if scalar:
        return float(U[0]), float(V[0])
    return U, V


def _default_sample_uv(z):
    """Placeholder velocity sampler for testing."""
    z_arr = np.asarray(z, dtype=float)
    scalar = z_arr.ndim == 0
    z_arr = np.atleast_1d(z_arr)
    U = np.where(z_arr == 0.0, 1.0, -1.0)
    V = np.where(z_arr == 0.0, 1.0, -1.0)
    if scalar:
        return float(U[0]), float(V[0])
    return U, V


def test_drogued_drifter_instantiation():
    dd = DroguedDrifter()
    assert dd.physics.l == 3.0


def test_MF_callable():
    dd = DroguedDrifter()
    assert callable(lambda p, s: eval_M(dd, p, s))
    assert callable(lambda p, s: eval_F(dd, p, s))


def test_qdd_func_evaluates():
    dd = DroguedDrifter()
    _qdd_func = _make_qdd_func(dd, "numpy")

    U_b, V_b = _default_sample_uv(0.0)
    U_d, V_d = _default_sample_uv(-3.0)

    state = EOMState(
        u_stereo=0.1,
        v_stereo=0.05,
        xd=0.0,
        yd=0.0,
        ud_stereo=0.0,
        vd_stereo=0.0,
        U_b=U_b,
        V_b=V_b,
        U_d=U_d,
        V_d=V_d,
    )
    qdd = _qdd_func(dd.physics, state)

    assert qdd.shape == (4,), f"Expected (4,), got {qdd.shape}"
    assert np.all(np.isfinite(qdd)), "qdd has non-finite values"


def test_no_drift_for_zero_currents():
    dd = DroguedDrifter()

    xd, yd, _ = dd.get_final_drift(_sample_uv_zero, t_span=(0.0, 30.0))

    np.testing.assert_almost_equal(xd, 0.0, decimal=1)
    np.testing.assert_almost_equal(yd, 0.0, decimal=1)


def test_no_drift_for_theta_pi_zero_currents():
    """Drogue hangs straight down (theta=pi), no currents: should stay at rest."""
    dd = DroguedDrifter()

    xd, yd, _ = dd.get_final_drift(_sample_uv_zero, t_span=(0.0, 30.0), theta=np.pi)

    np.testing.assert_almost_equal(xd, 0.0, decimal=1)
    np.testing.assert_almost_equal(yd, 0.0, decimal=1)


def test_parameterization_matches_table1():
    """Check that parameterization functions reproduce Callies et al. values."""
    rho = 1025.0
    # Drogue: cross of two plates, w_d=0.5m, h_d=0.5m
    m_tilde_d = drogue_horizontal_added_mass(rho=rho, w_d=0.5, h_d=0.5)
    np.testing.assert_almost_equal(m_tilde_d, 101.0, decimal=0)

    k_d = drogue_horizontal_drag_coeff(rho=rho, w_d=0.5, h_d=0.5)
    np.testing.assert_almost_equal(k_d, 154.0, decimal=-1)

    # Buoy: cylinder, d_b=0.1m, h_b=0.24m
    m_tilde_b = buoy_horizontal_added_mass(rho=rho, d_b=0.1, h_b=0.24)
    np.testing.assert_almost_equal(m_tilde_b, 1.9, decimal=1)

    k_b = buoy_horizontal_drag_coeff(rho=rho, d_b=0.1, h_b=0.24)
    np.testing.assert_almost_equal(k_b, 12.0, decimal=0)


def test_steady_state_independent_of_added_mass():
    """Added mass only affects acceleration, not steady-state drift."""

    def _sample_uv_sheared(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        factor = np.exp(-np.abs(z_arr) / 2.0)
        U = factor
        V = np.zeros_like(z_arr)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    dd_with = DroguedDrifter(m_tilde_d=101.0, m_tilde_b=1.9)
    dd_without = DroguedDrifter(m_tilde_d=0.0, m_tilde_b=0.0)

    xd_with, yd_with, _ = dd_with.get_final_drift(
        _sample_uv_sheared, t_span=(0.0, 600.0)
    )
    xd_without, yd_without, _ = dd_without.get_final_drift(
        _sample_uv_sheared, t_span=(0.0, 600.0)
    )

    np.testing.assert_almost_equal(xd_with, xd_without, decimal=1)
    np.testing.assert_almost_equal(yd_with, yd_without, decimal=1)


def test_get_full_solution_returns_xarray():
    """get_full_solution returns an xarray Dataset with named variables."""
    dd = DroguedDrifter()
    ds = dd.get_full_solution(_default_sample_uv, t_span=(0, 10), t_eval=[0, 5, 10])

    assert "time" in ds.coords
    for var in ["x", "y", "theta", "phi", "xd", "yd", "thetad", "phid"]:
        assert var in ds, f"missing variable {var}"
    assert len(ds.time) == 3

    # arithmetic preserves xarray type (needed for .plot())
    import xarray as xr

    theta_deg = ds.theta * 180 / np.pi
    assert isinstance(theta_deg, xr.DataArray)
    assert "time" in theta_deg.coords


# ---------------------------------------------------------------------------
# Tests for the batched RHS path
# ---------------------------------------------------------------------------


def _make_const_uv(U_b, V_b, U_d, V_d):
    """Return a sample_uv callback: buoy velocity at z=0, drogue velocity otherwise."""

    def _sample_uv(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, U_b, U_d)
        V = np.where(z_arr == 0.0, V_b, V_d)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    return _sample_uv


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

    # Build y0 in public format: (x, y, theta, phi, xd, yd, thetad, phid)
    y0_batch = np.zeros((N, 8))
    y0_batch[:, 2] = theta0  # theta column
    # phi, xd, yd, thetad, phid all zero

    # --- batch path ---
    dd_batch = DroguedDrifter()
    xd_batch, yd_batch, Y_batch, _ = dd_batch.get_final_drift_batch(
        _step_sampler(U_b, V_b, U_d, V_d),
        t_span=t_span,
        y0=y0_batch,
    )
    theta_batch = Y_batch[:, 2]

    # --- scalar path (one call per particle) ---
    xd_scalar = np.empty(N)
    yd_scalar = np.empty(N)
    theta_scalar = np.empty(N)

    for i in range(N):
        dd_i = DroguedDrifter()
        ds = dd_i.get_full_solution(
            _make_const_uv(U_b[i], V_b[i], U_d[i], V_d[i]),
            t_span=t_span, theta=theta0,
        )
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

    xd, yd, Y_final, _ = dd.get_final_drift_batch(
        _step_sampler(zeros, zeros, zeros, zeros),
        t_span=(0.0, 120.0),
    )

    np.testing.assert_allclose(xd, 0.0, atol=0.05)
    np.testing.assert_allclose(yd, 0.0, atol=0.05)
    np.testing.assert_allclose(Y_final[:, 2], np.pi, atol=0.05)


def test_batch_uniform_currents():
    """N=10 particles all seeing the same currents should produce identical drift."""
    N = 10
    dd = DroguedDrifter()

    xd, yd, Y_final, _ = dd.get_final_drift_batch(
        _step_sampler(
            np.full(N, 0.3),
            np.full(N, -0.1),
            np.full(N, 0.15),
            np.full(N, -0.05),
        ),
        t_span=(0.0, 120.0),
    )

    # All particles must agree with each other
    np.testing.assert_allclose(xd, xd[0], atol=1e-10)
    np.testing.assert_allclose(yd, yd[0], atol=1e-10)
    np.testing.assert_allclose(Y_final[:, 2], Y_final[0, 2], atol=1e-10)


def test_batch_opposite_shear():
    """Two particles with swapped buoy/drogue forcing should give different drifts."""
    dd = DroguedDrifter()

    xd, yd, Y_final, _ = dd.get_final_drift_batch(
        _step_sampler(
            np.array([0.1, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.1]),
            np.array([0.0, 0.0]),
        ),
        t_span=(0.0, 120.0),
    )

    # The drifter model is not symmetric in buoy vs drogue forcing,
    # so the two particles must produce different drift velocities.
    assert not np.allclose(
        xd[0], xd[1], atol=1e-4
    ), f"Expected different xd for swapped buoy/drogue forcing, got {xd}"


def test_batch_drift_between_buoy_and_drogue():
    """Drift velocity should lie between the buoy and drogue current speeds."""
    N = 1
    dd = DroguedDrifter()

    U_b_val, U_d_val = 0.2, 0.1

    xd, yd, Y_final, _ = dd.get_final_drift_batch(
        _step_sampler(
            np.array([U_b_val]),
            np.zeros(N),
            np.array([U_d_val]),
            np.zeros(N),
        ),
        t_span=(0.0, 120.0),
    )

    # The drifter cannot go faster than the fastest layer or slower than the slowest
    assert (
        U_d_val <= xd[0] <= U_b_val
    ), f"Expected {U_d_val} <= xd={xd[0]:.6f} <= {U_b_val}"
    # y-drift should be negligible (no V forcing)
    np.testing.assert_allclose(yd[0], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests for eval_M and eval_F shapes and values
# ---------------------------------------------------------------------------


from conftest import DEFAULT_PHYSICS as _DEFAULT_PHYSICS


def test_M_F_func_shapes():
    """Verify eval_M and eval_F return correct shapes for scalar and batch inputs."""
    dd = DroguedDrifter()

    # Scalar input
    M_scalar = eval_M(
        dd,
        _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=0.1, v_stereo=0.05,
            xd=0.0, yd=0.0,
            ud_stereo=0.0, vd_stereo=0.0,
            U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
        ),
    )
    assert M_scalar.shape == (4, 4), f"Expected (4,4), got {M_scalar.shape}"

    F_scalar = eval_F(
        dd,
        _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=0.1, v_stereo=0.05,
            xd=0.0, yd=0.0,
            ud_stereo=0.0, vd_stereo=0.0,
            U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
        ),
    )
    assert F_scalar.shape == (4,), f"Expected (4,), got {F_scalar.shape}"

    # Batch input (N=5)
    N = 5
    u_batch = np.full(N, 0.1)
    v_batch = np.full(N, 0.05)

    M_batch = eval_M(
        dd,
        _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=u_batch, v_stereo=v_batch,
            xd=np.zeros(N), yd=np.zeros(N),
            ud_stereo=np.zeros(N), vd_stereo=np.zeros(N),
            U_b=np.full(N, 0.5), V_b=np.full(N, -0.3),
            U_d=np.full(N, 0.2), V_d=np.full(N, 0.1),
        ),
    )
    assert M_batch.shape == (N, 4, 4), f"Expected (N,4,4), got {M_batch.shape}"

    F_batch = eval_F(
        dd,
        _DEFAULT_PHYSICS,
        EOMState(
            u_stereo=u_batch, v_stereo=v_batch,
            xd=np.zeros(N), yd=np.zeros(N),
            ud_stereo=np.zeros(N), vd_stereo=np.zeros(N),
            U_b=np.full(N, 0.5), V_b=np.full(N, -0.3),
            U_d=np.full(N, 0.2), V_d=np.full(N, 0.1),
        ),
    )
    assert F_batch.shape == (N, 4), f"Expected (N,4), got {F_batch.shape}"


def test_generated_vs_lambdified():
    """eval_M and eval_F must return consistent results at multiple test points."""
    dd = DroguedDrifter()
    test_points = [
        dict(u_stereo=0, v_stereo=0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(u_stereo=0.1, v_stereo=0.05, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(
            u_stereo=0.3, v_stereo=-0.2,
            xd=0.1, yd=-0.05,
            ud_stereo=0.01, vd_stereo=-0.02,
            U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
        ),
        dict(u_stereo=2.0, v_stereo=0.0, xd=0, yd=0, ud_stereo=0, vd_stereo=0, U_b=0, V_b=0, U_d=0, V_d=0),
        dict(
            u_stereo=0.5, v_stereo=0.5,
            xd=0.1, yd=0.1,
            ud_stereo=0.05, vd_stereo=0.05,
            U_b=1.0, V_b=1.0, U_d=-1.0, V_d=-1.0,
        ),
    ]

    for pt in test_points:
        state = EOMState(**pt)

        M_wrapped = eval_M(dd, _DEFAULT_PHYSICS, state)
        F_wrapped = eval_F(dd, _DEFAULT_PHYSICS, state)

        assert M_wrapped.shape == (4, 4), f"M shape mismatch at {pt}"
        assert F_wrapped.shape == (4,), f"F shape mismatch at {pt}"

        assert np.all(np.isfinite(M_wrapped)), f"M has non-finite values at {pt}"
        assert np.all(np.isfinite(F_wrapped)), f"F has non-finite values at {pt}"


def test_generated_vectorized():
    """eval_M and eval_F must work on (N,) arrays and match scalar results."""
    dd = DroguedDrifter()
    N = 5
    rng = np.random.default_rng(123)
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
    M_vec = eval_M(dd, _DEFAULT_PHYSICS, batch_state)
    F_vec = eval_F(dd, _DEFAULT_PHYSICS, batch_state)

    assert M_vec.shape == (N, 4, 4), f"Expected M shape (N,4,4), got {M_vec.shape}"
    assert F_vec.shape == (N, 4), f"Expected F shape (N,4), got {F_vec.shape}"

    for i in range(N):
        scalar_state = EOMState(
            u_stereo=u[i], v_stereo=v[i],
            xd=xd[i], yd=yd[i],
            ud_stereo=ud[i], vd_stereo=vd[i],
            U_b=U_b[i], V_b=V_b[i], U_d=U_d[i], V_d=V_d[i],
        )
        M_i = eval_M(dd, _DEFAULT_PHYSICS, scalar_state)
        F_i = eval_F(dd, _DEFAULT_PHYSICS, scalar_state)

        np.testing.assert_allclose(M_vec[i], M_i, atol=1e-14, err_msg=f"M mismatch at particle {i}")
        np.testing.assert_allclose(F_vec[i], F_i, atol=1e-14, err_msg=f"F mismatch at particle {i}")


# ---------------------------------------------------------------------------
# Pickle cache tests
# ---------------------------------------------------------------------------


def test_cache_file_exists():
    """Test that the pickle cache file exists."""
    dd = DroguedDrifter()
    assert dd._cache_path.exists(), f"Cache file not found at {dd._cache_path}"


def test_cache_loads_successfully():
    """Test that the pickle cache loads and has the expected keys."""
    import pickle

    from drogued_drifters.eom import _cache_key

    dd = DroguedDrifter()
    cache_path = dd._cache_path
    cached = pickle.loads(cache_path.read_bytes())
    assert "key" in cached
    assert "M" in cached
    assert "F" in cached
    assert "qdd" in cached
    assert "args" in cached
    assert cached["key"] == _cache_key(dd._derive_symbolic), "Cache key mismatch"


def test_cache_invalidation_on_stale_key(tmp_path, monkeypatch):
    """_load_or_derive should re-derive when pickle has wrong key."""
    import pickle

    from drogued_drifters import eom

    dd = DroguedDrifter()

    # Write a pickle with a wrong key
    stale_cache = tmp_path / "stale.pkl"
    stale_cache.write_bytes(
        pickle.dumps(
            {"key": "wrong_key", "M": None, "F": None, "qdd": None, "args": None}
        )
    )

    # Clear caches so they re-evaluate
    eom._CALLABLE_CACHE.pop("DroguedDrifter", None)
    eom._QDD_CACHE.pop(("DroguedDrifter", "numpy"), None)

    # Monkey-patch the cache path on the instance
    original_cache_path = type(dd)._cache_path
    monkeypatch.setattr(type(dd), "_cache_path", property(lambda self: stale_cache))

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qdd_raw, M_raw, F_raw, pack = eom._get_eom_callables(dd)

        assert callable(qdd_raw)
        assert callable(M_raw)
        assert callable(F_raw)
    finally:
        # Restore caches
        eom._CALLABLE_CACHE.pop("DroguedDrifter", None)
        eom._QDD_CACHE.pop(("DroguedDrifter", "numpy"), None)
        monkeypatch.setattr(type(dd), "_cache_path", original_cache_path)


# ---------------------------------------------------------------------------
# Batch 2: horizontal rename tests
# ---------------------------------------------------------------------------

_NEW_NAMES = [
    "drogue_horizontal_added_mass",
    "buoy_horizontal_added_mass",
    "drogue_horizontal_drag_coeff",
    "buoy_horizontal_drag_coeff",
]

_OLD_NAMES = [
    "drogue_added_mass",
    "buoy_added_mass",
    "drogue_drag_coeff",
    "buoy_drag_coeff",
]


class TestHorizontalRename:
    """Tests that the four helper functions have been renamed to include '_horizontal'."""

    def test_new_names_importable(self):
        """The new *_horizontal_* names must be importable from model module."""
        import importlib

        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        for name in _NEW_NAMES:
            assert hasattr(mod, name), f"{name} not found in drogued_drifters.models.drogued_drifter"

    def test_new_names_callable(self):
        """Each renamed function must be callable."""
        import importlib

        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        for name in _NEW_NAMES:
            fn = getattr(mod, name, None)
            assert fn is not None, f"{name} not found"
            assert callable(fn), f"{name} is not callable"

    def test_old_names_removed(self):
        """The old names must NOT exist in the module after rename."""
        import importlib

        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        for name in _OLD_NAMES:
            assert not hasattr(
                mod, name
            ), f"Old name {name} still exists — should have been renamed"

    def test_drogue_horizontal_added_mass_value(self):
        import importlib
        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        fn = getattr(mod, "drogue_horizontal_added_mass")
        result = fn(rho=1025.0, w_d=0.5, h_d=0.5)
        np.testing.assert_almost_equal(result, 101.0, decimal=0)

    def test_buoy_horizontal_added_mass_value(self):
        import importlib
        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        fn = getattr(mod, "buoy_horizontal_added_mass")
        result = fn(rho=1025.0, d_b=0.1, h_b=0.24)
        np.testing.assert_almost_equal(result, 1.9, decimal=1)

    def test_drogue_horizontal_drag_coeff_value(self):
        import importlib
        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        fn = getattr(mod, "drogue_horizontal_drag_coeff")
        result = fn(rho=1025.0, w_d=0.5, h_d=0.5)
        np.testing.assert_almost_equal(result, 154.0, decimal=-1)

    def test_buoy_horizontal_drag_coeff_value(self):
        import importlib
        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        fn = getattr(mod, "buoy_horizontal_drag_coeff")
        result = fn(rho=1025.0, d_b=0.1, h_b=0.24)
        np.testing.assert_almost_equal(result, 12.0, decimal=0)

    def test_docstrings_mention_horizontal(self):
        import importlib
        mod = importlib.import_module("drogued_drifters.models.drogued_drifter")
        for name in _NEW_NAMES:
            fn = getattr(mod, name, None)
            assert fn is not None, f"{name} not found"
            doc = fn.__doc__
            assert doc is not None, f"{name} has no docstring"
            assert (
                "horizontal" in doc.lower()
            ), f"{name}.__doc__ does not mention 'horizontal': {doc!r}"


# ---------------------------------------------------------------------------
# DW-B: State vector index round-trip test
# ---------------------------------------------------------------------------


def test_state_vector_round_trip():
    """Construct a known state, convert public->internal->public, check each component."""
    from drogued_drifters.coords import _spherical_to_uv, _uv_to_spherical
    from drogued_drifters.models.drogued_drifter import IX, IY, IU, IV, IXD, IYD, IUD, IVD

    x0, y0 = 100.0, -50.0
    theta0, phi0 = 2.8, 0.5
    xd0, yd0 = 0.3, -0.2
    thetad0, phid0 = 0.01, -0.005

    u0, v0, ud0, vd0 = _spherical_to_uv(theta0, phi0, thetad0, phid0)
    internal = np.array([x0, y0, u0, v0, xd0, yd0, ud0, vd0])

    assert internal[IX] == x0
    assert internal[IY] == y0
    assert internal[IU] == u0
    assert internal[IV] == v0
    assert internal[IXD] == xd0
    assert internal[IYD] == yd0
    assert internal[IUD] == ud0
    assert internal[IVD] == vd0

    theta_rt, phi_rt, thetad_rt, phid_rt = _uv_to_spherical(
        internal[IU], internal[IV], internal[IUD], internal[IVD],
    )

    np.testing.assert_allclose(theta_rt, theta0, atol=1e-12)
    np.testing.assert_allclose(phi_rt, phi0, atol=1e-12)
    np.testing.assert_allclose(thetad_rt, thetad0, atol=1e-12)
    np.testing.assert_allclose(phid_rt, phid0, atol=1e-12)

    assert internal[IX] == x0
    assert internal[IY] == y0
    assert internal[IXD] == xd0
    assert internal[IYD] == yd0


# ---------------------------------------------------------------------------
# DW-F: max_accel diagnostic tests
# ---------------------------------------------------------------------------


def test_max_accel_decreases_with_longer_t_span():
    """Longer integration should yield smaller max_accel (closer to steady state)."""

    def _sample_uv_sheared(z):
        z_arr = np.asarray(z, dtype=float)
        scalar = z_arr.ndim == 0
        z_arr = np.atleast_1d(z_arr)
        U = np.where(z_arr == 0.0, 0.3, 0.05)
        V = np.zeros_like(z_arr)
        if scalar:
            return float(U[0]), float(V[0])
        return U, V

    dd = DroguedDrifter()

    _, _, max_accel_short = dd.get_final_drift(_sample_uv_sheared, t_span=(0.0, 10.0))
    _, _, max_accel_long = dd.get_final_drift(_sample_uv_sheared, t_span=(0.0, 600.0))

    assert max_accel_long < max_accel_short, (
        f"Expected max_accel to decrease with longer integration: "
        f"short={max_accel_short}, long={max_accel_long}"
    )


def test_max_accel_zero_for_zero_currents():
    """Zero currents from equilibrium: max_accel should be ~0 (no forcing)."""
    dd = DroguedDrifter()
    _, _, max_accel = dd.get_final_drift(_sample_uv_zero, t_span=(0.0, 120.0))

    np.testing.assert_allclose(
        max_accel,
        0.0,
        atol=1e-6,
        err_msg=f"Expected max_accel ~0 for zero currents, got {max_accel}",
    )
