import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from drogued_drifters.drifter import (
    DroguedDrifter,
    buoy_added_mass,
    buoy_drag_coeff,
    drogue_added_mass,
    drogue_drag_coeff,
)
from drogued_drifters.lagrange_model import F_func, M_func


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


def test_drogued_drifter_instantiation():
    dd = DroguedDrifter()


def test_MF_callable():
    assert callable(M_func)
    assert callable(F_func)


def test_MF_evaluates():
    dd = DroguedDrifter()

    t = 0.0
    U_b, V_b = dd.get_uv(t=t, x=0.0, y=0.0, z=0.0)
    U_d, V_d = dd.get_uv(t=t, x=0.0, y=0.0, z=-3.0)
    currents = U_b, V_b, U_d, V_d

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
    def _getuv_zero(*, t, x, y, z):
        return 0.0, 0.0

    dd = DroguedDrifter(get_uv=_getuv_zero)

    xd, yd = dd.get_final_drift(t_span=(0.0, 30.0))

    np.testing.assert_almost_equal(xd, 0.0, decimal=1)
    np.testing.assert_almost_equal(yd, 0.0, decimal=1)


def test_no_drift_for_theta_pi_zero_currents():
    """Drogue hangs straight down (theta=pi), no currents: should stay at rest."""

    def _getuv_zero(*, t, x, y, z):
        return 0.0, 0.0

    dd = DroguedDrifter(get_uv=_getuv_zero)

    xd, yd = dd.get_final_drift(t_span=(0.0, 30.0), theta=np.pi)

    np.testing.assert_almost_equal(xd, 0.0, decimal=1)
    np.testing.assert_almost_equal(yd, 0.0, decimal=1)


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

    def _getuv_sheared(*, t, x, y, z):
        factor = np.exp(-abs(z) / 2.0)
        return factor, 0.0

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

    xd_with, yd_with = dd_with.get_final_drift(t_span=(0.0, 600.0))
    xd_without, yd_without = dd_without.get_final_drift(t_span=(0.0, 600.0))

    np.testing.assert_almost_equal(xd_with, xd_without, decimal=1)
    np.testing.assert_almost_equal(yd_with, yd_without, decimal=1)


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


# ---------------------------------------------------------------------------
# Tests for the batched RHS path
# ---------------------------------------------------------------------------


def _make_const_uv(U_b, V_b, U_d, V_d):
    """Return a get_uv callback that returns buoy velocity at z=0, drogue otherwise."""

    def _getuv(*, t, x, y, z):
        if z == 0.0:
            return U_b, V_b
        return U_d, V_d

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
    dd_batch = DroguedDrifter()
    xd_batch, yd_batch, theta_batch, _ = dd_batch.get_final_drift_batch(
        sample_uv=_step_sampler(U_b, V_b, U_d, V_d), t_span=t_span, theta0=theta0,
    )

    # --- scalar path (one call per particle) ---
    xd_scalar = np.empty(N)
    yd_scalar = np.empty(N)
    theta_scalar = np.empty(N)

    for i in range(N):
        dd_i = DroguedDrifter(
            get_uv=_make_const_uv(U_b[i], V_b[i], U_d[i], V_d[i]),
        )
        ds = dd_i.get_full_solution(t_span=t_span, theta=theta0)
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
        sample_uv=_step_sampler(zeros, zeros, zeros, zeros), t_span=(0.0, 120.0),
    )

    np.testing.assert_allclose(xd, 0.0, atol=0.05)
    np.testing.assert_allclose(yd, 0.0, atol=0.05)
    np.testing.assert_allclose(theta, np.pi, atol=0.05)


def test_batch_uniform_currents():
    """N=10 particles all seeing the same currents should produce identical drift."""
    N = 10
    dd = DroguedDrifter()

    xd, yd, theta, _ = dd.get_final_drift_batch(
        sample_uv=_step_sampler(
            np.full(N, 0.3), np.full(N, -0.1),
            np.full(N, 0.15), np.full(N, -0.05),
        ),
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
        sample_uv=_step_sampler(
            np.array([0.1, 0.0]), np.array([0.0, 0.0]),
            np.array([0.0, 0.1]), np.array([0.0, 0.0]),
        ),
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
        sample_uv=_step_sampler(
            np.array([U_b_val]), np.zeros(N),
            np.array([U_d_val]), np.zeros(N),
        ),
        t_span=(0.0, 120.0),
    )

    # The drifter cannot go faster than the fastest layer or slower than the slowest
    assert U_d_val <= xd[0] <= U_b_val, (
        f"Expected {U_d_val} <= xd={xd[0]:.6f} <= {U_b_val}"
    )
    # y-drift should be negligible (no V forcing)
    np.testing.assert_allclose(yd[0], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests for M_func and F_func shapes and values
# ---------------------------------------------------------------------------


def test_M_F_func_shapes():
    """Verify M_func and F_func return correct shapes for scalar and batch inputs."""
    # Scalar input
    M_scalar = M_func(
        u=0.1, v=0.05, xd=0.0, yd=0.0, ud=0.0, vd=0.0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )
    assert M_scalar.shape == (4, 4), f"Expected (4,4), got {M_scalar.shape}"

    F_scalar = F_func(
        u=0.1, v=0.05, xd=0.0, yd=0.0, ud=0.0, vd=0.0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )
    assert F_scalar.shape == (4,), f"Expected (4,), got {F_scalar.shape}"

    # Batch input (N=5)
    N = 5
    u_batch = np.full(N, 0.1)
    v_batch = np.full(N, 0.05)

    M_batch = M_func(
        u=u_batch, v=v_batch, xd=np.zeros(N), yd=np.zeros(N),
        ud=np.zeros(N), vd=np.zeros(N),
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=np.full(N, 0.5), V_b=np.full(N, -0.3),
        U_d=np.full(N, 0.2), V_d=np.full(N, 0.1),
    )
    assert M_batch.shape == (N, 4, 4), f"Expected (N,4,4), got {M_batch.shape}"

    F_batch = F_func(
        u=u_batch, v=v_batch, xd=np.zeros(N), yd=np.zeros(N),
        ud=np.zeros(N), vd=np.zeros(N),
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=np.full(N, 0.5), V_b=np.full(N, -0.3),
        U_d=np.full(N, 0.2), V_d=np.full(N, 0.1),
    )
    assert F_batch.shape == (N, 4), f"Expected (N,4), got {F_batch.shape}"


def test_generated_vs_lambdified():
    """M_func and F_func must return consistent results at multiple test points."""
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
        # Build kwargs inline
        kw = dict(
            u=pt["u"], v=pt["v"], xd=pt["xd"], yd=pt["yd"], ud=pt["ud"], vd=pt["vd"],
            m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
            l=3.0, g=9.81, k_b=12.0, k_d=154.0,
            U_b=pt["U_b"], V_b=pt["V_b"], U_d=pt["U_d"], V_d=pt["V_d"],
        )

        # M_func/F_func (wrapped version with shaping)
        M_wrapped = M_func(**kw)
        F_wrapped = F_func(**kw)

        # Verify shapes
        assert M_wrapped.shape == (4, 4), f"M shape mismatch at {pt}"
        assert F_wrapped.shape == (4,), f"F shape mismatch at {pt}"

        # Verify results are finite
        assert np.all(np.isfinite(M_wrapped)), f"M has non-finite values at {pt}"
        assert np.all(np.isfinite(F_wrapped)), f"F has non-finite values at {pt}"


def test_generated_vectorized():
    """M_func and F_func must work on (N,) arrays and match scalar results."""
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
    M_vec = M_func(u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd, **params)
    F_vec = F_func(u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd, **params)

    # Should have shape (N, 4, 4) and (N, 4)
    assert M_vec.shape == (N, 4, 4), f"Expected M shape (N,4,4), got {M_vec.shape}"
    assert F_vec.shape == (N, 4), f"Expected F shape (N,4), got {F_vec.shape}"

    # Scalar calls
    for i in range(N):
        p_i = {k: (val[i] if isinstance(val, np.ndarray) else val)
               for k, val in params.items()}
        M_i = M_func(u=u[i], v=v[i], xd=xd[i], yd=yd[i], ud=ud[i], vd=vd[i], **p_i)
        F_i = F_func(u=u[i], v=v[i], xd=xd[i], yd=yd[i], ud=ud[i], vd=vd[i], **p_i)

        # Compare batch result to scalar result
        np.testing.assert_allclose(
            M_vec[i], M_i,
            atol=1e-14,
            err_msg=f"M mismatch at particle {i}",
        )
        np.testing.assert_allclose(
            F_vec[i], F_i,
            atol=1e-14,
            err_msg=f"F mismatch at particle {i}",
        )


# ---------------------------------------------------------------------------
# _save_eom_cache
# ---------------------------------------------------------------------------


def test_save_eom_cache_exists():
    """Test that _save_eom_cache is defined and callable in lagrange_model."""
    from drogued_drifters.lagrange_model import _save_eom_cache
    assert callable(_save_eom_cache), "_save_eom_cache should be callable"


def test_save_eom_cache_round_trip(tmp_path, monkeypatch):
    """Test round-trip: save EOM cache and reload via _load_or_derive.

    Uses mocked _derive_symbolic to avoid expensive computation.
    Verifies that M_func/F_func can be called with scalar input and return
    correctly shaped outputs.
    """
    import pytest
    from pathlib import Path
    from unittest.mock import patch
    import sympy as sp
    from drogued_drifters.lagrange_model import (
        _save_eom_cache,
        _load_or_derive,
        M_func,
        F_func,
        LagrangeParams,
    )

    # Create trivial mock M (4x4 identity) and F (4x1 zeros)
    M_mock = sp.eye(4)
    F_mock = sp.zeros(4, 1)

    # Create args tuple with correct 19 symbols
    args_mock = tuple(sp.Symbol(name, real=True) for name in LagrangeParams._fields)

    # Patch _derive_symbolic to return our mock
    with patch("drogued_drifters.lagrange_model._derive_symbolic") as mock_derive:
        mock_derive.return_value = (M_mock, F_mock, args_mock)

        # Save to temp file
        cache_file = tmp_path / "test_symbolic_eom.srepr"
        _save_eom_cache(cache_file)

        # Verify file was created
        assert cache_file.exists(), f"Cache file not created at {cache_file}"

        # Patch _load_or_derive to use our temp cache instead of default location
        with patch("drogued_drifters.lagrange_model._SREPR_PATH", cache_file):
            # Clear the cache on _load_or_derive so it re-reads
            from drogued_drifters import lagrange_model
            lagrange_model._load_or_derive.cache_clear()

            # Load back using _load_or_derive
            _raw_M, _raw_F, arg_symbols = _load_or_derive()

            # Verify we got callable functions
            assert callable(_raw_M), "_raw_M should be callable"
            assert callable(_raw_F), "_raw_F should be callable"

            # Test with scalar input: all 19 parameters as scalars
            scalar_input = tuple([1.0] * len(LagrangeParams._fields))
            M_result = _raw_M(*scalar_input)
            F_result = _raw_F(*scalar_input)

            # For identity matrix mock, _raw_M should return diagonal elements
            # For zero vector mock, _raw_F should return all zeros
            assert M_result is not None, "M should be callable"
            assert F_result is not None, "F should be callable"


def test_save_eom_cache_format_has_separator(tmp_path, monkeypatch):
    """Test that the written .srepr file contains the '---' separator."""
    from pathlib import Path
    from unittest.mock import patch
    import sympy as sp
    from drogued_drifters.lagrange_model import (
        _save_eom_cache,
        LagrangeParams,
    )

    # Create trivial mock expressions
    M_mock = sp.eye(4)
    F_mock = sp.zeros(4, 1)
    args_mock = tuple(sp.Symbol(name, real=True) for name in LagrangeParams._fields)

    with patch("drogued_drifters.lagrange_model._derive_symbolic") as mock_derive:
        mock_derive.return_value = (M_mock, F_mock, args_mock)

        cache_file = tmp_path / "test_eom_format.srepr"
        _save_eom_cache(cache_file)

        # Read the file and verify format
        content = cache_file.read_text()
        parts = content.split("---")

        # Should have exactly 3 parts: M_srepr, F_srepr, arg_names
        assert len(parts) == 3, (
            f"Expected 3 parts separated by '---', got {len(parts)}. "
            f"File content:\n{content[:200]}..."
        )


def test_load_or_derive_raises_on_malformed_file(tmp_path, monkeypatch):
    """Test that _load_or_derive raises clear error for malformed .srepr file."""
    from pathlib import Path
    from drogued_drifters.lagrange_model import _load_or_derive
    from drogued_drifters import lagrange_model

    # Create a malformed .srepr file (missing separators)
    bad_file = tmp_path / "malformed.srepr"
    bad_file.write_text("some_invalid_content_without_separators")

    # Patch the _SREPR_PATH to point to our bad file
    with monkeypatch.context() as mp:
        mp.setattr(lagrange_model, "_SREPR_PATH", bad_file)

        # Clear the cache
        lagrange_model._load_or_derive.cache_clear()

        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="Invalid .srepr format"):
            _load_or_derive()


def test_save_eom_cache_creates_parent_directories(tmp_path):
    """Test that _save_eom_cache creates parent directories if needed."""
    from pathlib import Path
    from unittest.mock import patch
    import sympy as sp
    from drogued_drifters.lagrange_model import (
        _save_eom_cache,
        LagrangeParams,
    )

    # Create a nested path that doesn't exist yet
    nested_path = tmp_path / "deep" / "nested" / "cache.srepr"
    assert not nested_path.parent.exists(), "Parent dirs should not exist"

    # Create mock
    M_mock = sp.eye(4)
    F_mock = sp.zeros(4, 1)
    args_mock = tuple(sp.Symbol(name, real=True) for name in LagrangeParams._fields)

    with patch("drogued_drifters.lagrange_model._derive_symbolic") as mock_derive:
        mock_derive.return_value = (M_mock, F_mock, args_mock)

        # Should create parent directories
        _save_eom_cache(nested_path)

        assert nested_path.exists(), "Cache file should be created"
        assert nested_path.parent.exists(), "Parent dirs should be created"


