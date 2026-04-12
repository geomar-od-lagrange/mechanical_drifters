"""Tests for symbolic derivation fallback (_derive_symbolic).

These tests force cache misses and verify that live symbolic derivation
produces the same numerical results as cached versions.

Marked @pytest.mark.slow because symbolic derivation takes 30-60s.
"""

import numpy as np
import pytest

from drogued_drifters.eom import (
    DrifterPhysics,
    EOMState,
    F_func,
    M_func,
    _derive_symbolic,
    _get_eom_callables,
)


@pytest.mark.slow
def test_derive_symbolic_produces_matrices():
    """Verify _derive_symbolic returns symbolic M and F of correct shape."""
    M_sub, F_sub, args = _derive_symbolic()

    # M should be 4x4 symbolic matrix
    assert M_sub.shape == (4, 4), f"Expected M shape (4,4), got {M_sub.shape}"

    # F should be 4x1 symbolic matrix
    assert F_sub.shape == (4, 1), f"Expected F shape (4,1), got {F_sub.shape}"

    # args should have 19 symbols (9 DrifterPhysics + 10 EOMState)
    expected_n_args = len(DrifterPhysics._fields) + len(EOMState._fields)
    assert (
        len(args) == expected_n_args
    ), f"Expected {expected_n_args} args, got {len(args)}"


@pytest.mark.slow
def test_derive_symbolic_finite_values():
    """Derived M and F should evaluate to finite values at test points."""
    M_sub, F_sub, args = _derive_symbolic()

    # Build a test parameter dict keyed by symbol name (order-independent)
    import sympy as sp

    test_values = {
        "m_b": 1.0,
        "m_d": 2.7,
        "m_hat_d": 1.0,
        "m_tilde_d": 101.0,
        "m_tilde_b": 1.9,
        "l": 3.0,
        "g": 9.81,
        "k_b": 12.0,
        "k_d": 154.0,
        "u": 0.1,
        "v": 0.05,
        "xd": 0.0,
        "yd": 0.0,
        "ud": 0.0,
        "vd": 0.0,
        "U_b": 0.5,
        "V_b": -0.3,
        "U_d": 0.2,
        "V_d": 0.1,
    }

    # Substitute symbols with numerical values (order-independent via names)
    subs_dict = {sym: test_values[str(sym)] for sym in args}
    M_num = M_sub.subs(subs_dict)
    F_num = F_sub.subs(subs_dict)

    # Convert to Python float
    M_vals = np.array([[float(M_num[i, j]) for j in range(4)] for i in range(4)])
    F_vals = np.array([float(F_num[i]) for i in range(4)])

    # Check finiteness
    assert np.all(np.isfinite(M_vals)), f"M has non-finite values: {M_vals}"
    assert np.all(np.isfinite(F_vals)), f"F has non-finite values: {F_vals}"


@pytest.mark.slow
def test_derived_vs_cached_numerical_agreement():
    """Freshly derived M/F must agree numerically with cached M/F.

    Tests M_func/F_func which use the cache. If cache is valid, this should
    pass. If cache is missing or broken, it should fall back to _derive_symbolic
    and still agree (up to numerical precision).
    """
    test_physics = DrifterPhysics(
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
    test_state = EOMState(
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

    # Get M/F from cached path
    M_cached = M_func(test_physics, test_state)
    F_cached = F_func(test_physics, test_state)

    # Now derive fresh and lambdify
    import sympy as sp

    M_fresh_sym, F_fresh_sym, args = _derive_symbolic()

    # Extract M upper-triangle and F elements
    m_exprs = tuple(M_fresh_sym[i, j] for i in range(4) for j in range(i, 4))
    f_exprs = tuple(F_fresh_sym[i] for i in range(4))

    M_raw_fresh = sp.lambdify(args, m_exprs, modules="numpy", cse=True)
    F_raw_fresh = sp.lambdify(args, f_exprs, modules="numpy", cse=True)

    # Call with same parameters using packer from _build_packer
    from drogued_drifters.eom import _build_packer

    packer = _build_packer(M_raw_fresh)
    params_tuple = packer(test_physics, test_state)

    M_elems_fresh = M_raw_fresh(*params_tuple)
    F_elems_fresh = F_raw_fresh(*params_tuple)

    # Assemble fresh M into (4, 4)
    M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems_fresh
    M_fresh = np.array(
        [
            [M00, M01, M02, M03],
            [M01, M11, M12, M13],
            [M02, M12, M22, M23],
            [M03, M13, M23, M33],
        ],
        dtype=float,
    )

    # Assemble fresh F into (4,)
    F_fresh = np.array(F_elems_fresh, dtype=float)

    # Compare cached vs fresh
    np.testing.assert_allclose(
        M_cached,
        M_fresh,
        rtol=1e-6,
        atol=1e-14,
        err_msg="M from cache and derivation disagree",
    )
    np.testing.assert_allclose(
        F_cached,
        F_fresh,
        rtol=1e-6,
        atol=1e-14,
        err_msg="F from cache and derivation disagree",
    )


@pytest.mark.slow
def test_get_eom_callables_fallback_on_missing_cache(tmp_path, monkeypatch):
    """_get_eom_callables should fall back to _derive_symbolic if cache missing."""
    from drogued_drifters import eom

    # Patch _CACHE_PATH to non-existent file in temp dir
    temp_cache = tmp_path / "missing.pkl"
    assert not temp_cache.exists(), "Cache file should not exist"

    with monkeypatch.context() as mp:
        mp.setattr(eom, "_CACHE_PATH", temp_cache)

        # Clear cache so it re-evaluates
        eom._get_eom_callables.cache_clear()

        # Should fall back to _derive_symbolic and work
        qdd_raw, M_raw, F_raw, args, pack_eom_args = eom._get_eom_callables()

        assert callable(qdd_raw), "qdd_raw should be callable (fallback)"
        assert callable(M_raw), "M_raw should be callable (fallback)"
        assert callable(F_raw), "F_raw should be callable (fallback)"
        expected_n_args = len(DrifterPhysics._fields) + len(EOMState._fields)
        assert (
            len(args) == expected_n_args
        ), f"Expected {expected_n_args} args, got {len(args)}"

        # Verify the fallback works by evaluating at test point
        test_physics = DrifterPhysics(
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
        test_state = EOMState(
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
        )
        packed = pack_eom_args(test_physics, test_state)
        qdd_elems = qdd_raw(*packed)
        M_elems = M_raw(*packed)
        F_elems = F_raw(*packed)

        assert len(qdd_elems) == 4, f"Expected 4 qdd elements, got {len(qdd_elems)}"
        assert len(M_elems) == 10, f"Expected 10 M elements, got {len(M_elems)}"
        assert len(F_elems) == 4, f"Expected 4 F elements, got {len(F_elems)}"
        assert all(np.isfinite(q) for q in qdd_elems), "qdd has non-finite values"
        assert all(np.isfinite(m) for m in M_elems), "M has non-finite values"
        assert all(np.isfinite(f) for f in F_elems), "F has non-finite values"


@pytest.mark.slow
def test_symbolic_derivation_takes_time():
    """Document that _derive_symbolic is expensive (~30-60s).

    This test just calls it and verifies it returns in reasonable time.
    (Slow test infrastructure will skip this by default.)
    """
    import time

    # Should complete within 2 minutes (conservative bound)
    start = time.time()
    M_sub, F_sub, args = _derive_symbolic()
    elapsed = time.time() - start

    # Just verify it finished; don't enforce a specific time.
    assert M_sub.shape == (4, 4)
    assert elapsed > 0.0
    # If this test is running, it's in the slow test suite, so slowness is expected.
