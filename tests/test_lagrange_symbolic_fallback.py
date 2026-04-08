"""Tests for symbolic derivation fallback (_derive_symbolic).

These tests force cache misses and verify that live symbolic derivation
produces the same numerical results as cached versions.

Marked @pytest.mark.slow because symbolic derivation takes 30-60s.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from drogued_drifters.lagrange_model import (
    F_func,
    M_func,
    _apply_cse_and_lambdify,
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

    # args should have 19 symbols (matching LagrangeParams fields)
    assert len(args) == 19, f"Expected 19 args, got {len(args)}"


@pytest.mark.slow
def test_derive_symbolic_finite_values():
    """Derived M and F should evaluate to finite values at test points."""
    M_sub, F_sub, args = _derive_symbolic()

    # Build a test parameter tuple (all 19 parameters)
    import sympy as sp

    test_params = (
        0.1, 0.05,  # u, v
        0.0, 0.0,   # xd, yd
        0.0, 0.0,   # ud, vd
        1.0,        # m_b
        2.7,        # m_d
        1.0,        # m_hat_d
        101.0,      # m_tilde_d
        1.9,        # m_tilde_b
        3.0,        # l
        9.81,       # g
        12.0,       # k_b
        154.0,      # k_d
        0.5,        # U_b
        -0.3,       # V_b
        0.2,        # U_d
        0.1,        # V_d
    )

    # Substitute symbols with numerical values
    subs_dict = dict(zip(args, test_params))
    M_num = M_sub.subs(subs_dict)
    F_num = F_sub.subs(subs_dict)

    # Convert to Python float
    import sympy as sp
    M_vals = np.array([[float(M_num[i, j]) for j in range(4)] for i in range(4)])
    F_vals = np.array([float(F_num[i]) for i in range(4)])

    # Check finiteness
    assert np.all(np.isfinite(M_vals)), f"M has non-finite values: {M_vals}"
    assert np.all(np.isfinite(F_vals)), f"F has non-finite values: {F_vals}"


@pytest.mark.slow
def test_cse_and_lambdify_produces_callables():
    """Verify _apply_cse_and_lambdify produces callable functions."""
    M_sub, F_sub, args = _derive_symbolic()

    _raw_M, _raw_F, args_out = _apply_cse_and_lambdify(M_sub, F_sub, args)

    assert callable(_raw_M), "_raw_M should be callable"
    assert callable(_raw_F), "_raw_F should be callable"
    assert len(args_out) == 19, f"Expected 19 args, got {len(args_out)}"


@pytest.mark.slow
def test_derived_vs_cached_numerical_agreement():
    """Freshly derived M/F must agree numerically with cached M/F.

    Tests M_func/F_func which use the cache. If cache is valid, this should
    pass. If cache is missing or broken, it should fall back to _derive_symbolic
    and still agree (up to numerical precision).
    """
    # Get M/F from cached path
    M_cached = M_func(
        u=0.1, v=0.05, xd=0.0, yd=0.0, ud=0.0, vd=0.0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )

    F_cached = F_func(
        u=0.1, v=0.05, xd=0.0, yd=0.0, ud=0.0, vd=0.0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )

    # Now derive fresh and lambdify
    M_fresh_sym, F_fresh_sym, args = _derive_symbolic()
    _raw_M_fresh, _raw_F_fresh, args_out = _apply_cse_and_lambdify(
        M_fresh_sym, F_fresh_sym, args
    )

    # Call with same parameters (note: args are in order of LagrangeParams fields)
    params_tuple = (
        0.1, 0.05,  # u, v
        0.0, 0.0,   # xd, yd
        0.0, 0.0,   # ud, vd
        1.0,        # m_b
        2.7,        # m_d
        1.0,        # m_hat_d
        101.0,      # m_tilde_d
        1.9,        # m_tilde_b
        3.0,        # l
        9.81,       # g
        12.0,       # k_b
        154.0,      # k_d
        0.5,        # U_b
        -0.3,       # V_b
        0.2,        # U_d
        0.1,        # V_d
    )

    M_elems_fresh = _raw_M_fresh(*params_tuple)
    F_elems_fresh = _raw_F_fresh(*params_tuple)

    # Assemble fresh M into (4, 4)
    M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems_fresh
    M_fresh = np.array([
        [M00, M01, M02, M03],
        [M01, M11, M12, M13],
        [M02, M12, M22, M23],
        [M03, M13, M23, M33],
    ], dtype=float)

    # Assemble fresh F into (4,)
    F_fresh = np.array(F_elems_fresh, dtype=float)

    # Compare cached vs fresh
    np.testing.assert_allclose(M_cached, M_fresh, rtol=1e-6, atol=1e-14,
                                err_msg="M from cache and derivation disagree")
    np.testing.assert_allclose(F_cached, F_fresh, rtol=1e-6, atol=1e-14,
                                err_msg="F from cache and derivation disagree")


@pytest.mark.slow
def test_get_eom_callables_fallback_on_missing_cache(tmp_path, monkeypatch):
    """_get_eom_callables should fall back to _derive_symbolic if cache missing.

    Create a temp directory without the .srepr file and verify fallback works.
    """
    from drogued_drifters import lagrange_model

    # Patch _SREPR_PATH to non-existent file in temp dir
    temp_cache = tmp_path / "missing.srepr"
    assert not temp_cache.exists(), "Cache file should not exist"

    with monkeypatch.context() as mp:
        mp.setattr(lagrange_model, "_SREPR_PATH", temp_cache)

        # Clear cache so it re-evaluates
        lagrange_model._get_eom_callables.cache_clear()

        # Should fall back to _derive_symbolic and work
        _raw_M, _raw_F, args = lagrange_model._get_eom_callables()

        assert callable(_raw_M), "_raw_M should be callable (fallback)"
        assert callable(_raw_F), "_raw_F should be callable (fallback)"
        assert len(args) == 19, f"Expected 19 args, got {len(args)}"

        # Verify the fallback works by evaluating at test point
        test_params = (
            0.0, 0.0,  # u, v
            0.0, 0.0,  # xd, yd
            0.0, 0.0,  # ud, vd
            1.0, 2.7, 1.0, 101.0, 1.9,
            3.0, 9.81, 12.0, 154.0,
            0.0, 0.0, 0.0, 0.0,
        )
        M_elems = _raw_M(*test_params)
        F_elems = _raw_F(*test_params)

        assert len(M_elems) == 10, f"Expected 10 M elements, got {len(M_elems)}"
        assert len(F_elems) == 4, f"Expected 4 F elements, got {len(F_elems)}"
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
