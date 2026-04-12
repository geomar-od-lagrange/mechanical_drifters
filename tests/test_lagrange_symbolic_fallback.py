"""Tests for symbolic derivation fallback (_derive_symbolic).

Marked @pytest.mark.slow because symbolic derivation takes 30-60s.
"""

import numpy as np
import pytest

from drogued_drifters.models.drogued_drifter import DroguedDrifter, DrifterPhysics, EOMState
from drogued_drifters.eom import (
    eval_M,
    eval_F,
    _load_or_derive,
    _get_eom_callables,
    _build_packer,
)


@pytest.mark.slow
def test_derive_symbolic_produces_matrices():
    """Verify _derive_symbolic returns symbolic M and F of correct shape."""
    dd = DroguedDrifter()
    M_sub, F_sub, args = dd._derive_symbolic()

    assert M_sub.shape == (4, 4), f"Expected M shape (4,4), got {M_sub.shape}"
    assert F_sub.shape == (4, 1), f"Expected F shape (4,1), got {F_sub.shape}"

    expected_n_args = len(DrifterPhysics._fields) + len(EOMState._fields)
    assert len(args) == expected_n_args, f"Expected {expected_n_args} args, got {len(args)}"


@pytest.mark.slow
def test_derive_symbolic_finite_values():
    """Derived M and F should evaluate to finite values at test points."""
    dd = DroguedDrifter()
    M_sub, F_sub, args = dd._derive_symbolic()

    import sympy as sp

    test_values = {
        "m_b": 1.0, "m_d": 2.7, "m_hat_d": 1.0,
        "m_tilde_d": 101.0, "m_tilde_b": 1.9,
        "l": 3.0, "g": 9.81, "k_b": 12.0, "k_d": 154.0,
        "u": 0.1, "v": 0.05,
        "xd": 0.0, "yd": 0.0, "ud": 0.0, "vd": 0.0,
        "U_b": 0.5, "V_b": -0.3, "U_d": 0.2, "V_d": 0.1,
    }

    subs_dict = {sym: test_values[str(sym)] for sym in args}
    M_num = M_sub.subs(subs_dict)
    F_num = F_sub.subs(subs_dict)

    M_vals = np.array([[float(M_num[i, j]) for j in range(4)] for i in range(4)])
    F_vals = np.array([float(F_num[i]) for i in range(4)])

    assert np.all(np.isfinite(M_vals)), f"M has non-finite values: {M_vals}"
    assert np.all(np.isfinite(F_vals)), f"F has non-finite values: {F_vals}"


@pytest.mark.slow
def test_derived_vs_cached_numerical_agreement():
    """Freshly derived M/F must agree numerically with cached M/F."""
    dd = DroguedDrifter()
    test_physics = DrifterPhysics(
        m_b=1.0, m_d=2.7, m_hat_d=1.0,
        m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
    )
    test_state = EOMState(
        u_stereo=0.1, v_stereo=0.05,
        xd=0.0, yd=0.0,
        ud_stereo=0.0, vd_stereo=0.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )

    M_cached = eval_M(dd, test_physics, test_state)
    F_cached = eval_F(dd, test_physics, test_state)

    import sympy as sp

    M_fresh_sym, F_fresh_sym, args = dd._derive_symbolic()

    m_exprs = tuple(M_fresh_sym[i, j] for i in range(4) for j in range(i, 4))
    f_exprs = tuple(F_fresh_sym[i] for i in range(4))

    M_raw_fresh = sp.lambdify(args, m_exprs, modules="numpy", cse=True)
    F_raw_fresh = sp.lambdify(args, f_exprs, modules="numpy", cse=True)

    packer = _build_packer(M_raw_fresh, DrifterPhysics, EOMState)
    params_tuple = packer(test_physics, test_state)

    M_elems_fresh = M_raw_fresh(*params_tuple)
    F_elems_fresh = F_raw_fresh(*params_tuple)

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

    F_fresh = np.array(F_elems_fresh, dtype=float)

    np.testing.assert_allclose(M_cached, M_fresh, rtol=1e-6, atol=1e-14, err_msg="M from cache and derivation disagree")
    np.testing.assert_allclose(F_cached, F_fresh, rtol=1e-6, atol=1e-14, err_msg="F from cache and derivation disagree")


@pytest.mark.slow
def test_get_eom_callables_fallback_on_missing_cache(tmp_path, monkeypatch):
    """_get_eom_callables should fall back to _derive_symbolic if cache missing."""
    from drogued_drifters import eom

    dd = DroguedDrifter()

    temp_cache = tmp_path / "missing.pkl"
    assert not temp_cache.exists()

    # Clear caches
    eom._CALLABLE_CACHE.pop("DroguedDrifter", None)
    eom._QDD_CACHE.pop(("DroguedDrifter", "numpy"), None)

    original_cache_path = type(dd)._cache_path
    monkeypatch.setattr(type(dd), "_cache_path", property(lambda self: temp_cache))

    try:
        qdd_raw, M_raw, F_raw, pack_eom_args = eom._get_eom_callables(dd)

        assert callable(qdd_raw)
        assert callable(M_raw)
        assert callable(F_raw)

        expected_n_args = len(DrifterPhysics._fields) + len(EOMState._fields)

        test_physics = DrifterPhysics(
            m_b=1.0, m_d=2.7, m_hat_d=1.0,
            m_tilde_d=101.0, m_tilde_b=1.9,
            l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        )
        test_state = EOMState(
            u_stereo=0.0, v_stereo=0.0,
            xd=0.0, yd=0.0,
            ud_stereo=0.0, vd_stereo=0.0,
            U_b=0.0, V_b=0.0, U_d=0.0, V_d=0.0,
        )
        packed = pack_eom_args(test_physics, test_state)
        qdd_elems = qdd_raw(*packed)
        M_elems = M_raw(*packed)
        F_elems = F_raw(*packed)

        assert len(qdd_elems) == 4
        assert len(M_elems) == 10
        assert len(F_elems) == 4
        assert all(np.isfinite(q) for q in qdd_elems)
        assert all(np.isfinite(m) for m in M_elems)
        assert all(np.isfinite(f) for f in F_elems)
    finally:
        eom._CALLABLE_CACHE.pop("DroguedDrifter", None)
        eom._QDD_CACHE.pop(("DroguedDrifter", "numpy"), None)
        monkeypatch.setattr(type(dd), "_cache_path", original_cache_path)


@pytest.mark.slow
def test_symbolic_derivation_takes_time():
    """Document that _derive_symbolic is expensive (~30-60s)."""
    import time

    dd = DroguedDrifter()
    start = time.time()
    M_sub, F_sub, args = dd._derive_symbolic()
    elapsed = time.time() - start

    assert M_sub.shape == (4, 4)
    assert elapsed > 0.0
