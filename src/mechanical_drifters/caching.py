"""Disk-cache for symbolic EOM derivations.

Caches the expensive symbolic derivation (M, F, qdd expressions) in
pickle files keyed by a hash of the derivation source, SymPy version,
and Python version.  A cache miss triggers re-derivation (~2 min for
DroguedDrifter, negligible for PointSurfaceDrifter).
"""

import hashlib
import inspect
import os
import pickle
import sys
import warnings

import sympy as sp


def _cache_key(derive_fn):
    """Hash of derive function source + sympy version + Python version."""
    source = inspect.getsource(derive_fn)
    key_data = (
        source
        + sp.__version__
        + str(sys.version_info[:2])
        + str(pickle.HIGHEST_PROTOCOL)
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _load_or_derive(model):
    """Load symbolic EOM from pickle cache, or derive from scratch.

    Args:
        model: A LagrangianMechanicsModel instance.

    Returns:
        (M_static, F_static, qdd_exprs, args)
        where qdd_exprs is a tuple of n_q scalar sympy expressions
        representing M^{-1}F (the generalized accelerations).
    """
    cache_path = model._cache_path
    key = _cache_key(model._derive_symbolic)

    if cache_path is not None and cache_path.exists():
        try:
            cached = pickle.loads(cache_path.read_bytes())
            if cached.get("key") == key:
                return cached["M"], cached["F"], cached["qdd"], cached["args"]
        except (OSError, pickle.UnpicklingError, KeyError) as e:
            warnings.warn(f"EOM cache load failed: {e}", stacklevel=2)

    # Miss or stale — re-derive (slow, ~2 min for DroguedDrifter)
    warnings.warn(
        "EOM cache miss — running symbolic derivation. "
        "This happens once after code or sympy version changes.",
        stacklevel=2,
    )
    M_static, F_static, args = model._derive_symbolic()
    qdd_vec = M_static.LUsolve(F_static)
    qdd_exprs = tuple(qdd_vec[i] for i in range(model.n_q))

    if cache_path is not None:
        data = {
            "key": key,
            "M": M_static,
            "F": F_static,
            "qdd": qdd_exprs,
            "args": args,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(data))
            os.replace(tmp, cache_path)
        except OSError:
            pass  # read-only install — skip cache write

    return M_static, F_static, qdd_exprs, args
