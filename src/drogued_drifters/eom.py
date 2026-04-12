"""Generic EOM machinery: caching, lambdification, packing, evaluation.

All functions are parameterized on a model instance (any subclass of
LagrangianMechanicsModel). Model-specific code (Physics/State NamedTuples,
_derive_symbolic) lives in the model modules.
"""

import hashlib
import inspect
import os
import pickle
import sys
import warnings

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Packer: maps lambda parameter names to NamedTuple fields
# ---------------------------------------------------------------------------


def _build_packer(raw_func, physics_type, state_type):
    """Inspect raw_func's signature, return a pack_eom_args(physics, state) callable.

    Called once (cached). Maps each lambda parameter name to a field in
    the Physics or State NamedTuple by name. Returns a closure that
    assembles the positional arg tuple from (physics, state).

    Args:
        raw_func: A lambdified callable whose parameter names exactly match
            fields in the Physics or State NamedTuple.
        physics_type: The Physics NamedTuple class.
        state_type: The State NamedTuple class.

    Returns:
        A callable ``pack_eom_args(physics, state)`` that returns a positional
        argument tuple suitable for calling ``raw_func(*pack_eom_args(p, s))``.

    Raises:
        KeyError: If any parameter name in ``raw_func``'s signature is not
            found in either NamedTuple's fields.
    """
    param_names = list(inspect.signature(raw_func).parameters)
    physics_fields = physics_type._fields
    state_fields = state_type._fields

    indices = []  # list of ('p'|'s', field_index)
    for name in param_names:
        if name in physics_fields:
            indices.append(("p", physics_fields.index(name)))
        elif name in state_fields:
            indices.append(("s", state_fields.index(name)))
        else:
            raise KeyError(
                f"Lambda param {name!r} not in {physics_type.__name__} "
                f"or {state_type.__name__} fields"
            )

    def pack_eom_args(physics, state):
        return tuple(physics[i] if src == "p" else state[i] for src, i in indices)

    return pack_eom_args


# ---------------------------------------------------------------------------
# Cache key and load/derive
# ---------------------------------------------------------------------------


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
        except Exception as e:
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


# ---------------------------------------------------------------------------
# Callable cache: keyed by model class name
# ---------------------------------------------------------------------------

_CALLABLE_CACHE = {}  # class_name -> (qdd_raw, M_raw, F_raw, pack_eom_args)


def _get_eom_callables(model):
    """Load or derive symbolic EOM; lambdify qdd, M, F; return raw callables.

    Args:
        model: A LagrangianMechanicsModel instance.

    Returns:
        (qdd_raw, M_raw, F_raw, pack_eom_args)
    """
    key = type(model).__name__
    if key not in _CALLABLE_CACHE:
        M_static, F_static, qdd_exprs, args = _load_or_derive(model)

        n_q = model.n_q

        # Extract M upper-triangle elements (symmetric)
        m_exprs = tuple(M_static[i, j]
                        for i in range(n_q) for j in range(i, n_q))

        # Extract F elements
        f_exprs = tuple(F_static[i] for i in range(n_q))

        # Lambdify with CSE
        qdd_raw = sp.lambdify(args, qdd_exprs, modules="numpy", cse=True)
        M_raw = sp.lambdify(args, m_exprs, modules="numpy", cse=True)
        F_raw = sp.lambdify(args, f_exprs, modules="numpy", cse=True)

        # Build packer once by inspecting the lambda signature
        pack_eom_args = _build_packer(qdd_raw, model.Physics, model.State)

        _CALLABLE_CACHE[key] = (qdd_raw, M_raw, F_raw, pack_eom_args)

    return _CALLABLE_CACHE[key]


# ---------------------------------------------------------------------------
# qdd evaluator cache: keyed by (class_name, backend)
# ---------------------------------------------------------------------------

_QDD_CACHE = {}  # (class_name, backend) -> qdd_func


def _make_qdd_func(model, backend="numpy"):
    """Build a qdd evaluator for the given model and backend.

    Args:
        model: A LagrangianMechanicsModel instance.
        backend: ``"numpy"`` (default) or ``"numba"``.

    Returns:
        A callable ``qdd_func(physics, state)`` that evaluates the
        generalized accelerations ``qdd = M^{-1}F``.
    """
    key = (type(model).__name__, backend)
    if key not in _QDD_CACHE:
        qdd_raw, _, _, pack_eom_args = _get_eom_callables(model)

        if backend == "numpy":
            raw = qdd_raw
        elif backend == "numba":
            from numba import njit

            raw = njit(qdd_raw)

            # Warm up the JIT
            _n_args = len(model.Physics._fields) + len(model.State._fields)
            _dummy_args = tuple(
                np.ones(1) if i >= len(model.Physics._fields) else 1.0
                for i in range(_n_args)
            )
            raw(*_dummy_args)
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. Must be 'numpy' or 'numba'."
            )

        def qdd_func(physics, state):
            u_arr = np.asarray(state[0])
            batch_ndim = u_arr.ndim

            result = raw(*pack_eom_args(physics, state))

            if batch_ndim == 0:
                return np.array(result, dtype=float)
            else:
                return np.column_stack(result)

        _QDD_CACHE[key] = qdd_func

    return _QDD_CACHE[key]


# ---------------------------------------------------------------------------
# Public evaluation functions (take model instance as first argument)
# ---------------------------------------------------------------------------


def eval_qdd(model, physics, state, *, backend="numpy"):
    """Evaluate generalized accelerations qdd = M^{-1}F.

    Args:
        model: A LagrangianMechanicsModel instance.
        physics: Physics NamedTuple instance.
        state: State NamedTuple instance.
        backend: ``"numpy"`` or ``"numba"``.

    Returns:
        Generalized accelerations ``qdd = M^{-1}F``:
        - ``(n_q,)`` array for scalar input.
        - ``(N, n_q)`` array for batch input.
    """
    return _make_qdd_func(model, backend)(physics, state)


def eval_M(model, physics, state):
    """Evaluate the mass matrix M.

    Args:
        model: A LagrangianMechanicsModel instance.
        physics: Physics NamedTuple instance.
        state: State NamedTuple instance.

    Returns:
        Mass matrix:
        - ``(n_q, n_q)`` for scalar input.
        - ``(N, n_q, n_q)`` for batch input.
    """
    _, M_raw, _, pack_eom_args = _get_eom_callables(model)
    n_q = model.n_q

    u_arr = np.asarray(state[0])
    batch_ndim = u_arr.ndim

    M_elems = M_raw(*pack_eom_args(physics, state))

    if batch_ndim == 0:
        # Scalar: assemble (n_q, n_q)
        M = np.zeros((n_q, n_q))
        k = 0
        for i in range(n_q):
            for j in range(i, n_q):
                M[i, j] = M[j, i] = float(M_elems[k])
                k += 1
    else:
        # Batch: assemble (N, n_q, n_q)
        N = u_arr.shape[0]
        M = np.zeros((N, n_q, n_q))
        k = 0
        for i in range(n_q):
            for j in range(i, n_q):
                val = np.broadcast_to(M_elems[k], N)
                M[:, i, j] = val
                M[:, j, i] = val
                k += 1

    return M


def eval_F(model, physics, state):
    """Evaluate the force vector F.

    Args:
        model: A LagrangianMechanicsModel instance.
        physics: Physics NamedTuple instance.
        state: State NamedTuple instance.

    Returns:
        Force vector:
        - ``(n_q,)`` for scalar input.
        - ``(N, n_q)`` for batch input.
    """
    _, _, F_raw, pack_eom_args = _get_eom_callables(model)
    n_q = model.n_q

    u_arr = np.asarray(state[0])
    batch_ndim = u_arr.ndim

    F_elems = F_raw(*pack_eom_args(physics, state))

    if batch_ndim == 0:
        F = np.array(F_elems, dtype=float)
    else:
        N = u_arr.shape[0]
        F = np.column_stack([np.broadcast_to(f, N) for f in F_elems])

    return F
