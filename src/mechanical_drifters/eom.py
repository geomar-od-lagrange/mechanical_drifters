"""Generic EOM machinery: lambdification, packing, evaluation.

All functions are parameterized on a model instance (any subclass of
LagrangianMechanicsModel). Model-specific code (Physics/State NamedTuples,
_derive_symbolic) lives in the model modules.

Disk-cache logic lives in ``caching.py``.
"""

import numpy as np
import sympy as sp

from .caching import _load_or_derive


# ---------------------------------------------------------------------------
# Packer: Physics fields then State fields — trivial by construction
# ---------------------------------------------------------------------------


def pack_eom_args(physics, state):
    """Pack physics and state NamedTuples into a flat positional arg tuple.

    The lambdified arg order is Physics fields then State fields
    by construction in ``_derive_symbolic``.
    """
    return (*physics, *state)



# ---------------------------------------------------------------------------
# Callable + evaluator cache: keyed by (class_name, backend)
# ---------------------------------------------------------------------------

_CALLABLE_CACHE = {}  # class_name -> (M_raw, F_raw)
_QDD_CACHE = {}  # (class_name, backend) -> qdd_func


def get_eom_callables(model, backend="numpy"):
    """Load or derive symbolic EOM; lambdify; return backend-wrapped qdd + raw M, F.

    Args:
        model: A LagrangianMechanicsModel instance.
        backend: ``"numpy"`` (default) or ``"numba"``.

    Returns:
        ``(qdd_func, M_raw, F_raw, pack_eom_args)``

        - ``qdd_func(physics, state, *, batch)``: backend-wrapped
          evaluator. With ``batch=False``, returns ``(n_q,)`` from scalar
          State fields. With ``batch=True``, returns ``(N, n_q)`` from
          ``(N,)``-shaped State fields.
        - ``M_raw``: raw lambdified mass matrix (for exploration).
        - ``F_raw``: raw lambdified force vector (for exploration).
        - ``pack_eom_args``: ``pack_eom_args(physics, state) -> tuple``
          for calling ``M_raw`` and ``F_raw``.
    """
    cls_name = type(model).__name__

    # --- raw lambdification (shared across backends) ---
    if cls_name not in _CALLABLE_CACHE:
        M_static, F_static, qdd_exprs, args = _load_or_derive(model)

        qdd_raw = sp.lambdify(args, qdd_exprs, modules="numpy", cse=True)
        M_raw = sp.lambdify(args, M_static, modules="numpy", cse=True)
        F_raw = sp.lambdify(args, F_static, modules="numpy", cse=True)

        _CALLABLE_CACHE[cls_name] = (qdd_raw, M_raw, F_raw)

    qdd_raw, M_raw, F_raw = _CALLABLE_CACHE[cls_name]

    # --- backend-wrapped qdd evaluator ---
    qdd_key = (cls_name, backend)
    if qdd_key not in _QDD_CACHE:
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

        def qdd_func(physics, state, *, batch):
            result = raw(*pack_eom_args(physics, state))
            if batch:
                return np.column_stack(result)
            else:
                return np.array(result, dtype=float)

        _QDD_CACHE[qdd_key] = qdd_func

    qdd_func = _QDD_CACHE[qdd_key]

    return qdd_func, M_raw, F_raw, pack_eom_args
