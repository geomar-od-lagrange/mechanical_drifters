# Architecture v4: class-based multi-object generalization

Replaces proposal C's `ModelSpec` frozen dataclass with a
`LagrangianMechanicsModel` base class. Builds on proposal B's
function-first ideas (velocity as arrays, explicit parameter plumbing,
backend as a parameter) and proposal C's analysis of what is generic vs.
model-specific. The base class carries the same information as C's 12-field
`ModelSpec`, but as overridable methods and class attributes rather than
callable fields on a data object.

**Why "LagrangianMechanicsModel"?** The package uses Lagrangian mechanics
(variational principle, generalized coordinates, Euler-Lagrange equations)
to derive the equations of motion. "LagrangianModel" would be ambiguous:
in ocean science, "Lagrangian" usually means particle tracking, which is
the *application* of this code but not the *method*. The full name avoids
the confusion.

## Design motivation

Proposal C's `ModelSpec` is a frozen dataclass with 12 fields, 5 of which
are callables. Adding a new model means constructing a `ModelSpec(...)` with
the right functions passed in the right slots. This is a data-driven
approach that works well for experienced developers but fails the target
audience test: developers who have just learned about modules
and classes.

The familiar pattern for this audience is:

```python
class MyModel(SomeBaseClass):
    """Override the methods that matter."""

    def some_method(self):
        ...
```

Students understand this from textbook examples. "Subclass this, fill in
these methods" is a pattern they have seen. A frozen dataclass with 12
fields of callables is not.

The base class approach also provides:

- **Discoverability.** `help(LagrangianMechanicsModel)` shows all
  overridable methods with docstrings. A `ModelSpec` dataclass shows 12
  type-annotated fields but no guidance on what each callable should do.
- **Defaults and validation.** The base class can provide concrete default
  implementations (e.g., the standard `[q, qdot]` derivative packing) and
  validate that subclasses have filled in required pieces.
- **IDE support.** Method override completion works out of the box.

## What is generic vs. model-specific

This table summarizes what varies across models and what stays the same.
It is the same analysis as proposal C, just expressed as method categories
instead of `ModelSpec` fields.

| Concern | Generic | Model-specific |
|---|---|---|
| Symbolic derivation (`_derive_symbolic`) | Cache loading, pickle I/O, hash invalidation | The actual Lagrangian, generalized coordinates, sympy expressions |
| Lambdification | `sp.lambdify` with CSE, `_build_packer`, batch reshaping | Symbol-to-field-name mapping (determined by Physics and State types) |
| EOM evaluation (`qdd_func`) | Pack args, call raw lambda, reshape output, optional numba JIT | Nothing (fully generic once packer exists) |
| RHS assembly (`_rhs_batch`) | NaN guarding, solve_ivp wrapper | State vector layout, depth queries, state construction, derivative packing |
| ODE integration | `solve_ivp` call, flat/structured reshaping, convergence diagnostic | Which state indices are the "drift velocity answer" |
| Parcels coupling | Profile extraction from fieldset, position update | Maximum depth to sample (determined by geometry) |
| Velocity interpolation | Linear interpolation in z | Nothing (fully generic) |

## The base class

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple
import numpy as np


class LagrangianMechanicsModel(ABC):
    """Base class for drifting-object models derived via Lagrangian mechanics.

    Subclass this to define a new drifting object. You must provide:

    1. A ``Physics`` NamedTuple (class attribute) with the physical
       constants for your object (masses, lengths, drag coefficients, ...).
    2. A ``State`` NamedTuple (class attribute) with the per-timestep
       variables: generalized velocities plus ocean current forcing terms.
    3. ``n_q``: the number of generalized coordinates.
    4. ``_derive_symbolic()``: the sympy derivation of M and F.
    5. ``_z_query(physics, Y)``: which depths to sample velocity at.
    6. ``_build_state(Y, velocity_dict)``: assemble the State NamedTuple.
    7. ``_drift_velocity_indices``: which state components are the answer.
    8. ``_max_depth(physics)``: deepest depth to sample for Parcels.

    The base class provides:

    - EOM caching, lambdification, and numeric evaluation.
    - ODE integration (batch steady-state and single-particle trajectory).
    - Parcels kernel generation.
    - Derivative packing (overridable if your state layout is non-standard).
    - A constructor that takes a Physics instance and backend choice.

    Example
    -------
    See ``DroguedDrifter`` for a complete implementation.
    """

    # --- Class attributes (override in subclass) ---

    Physics: type = None  # NamedTuple class for physical constants
    State: type = None    # NamedTuple class for per-timestep state + forcing
    n_q: int = None       # number of generalized coordinates

    # Indices into the state vector [q0, q1, ..., qdot0, qdot1, ...] that
    # are the "drift velocity" output. For DroguedDrifter: (4, 5) = (xd, yd).
    _drift_velocity_indices: tuple = None

    # Where to cache the pickled symbolic derivation. Each subclass should
    # set this to a unique path under data/.
    _cache_path: Path = None

    # --- Constructor ---

    def __init__(self, physics=None, *, backend="numpy"):
        """Create a model instance.

        Args:
            physics: A Physics NamedTuple with physical constants.
                If None, uses ``self.default_physics()``.
            backend: "numpy" (default) or "numba".
        """
        if physics is None:
            physics = self.default_physics()
        self.physics = physics
        self.backend = backend
        self._qdd_func = _make_qdd_func(self, backend)

    # --- Abstract methods (must override) ---

    @staticmethod
    @abstractmethod
    def _derive_symbolic():
        """Derive symbolic M and F from the Lagrangian.

        This is where the physics lives. Use sympy to:
        1. Define generalized coordinates and physical parameters.
        2. Write down kinetic energy T, potential energy V, Lagrangian L = T - V.
        3. Compute generalized forces Q from non-conservative forces (drag).
        4. Apply the Euler-Lagrange equations.
        5. Extract M and F such that M * qdd = F.

        The symbols in the returned ``args`` tuple must have names that
        exactly match field names in your ``Physics`` and ``State``
        NamedTuples. This is how the automatic argument packer maps
        struct fields to lambda parameters.

        Returns:
            Tuple ``(M_static, F_static, args)`` where:
            - ``M_static``: sympy Matrix (n_q x n_q), the mass matrix.
            - ``F_static``: sympy Matrix (n_q x 1), the force vector.
            - ``args``: tuple of sympy Symbols in the order for lambdification.
        """

    @staticmethod
    @abstractmethod
    def _z_query(physics, Y):
        """Determine which depths to sample ocean velocity at.

        Given the current state of all particles, compute the depth(s)
        where each body in the model needs a velocity value. Different
        models have different bodies at different depths.

        Args:
            physics: Physics NamedTuple (for geometry parameters like
                pole length).
            Y: State array of shape ``(N, state_size)`` where
                ``state_size = 2 * n_q``.

        Returns:
            dict mapping body names (strings) to ``(N,)`` depth arrays
            [m, positive upward, 0 = surface].

        Example (DroguedDrifter):
            Returns ``{"buoy": zeros(N), "drogue": z_eff(N,)}``.
        """

    @staticmethod
    @abstractmethod
    def _build_state(Y, velocity_dict):
        """Assemble the State NamedTuple from state array and velocities.

        This is called once per RHS evaluation. It takes the raw ODE
        state array and the interpolated velocity at each body, and
        packs them into the State NamedTuple that the lambdified EOM
        functions expect.

        Args:
            Y: State array of shape ``(N, state_size)``.
            velocity_dict: dict mapping body names (same keys as
                ``_z_query`` returns) to ``(U, V)`` tuples of ``(N,)``
                arrays [m/s].

        Returns:
            A State NamedTuple instance with ``(N,)`` arrays in each
            field.
        """

    @staticmethod
    @abstractmethod
    def default_physics():
        """Return the default Physics instance for this model.

        This provides sensible defaults so users can write
        ``DroguedDrifter()`` without specifying all 9 parameters.

        Returns:
            A Physics NamedTuple with default values.
        """

    @staticmethod
    @abstractmethod
    def _max_depth(physics):
        """Maximum depth [m, positive] to sample from the fieldset.

        The Parcels coupling uses this to decide how many depth levels
        to extract from the ocean model data.

        Args:
            physics: Physics NamedTuple.

        Returns:
            float, positive depth in meters.
        """

    # --- Concrete methods with sensible defaults (override if needed) ---

    @staticmethod
    def _pack_derivatives(Y, qdd):
        """Pack ODE derivatives from state array and accelerations.

        Default implementation assumes the standard Lagrangian state
        layout ``[q_0, ..., q_{n-1}, qdot_0, ..., qdot_{n-1}]``
        where ``dq/dt = qdot`` and ``d(qdot)/dt = qdd``.

        Override this if your model has a non-standard state layout
        (e.g., extra non-dynamic state variables).

        Args:
            Y: State array, shape ``(N, 2*n_q)``.
            qdd: Generalized accelerations, shape ``(N, n_q)``.

        Returns:
            dY: Derivative array, shape ``(N, 2*n_q)``.
        """
        n_q = qdd.shape[1]
        dY = np.empty_like(Y)
        dY[:, :n_q] = Y[:, n_q:]   # d(q)/dt = qdot
        dY[:, n_q:] = qdd           # d(qdot)/dt = qdd
        return dY

    @property
    def state_size(self):
        """Total state vector length (2 * n_q for standard layouts)."""
        return 2 * self.n_q
```

### What developers see

When a developer opens `LagrangianMechanicsModel`, they see:

1. **Three class attributes** to set: `Physics`, `State`, `n_q`.
2. **Five methods to fill in**, each with a clear docstring and example.
3. **One method they probably don't need to touch** (`_pack_derivatives`),
   with a default that handles the common case.
4. A **constructor** they can call with just `MyModel()` thanks to
   `default_physics()`.

The data flow is readable from top to bottom: "derive the equations, say
where the bodies are, build the state, define the defaults."

### What is NOT in the base class

The base class does not contain the ODE integration, Parcels coupling,
or EOM evaluation logic. Those live in separate modules (`solve.py`,
`parcels.py`, `eom.py`) as free functions that take a model instance.
The base class is the *contract*, not the *implementation*.

This is a deliberate choice. Putting `steady_state_drift` as a method on
the base class would make the class >300 lines and mix concerns (physics
definition + ODE integration + Parcels I/O). Instead:

- The model defines *what* to compute (physics, state layout, depth
  queries).
- Separate modules define *how* to compute it (caching, lambdification,
  integration, Parcels coupling).
- The model instance is passed to those modules as a parameter.

Convenience methods can be added on the model class as thin wrappers (see
DroguedDrifter below), but they delegate to the free functions.

## Generic machinery: how it uses the model

### `eom.py` -- parameterized on the model instance

The key change from proposal C: instead of `spec.physics_type`, the
generic code reads `model.Physics`. Instead of `spec.derive_symbolic()`,
it calls `model._derive_symbolic()`. The model instance replaces the
ModelSpec data object.

```python
# eom.py

import functools
import hashlib
import inspect
import pickle
import warnings

import numpy as np
import sympy as sp


def _build_packer(raw_func, physics_type, state_type):
    """Inspect raw_func's signature, map to Physics/State field names.

    Identical to current _build_packer but takes the types as arguments
    instead of hardcoding DrifterPhysics and EOMState.
    """
    param_names = list(inspect.signature(raw_func).parameters)
    physics_fields = physics_type._fields
    state_fields = state_type._fields

    indices = []
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


def _cache_key(derive_fn):
    """Hash of derive_fn source + sympy version + Python version."""
    source = inspect.getsource(derive_fn)
    key_data = source + sp.__version__ + str(sys.version_info[:2])
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _load_or_derive(model):
    """Load symbolic EOM from pickle cache, or derive from scratch.

    Args:
        model: LagrangianMechanicsModel instance (or class -- only needs
            _derive_symbolic, _cache_path, and n_q).

    Returns:
        (M_static, F_static, qdd_exprs, args)
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

    warnings.warn(
        "EOM cache miss -- running symbolic derivation. "
        "This happens once after code or sympy version changes.",
        stacklevel=2,
    )
    M_static, F_static, args = model._derive_symbolic()
    qdd_vec = M_static.LUsolve(F_static)
    qdd_exprs = tuple(qdd_vec[i] for i in range(model.n_q))

    if cache_path is not None:
        data = {"key": key, "M": M_static, "F": F_static,
                "qdd": qdd_exprs, "args": args}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(data))
            os.replace(tmp, cache_path)
        except OSError:
            pass

    return M_static, F_static, qdd_exprs, args


# Cache by model class name -- all instances of the same model share callables.
_CALLABLE_CACHE = {}

def _get_eom_callables(model):
    """Get or build (qdd_raw, M_raw, F_raw, pack_eom_args) for a model."""
    key = type(model).__name__
    if key not in _CALLABLE_CACHE:
        M_static, F_static, qdd_exprs, args = _load_or_derive(model)

        m_exprs = tuple(M_static[i, j]
                        for i in range(model.n_q)
                        for j in range(i, model.n_q))
        f_exprs = tuple(F_static[i] for i in range(model.n_q))

        qdd_raw = sp.lambdify(args, qdd_exprs, modules="numpy", cse=True)
        M_raw = sp.lambdify(args, m_exprs, modules="numpy", cse=True)
        F_raw = sp.lambdify(args, f_exprs, modules="numpy", cse=True)

        pack_eom_args = _build_packer(qdd_raw, model.Physics, model.State)
        _CALLABLE_CACHE[key] = (qdd_raw, M_raw, F_raw, pack_eom_args)

    return _CALLABLE_CACHE[key]


# Cache by (model class name, backend).
_QDD_CACHE = {}

def _make_qdd_func(model, backend="numpy"):
    """Build a qdd evaluator for the given model and backend."""
    key = (type(model).__name__, backend)
    if key not in _QDD_CACHE:
        qdd_raw, _, _, pack_eom_args = _get_eom_callables(model)

        if backend == "numpy":
            raw = qdd_raw
        elif backend == "numba":
            from numba import njit
            raw = njit(qdd_raw)
            # JIT warmup ...
        else:
            raise ValueError(f"Unknown backend {backend!r}")

        n_q = model.n_q

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


# Public evaluation functions (take model instance)

def eval_qdd(model, physics, state, *, backend="numpy"):
    """Evaluate generalized accelerations qdd = M^{-1}F."""
    return _make_qdd_func(model, backend)(physics, state)

def eval_M(model, physics, state):
    """Evaluate the mass matrix M. Returns (n_q, n_q) or (N, n_q, n_q)."""
    ...  # Same pattern as current M_func, but uses model.n_q

def eval_F(model, physics, state):
    """Evaluate the force vector F. Returns (n_q,) or (N, n_q)."""
    ...  # Same pattern as current F_func
```

The caching strategy is simple: cache by class name. All `DroguedDrifter`
instances share the same compiled EOM callables. A `SparBuoy` gets its own
entry. No need for LRU or weak refs -- there are 2-3 models.

### `solve.py` -- generic ODE integration

```python
# solve.py

import numpy as np
from scipy.integrate import solve_ivp

from .eom import _make_qdd_func
from .velocity import interpolate_profile


def _rhs_batch(Y, *, model, physics, depth_levels, U_profiles, V_profiles, qdd_func):
    """Vectorized RHS for N particles, generic over model type.

    The model instance provides the three object-specific operations:
    1. model._z_query(physics, Y) -- which depths to sample
    2. model._build_state(Y, velocity_dict) -- assemble State NamedTuple
    3. model._pack_derivatives(Y, qdd) -- write derivatives

    Everything else is generic: velocity interpolation, EOM evaluation,
    NaN guarding.

    Args:
        Y: (N, model.state_size) state array.
        model: LagrangianMechanicsModel instance.
        physics: Physics NamedTuple.
        depth_levels: (D,) depth levels, z-up ascending.
        U_profiles, V_profiles: (D, N) velocity profiles.
        qdd_func: callable(physics, state) -> (N, n_q).

    Returns:
        dY: (N, model.state_size) derivatives.
    """
    # 1. Query depths from generalized coordinates
    z_dict = model._z_query(physics, Y)

    # 2. Interpolate velocity at each queried depth
    velocity_dict = {}
    for name, z_arr in z_dict.items():
        U, V = interpolate_profile(z_arr, depth_levels, U_profiles, V_profiles)
        velocity_dict[name] = (U, V)

    # 3. Build model-specific state NamedTuple
    state = model._build_state(Y, velocity_dict)

    # 4. Evaluate accelerations (generic)
    qdd = qdd_func(physics, state)

    # 5. Guard NaN/inf (generic)
    bad = ~np.isfinite(qdd).all(axis=1)
    if np.any(bad):
        qdd[bad] = 0.0

    # 6. Pack derivatives (default or model-specific)
    return model._pack_derivatives(Y, qdd)


def steady_state_drift(
    model,
    physics,
    depth_levels,
    U_profiles,
    V_profiles,
    *,
    t_span=(0, 120),
    y0=None,
    atol=1e-3,
    rtol=1e-3,
    backend="numpy",
):
    """Compute steady-state drift for N particles.

    Args:
        model: LagrangianMechanicsModel instance.
        physics: Physics NamedTuple.
        depth_levels: (D,) z-up ascending.
        U_profiles, V_profiles: (D, N) velocity profiles.
        t_span: Integration window [s].
        y0: Initial state (N, state_size), or None for cold start (zeros).
        atol, rtol: ODE solver tolerances.
        backend: "numpy" or "numba".

    Returns:
        (drift_vel, Y_final, max_accel) where:
        - drift_vel: (N, len(drift_velocity_indices)) drift velocities.
        - Y_final: (N, state_size) final state (pass back as y0).
        - max_accel: scalar convergence diagnostic.
    """
    qdd_fn = _make_qdd_func(model, backend)
    ss = model.state_size

    if y0 is not None:
        N = y0.shape[0]
        y0_flat = y0.ravel()
    else:
        _, N = U_profiles.shape
        y0_flat = np.zeros(N * ss)

    def rhs_flat(t, y_flat):
        Y = y_flat.reshape(N, ss)
        dY = _rhs_batch(Y, model=model, physics=physics,
                        depth_levels=depth_levels,
                        U_profiles=U_profiles, V_profiles=V_profiles,
                        qdd_func=qdd_fn)
        return dY.ravel()

    sol = solve_ivp(rhs_flat, t_span, y0_flat, atol=atol, rtol=rtol)
    Y_final = sol.y[:, -1].reshape(N, ss)

    # Convergence diagnostic
    dY_final = _rhs_batch(Y_final, model=model, physics=physics,
                          depth_levels=depth_levels,
                          U_profiles=U_profiles, V_profiles=V_profiles,
                          qdd_func=qdd_fn)
    idx = list(model._drift_velocity_indices)
    max_accel = float(np.max(np.abs(dY_final[:, idx])))

    drift_vel = Y_final[:, idx]
    return drift_vel, Y_final, max_accel


def integrate(
    model,
    physics,
    depth_levels,
    U_profiles,
    V_profiles,
    *,
    t_span,
    y0=None,
    t_eval=None,
    atol=1e-3,
    rtol=1e-3,
    backend="numpy",
):
    """Integrate the ODE over a time span. Returns (t_array, Y_array).

    The raw solver interface. Model-specific convenience functions
    (e.g., DroguedDrifter.full_trajectory) wrap this and convert
    the output to an xr.Dataset with model-specific variable names.
    """
    ...
```

### `parcels.py` -- generic Parcels coupling

```python
# parcels.py

import numpy as np
from .velocity import interpolate_profile
from .solve import steady_state_drift

_DEG2M = 1852.0 * 60.0


def extract_profiles(particles, fieldset, max_depth):
    """Sample velocity profiles from fieldset. Returns (depth_levels, U, V)."""
    ...  # Same as current _extract_profiles but takes max_depth, not dd


def position_update(particles, xd_ms, yd_ms, fieldset):
    """Euler-forward position update. Unchanged."""
    ...


def make_kernel(model, physics=None, *, backend="numpy"):
    """Create a Parcels kernel for any LagrangianMechanicsModel.

    Args:
        model: LagrangianMechanicsModel instance (or class to instantiate).
        physics: Physics NamedTuple, or None for model defaults.
        backend: "numpy" or "numba".

    Returns:
        Kernel function (particles, fieldset) for pset.execute().
    """
    if isinstance(model, type):
        model = model(physics, backend=backend)
    elif physics is not None:
        model = type(model)(physics, backend=backend)

    physics = model.physics
    max_depth = model._max_depth(physics)

    def _kernel(particles, fieldset):
        depth_levels, U, V = extract_profiles(particles, fieldset, max_depth)
        drift_vel, _, _ = steady_state_drift(
            model, physics, depth_levels, U, V, backend=model.backend,
        )
        position_update(particles, drift_vel[:, 0], drift_vel[:, 1], fieldset)

    return _kernel
```

## DroguedDrifter as a subclass

This is the complete implementation. Students can read it end-to-end and
understand what a model needs to provide.

```python
# models/drogued_drifter.py

from pathlib import Path
from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel
from ..coords import uv_to_spherical, spherical_to_uv


# ---------- Physics ----------

class DrifterPhysics(NamedTuple):
    """Physical constants for a drogued drifter."""
    m_b: float          # buoy dry mass [kg]
    m_d: float          # drogue dry mass [kg]
    m_hat_d: float      # drogue buoyancy correction [kg]
    m_tilde_d: float    # drogue added mass [kg]
    m_tilde_b: float    # buoy added mass [kg]
    l: float            # pole length [m]
    g: float            # gravitational acceleration [m/s^2]
    k_b: float          # buoy drag coefficient [kg/m]
    k_d: float          # drogue drag coefficient [kg/m]


class EOMState(NamedTuple):
    """Per-timestep state: generalized velocities + ocean forcing."""
    u_stereo: float | np.ndarray
    v_stereo: float | np.ndarray
    xd: float | np.ndarray
    yd: float | np.ndarray
    ud_stereo: float | np.ndarray
    vd_stereo: float | np.ndarray
    U_b: float | np.ndarray
    V_b: float | np.ndarray
    U_d: float | np.ndarray
    V_d: float | np.ndarray


# ---------- Drag / added-mass helpers ----------

def drogue_horizontal_added_mass(*, rho, w_d, h_d, C_perp_d=np.pi / 4):
    ...  # unchanged

def buoy_horizontal_added_mass(*, rho, d_b, h_b, C_perp_b=1.0):
    ...  # unchanged

def drogue_horizontal_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2):
    ...  # unchanged

def buoy_horizontal_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0):
    ...  # unchanged


# ---------- The model ----------

class DroguedDrifter(LagrangianMechanicsModel):
    """A surface buoy connected by a rigid pole to a subsurface drogue.

    Both bodies experience quadratic drag from the ocean current at their
    respective depths. The equations of motion are derived from a
    Lagrangian formulation with 4 generalized coordinates:
    buoy position (x, y) and stereographic pole direction (u, v).

    State vector: [x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo]
    """

    Physics = DrifterPhysics
    State = EOMState
    n_q = 4
    _drift_velocity_indices = (4, 5)  # xd, yd
    _cache_path = Path(__file__).resolve().parent.parent / "data" / "eom_cache_drogued_drifter.pkl"

    @staticmethod
    def default_physics():
        """Callies et al. (2017) drifter at rho=1025 kg/m^3."""
        return DrifterPhysics(
            m_b=1.0, m_d=2.7, m_hat_d=1.0,
            m_tilde_d=101.0, m_tilde_b=1.9,
            l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        )

    @staticmethod
    def _derive_symbolic():
        """Derive M, F for the buoy-pole-drogue system."""
        # ... existing _derive_symbolic body, unchanged ...
        # Returns (M_static, F_static, args)

    @staticmethod
    def _z_query(physics, Y):
        """Buoy at surface, drogue at effective depth from pole tilt."""
        N = Y.shape[0]
        u, v = Y[:, 2], Y[:, 3]
        s = u**2 + v**2
        cos_theta = (s - 4) / (s + 4)
        z_eff = np.minimum(0.0, physics.l * cos_theta)
        return {"buoy": np.zeros(N), "drogue": z_eff}

    @staticmethod
    def _build_state(Y, velocity_dict):
        """Pack state vector and velocities into EOMState."""
        U_b, V_b = velocity_dict["buoy"]
        U_d, V_d = velocity_dict["drogue"]
        return EOMState(
            u_stereo=Y[:, 2], v_stereo=Y[:, 3],
            xd=Y[:, 4], yd=Y[:, 5],
            ud_stereo=Y[:, 6], vd_stereo=Y[:, 7],
            U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
        )

    @staticmethod
    def _max_depth(physics):
        """Drogue hangs at most one pole-length below the surface."""
        return physics.l

    # ---------- Convenience methods (DroguedDrifter-specific) ----------

    def full_trajectory(self, depth_levels, U_profile, V_profile, **kw):
        """Integrate single-particle trajectory, return xr.Dataset.

        Wraps solve.integrate() and converts internal stereographic
        coordinates to spherical angles for display.
        """
        from ..solve import integrate
        import xarray as xr

        t, Y = integrate(
            self, self.physics,
            depth_levels, U_profile[:, np.newaxis], V_profile[:, np.newaxis],
            **kw,
        )
        theta, phi, thetad, phid = uv_to_spherical(
            Y[:, 2], Y[:, 3], Y[:, 6], Y[:, 7],
        )
        return xr.Dataset(
            {"x": ("time", Y[:, 0]), "y": ("time", Y[:, 1]),
             "theta": ("time", theta), "phi": ("time", phi),
             "xd": ("time", Y[:, 4]), "yd": ("time", Y[:, 5]),
             "thetad": ("time", thetad), "phid": ("time", phid)},
            coords={"time": t},
        )

    def steady_state(self, depth_levels, U_profiles, V_profiles, **kw):
        """Compute steady-state drift velocities.

        Convenience wrapper around solve.steady_state_drift().
        Returns (xd_final, yd_final, Y_final, max_accel).
        """
        from ..solve import steady_state_drift
        drift_vel, Y_final, max_accel = steady_state_drift(
            self, self.physics,
            depth_levels, U_profiles, V_profiles,
            backend=self.backend, **kw,
        )
        return drift_vel[:, 0], drift_vel[:, 1], Y_final, max_accel
```

### Line count

The model-specific code above is approximately:

- `DrifterPhysics`: 11 lines
- `EOMState`: 12 lines
- `DroguedDrifter` class: ~75 lines (including convenience methods)
- Drag helpers: ~40 lines (unchanged)
- `_derive_symbolic`: ~130 lines (unchanged, moved here)
- **Total**: ~270 lines

This is comparable to C's `models/drogued_drifter.py` (~250 lines). The
base class adds ~80 lines of docstrings and method signatures. Not a
significant overhead.

## What SparBuoy would look like

A spar buoy is a single elongated float with a keel mass. It has 3
generalized coordinates: horizontal position (x, y) and tilt angle (theta).
The ocean current acts at the float (surface) and at the keel (depth).

```python
# models/spar_buoy.py

from pathlib import Path
from typing import NamedTuple
import numpy as np
from ..base import LagrangianMechanicsModel


class SparBuoyPhysics(NamedTuple):
    m_float: float      # float mass [kg]
    m_keel: float       # keel mass [kg]
    l_float: float      # float length [m]
    l_keel: float       # keel depth below surface [m]
    g: float            # gravitational acceleration [m/s^2]
    k_float: float      # float drag coefficient [kg/m]
    k_keel: float       # keel drag coefficient [kg/m]


class SparBuoyState(NamedTuple):
    theta: float | np.ndarray      # tilt angle
    xd: float | np.ndarray         # x velocity
    yd: float | np.ndarray         # y velocity
    thetad: float | np.ndarray     # angular velocity
    U_float: float | np.ndarray    # current at float
    V_float: float | np.ndarray
    U_keel: float | np.ndarray     # current at keel
    V_keel: float | np.ndarray


class SparBuoy(LagrangianMechanicsModel):
    """A spar buoy with a keel mass."""

    Physics = SparBuoyPhysics
    State = SparBuoyState
    n_q = 3                               # x, y, theta
    _drift_velocity_indices = (3, 4)      # xd, yd
    _cache_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "eom_cache_spar_buoy.pkl"
    )

    @staticmethod
    def default_physics():
        return SparBuoyPhysics(
            m_float=5.0, m_keel=10.0,
            l_float=1.0, l_keel=5.0,
            g=9.81, k_float=20.0, k_keel=50.0,
        )

    @staticmethod
    def _derive_symbolic():
        """Derive M, F for the spar buoy system."""
        # ... sympy derivation specific to the spar buoy Lagrangian ...
        # Returns (M_static, F_static, args) with 3x3 M.

    @staticmethod
    def _z_query(physics, Y):
        N = Y.shape[0]
        # Float at surface, keel at depth computed from tilt
        theta = Y[:, 2]
        z_keel = -physics.l_keel * np.cos(theta)
        return {"float": np.zeros(N), "keel": np.minimum(0.0, z_keel)}

    @staticmethod
    def _build_state(Y, velocity_dict):
        U_f, V_f = velocity_dict["float"]
        U_k, V_k = velocity_dict["keel"]
        return SparBuoyState(
            theta=Y[:, 2],
            xd=Y[:, 3], yd=Y[:, 4],
            thetad=Y[:, 5],
            U_float=U_f, V_float=V_f,
            U_keel=U_k, V_keel=V_k,
        )

    @staticmethod
    def _max_depth(physics):
        return physics.l_keel
```

That is the complete SparBuoy model (minus the sympy derivation body).
No changes to `eom.py`, `solve.py`, `velocity.py`, or `parcels.py`.

## Module layout

```
src/drogued_drifters/
    __init__.py               Public API re-exports
    base.py                   LagrangianMechanicsModel ABC
    eom.py                    Generic: derivation, caching, lambdification,
                                packing, qdd/M/F evaluation
    coords.py                 Stereographic <-> spherical transforms
    velocity.py               Profile interpolation (make_profile_sampler
                                or interpolate_profile)
    solve.py                  Generic ODE integration
    parcels.py                Generic Parcels coupling
    stokes.py                 Stokes drift profiles (unchanged)

    models/
        __init__.py
        drogued_drifter.py    DroguedDrifter + DrifterPhysics + EOMState +
                                _derive_symbolic + drag helpers
        spar_buoy.py          (future)

    data/
        eom_cache_drogued_drifter.pkl
```

### Dependency graph

```
base.py                    (abc, numpy -- no internal imports)
coords.py                  (standalone, numpy)
velocity.py                (standalone, numpy)
stokes.py                  (standalone, numpy)

eom.py                     (imports base for type hints only)
  |
  v
solve.py                   (imports eom, velocity)
  |
  v
parcels.py                 (imports solve, velocity)

models/drogued_drifter.py  (imports base, coords)
models/spar_buoy.py        (imports base)
```

No circular dependencies. Model modules import the base class but not
`eom.py` or `solve.py` -- they define structure and physics that is
*consumed* by those modules. The dependency arrow flows from generic
machinery toward model definitions at call time, via the model instance
passed as a parameter.

## Data flow

### Parcels path (production)

```
pset.execute(kernel)
  |
  v
_kernel(particles, fieldset)                    # from parcels.make_kernel
  |
  +-- extract_profiles(particles, fieldset, max_depth)
  |     -> (depth_levels, U_profiles, V_profiles)
  |
  +-- steady_state_drift(model, physics, depth_levels, U, V, backend=)
  |     |
  |     +-- _make_qdd_func(model, backend)  -> qdd_fn          [cached]
  |     +-- _rhs_batch(Y, model=model, physics=physics, ...)
  |     |     |
  |     |     +-- model._z_query(physics, Y) -> z_dict          [model-specific]
  |     |     +-- interpolate_profile(z, ...) for each body     [generic]
  |     |     +-- model._build_state(Y, velocity_dict) -> state [model-specific]
  |     |     +-- qdd_fn(physics, state) -> accelerations       [generic]
  |     |     +-- model._pack_derivatives(Y, qdd) -> dY         [default/override]
  |     |
  |     `-- solve_ivp(rhs_flat, ...)                            [generic]
  |
  +-- position_update(particles, xd, yd, fieldset)             [generic]
```

### Standalone use

```python
from drogued_drifters import DroguedDrifter
import numpy as np

dd = DroguedDrifter()  # default Callies et al. physics

# Build a velocity profile
z = np.linspace(-20, 0, 50)
U = 0.5 * np.exp(z / 5.0)
V = np.zeros_like(U)

# Single-particle trajectory
ds = dd.full_trajectory(z, U, V, t_span=(0, 120))
ds.xd.plot()

# Batch steady-state (N=100 particles)
U_batch = np.tile(U[:, np.newaxis], (1, 100))
V_batch = np.tile(V[:, np.newaxis], (1, 100))
xd, yd, Y_final, max_accel = dd.steady_state(z, U_batch, V_batch)
```

### Direct EOM study

```python
from drogued_drifters import DroguedDrifter, DrifterPhysics, EOMState
from drogued_drifters.eom import eval_qdd, eval_M, eval_F

dd = DroguedDrifter()  # only needed to identify the model
physics = dd.physics
state = EOMState(u_stereo=0.1, v_stereo=0.0, ...)

qdd = eval_qdd(dd, physics, state)
M = eval_M(dd, physics, state)
F = eval_F(dd, physics, state)
assert np.allclose(M @ qdd, F)
```

## Design decisions

### Why `@staticmethod` for the abstract methods

The abstract methods (`_derive_symbolic`, `_z_query`, `_build_state`,
`default_physics`, `_max_depth`) do not use `self`. They are pure
functions of their arguments. Making them static methods communicates this
clearly: "this method depends only on its arguments, not on instance
state."

This also means the methods can be called on the class directly
(`DroguedDrifter._z_query(physics, Y)`) without instantiation, which is
useful for testing.

The generic machinery calls them via the instance (`model._z_query(...)`)
which works fine for static methods.

### Why not put ODE integration on the base class

The base class could have `steady_state_drift` and `integrate` as
methods. Arguments for:

- Simpler API: `dd.steady_state(...)` instead of
  `steady_state_drift(dd, ...)`.
- One import instead of two.

Arguments against (why we keep them as free functions):

- **Separation of concerns.** The base class defines physics; the solver
  defines integration. Mixing them in one class makes it >300 lines and
  harder to understand.
- **Testability.** The solver can be tested with a mock model object.
  If it is a method, testing requires subclassing.
- **B's principle.** Proposal B established that the core computation is a
  function, not a method. The class is a convenience layer.

The compromise (used in the DroguedDrifter sketch above) is: convenience
methods on the subclass that delegate to free functions. Students use
`dd.steady_state(...)`. Power users use `steady_state_drift(model, ...)`.

### Why the physics NamedTuple is a class attribute, not a constructor parameter

The `Physics` type is set as a class attribute (`Physics = DrifterPhysics`)
rather than passed to the constructor. Reasons:

1. **It is a type, not a value.** The NamedTuple *class* is shared by all
   instances. Only the *instance* (with specific numeric values) varies.
2. **The generic machinery needs the type at class level** for
   `_build_packer`, which inspects field names. Storing it on the class
   makes it available without instantiation.
3. **Subclasses just write `Physics = MyPhysicsType`** at the class body
   level. This is the simplest possible way to declare it.

### Why the constructor takes `physics` (an instance), not individual parameters

The current `DroguedDrifter.__init__` takes 9 individual physics
parameters (`m_b=`, `m_d=`, ...). This was convenient for interactive use
but does not generalize: each model has different parameters, so the base
class constructor cannot enumerate them.

Instead, the base class constructor takes a `Physics` NamedTuple instance.
Defaults come from `default_physics()`, so `DroguedDrifter()` still works
with no arguments.

For backward compatibility during the transition, `DroguedDrifter` could
accept kwargs and construct `DrifterPhysics` from them:

```python
class DroguedDrifter(LagrangianMechanicsModel):

    def __init__(self, physics=None, *, backend="numpy", **kwargs):
        if physics is None and kwargs:
            defaults = self.default_physics()._asdict()
            defaults.update(kwargs)
            physics = DrifterPhysics(**defaults)
        super().__init__(physics, backend=backend)
```

This preserves `DroguedDrifter(l=5.0)` for notebooks while moving toward
the generic constructor.

### Why two levels of inheritance max

The design has exactly one level: `LagrangianMechanicsModel` (abstract
base) -> `DroguedDrifter` / `SparBuoy` (concrete). No intermediate
abstract classes, no mixins, no diamond inheritance.

If two models share structure (e.g., both have a buoy + submerged body),
the shared code should be a utility function imported by both, not a
shared intermediate class. Function reuse over inheritance reuse. This
keeps the class hierarchy flat and predictable.

### Comparison with C's ModelSpec

| Concern | C (ModelSpec) | D (Base class) |
|---|---|---|
| Extension mechanism | Construct a dataclass with 12 fields | Subclass and override 5 methods |
| Discoverability | Read the dataclass definition | `help(LagrangianMechanicsModel)` |
| IDE support | No override completion | Override completion works |
| Default implementations | Not possible (all fields required) | `_pack_derivatives` has a default |
| Caching key | `spec.name` (string) | `type(model).__name__` (class name) |
| Validation | None (duck typing) | ABC enforcement on abstract methods |
| Target audience | Experienced developers | Students who know classes |
| Number of concepts | NamedTuple, dataclass, callable fields | Class, inheritance, abstract method |
| Information content | Identical | Identical |

The base class carries exactly the same information as the ModelSpec. The
difference is entirely in how that information is expressed: as method
overrides instead of callable fields.

## Velocity protocol

Unchanged from proposal B. Velocity is passed as arrays
`(depth_levels, U_profiles, V_profiles)` to the solver functions. The
`sample_uv` closure protocol is eliminated from the internal architecture.
`make_profile_sampler` becomes `interpolate_profile` in `velocity.py`
(or stays as `make_profile_sampler` if the closure form is preferred for
Parcels -- this is an implementation detail, not an architecture decision).

For backward compatibility, the `DroguedDrifter` convenience methods can
still accept a `sample_uv` callable, but this is a DroguedDrifter-specific
convenience, not part of the base class contract.

## Public API (`__init__.py`)

```python
# Backward-compatible re-exports (DroguedDrifter is the default model)
from .models.drogued_drifter import (
    DroguedDrifter,
    DrifterPhysics,
    EOMState,
    drogue_horizontal_added_mass,
    buoy_horizontal_added_mass,
    drogue_horizontal_drag_coeff,
    buoy_horizontal_drag_coeff,
)
from .base import LagrangianMechanicsModel
from .eom import eval_qdd, eval_M, eval_F
from .coords import uv_to_spherical, spherical_to_uv
from .velocity import interpolate_profile
from .stokes import compute_stokes_profile
from .parcels import make_kernel

# Backward-compatible alias
make_dd_kernel = make_kernel
```

## Migration path

### Phase 1: Module restructuring (mechanical)

1. Create `base.py` with `LagrangianMechanicsModel`.
2. Create `coords.py` (move coordinate transforms from `lagrange_model.py`).
3. Create `velocity.py` (move `make_profile_sampler` from `parcels_v4.py`).
4. Create `eom.py` (extract generic machinery from `lagrange_model.py`,
   parameterize on model instance).
5. Create `models/drogued_drifter.py` (move model-specific code from
   `lagrange_model.py` and `drifter.py`, make it a subclass).
6. Update all imports. Run tests.

### Phase 2: Generic solver (substantive)

7. Create `solve.py` with generic `_rhs_batch`, `steady_state_drift`,
   `integrate`.
8. Wire `DroguedDrifter` convenience methods to delegate to `solve.py`.
9. Update tests.

### Phase 3: Generic Parcels coupling

10. Create `parcels.py` with `make_kernel`, `extract_profiles`,
    `position_update`.
11. `make_dd_kernel` becomes a backward-compatible alias.
12. Update Parcels tests.

### Phase 4: Cleanup

13. Delete `lagrange_model.py`, `drifter.py`, `parcels_v4.py`.
14. Update `__init__.py`.
15. Update example notebooks.

### Phase 5: SparBuoy (when ready)

16. Create `models/spar_buoy.py`.
17. Test with the generic machinery.

Phases 1 and 2 are the substantive work. Each phase is independently
committable and testable. The existing test suite stays valid throughout
(only import paths change).

## What does NOT change

- **Symbolic derivation.** `_derive_symbolic()` body is identical, just
  moved to the model module.
- **Pickle cache format.** Same keys, same invalidation logic.
- **`_build_packer` logic.** Same signature-inspection approach, just
  parameterized on types.
- **ODE solver.** Still `scipy.integrate.solve_ivp`.
- **Backend handling.** Still `_make_qdd_func(backend)` with optional
  numba.
- **Stokes drift.** `stokes.py` unchanged.
- **Coordinate convention.** z-up, stereographic internal, spherical
  public.
- **Test coverage.** Existing physics tests apply to the moved code
  without modification.

## Trade-offs

### ABC enforcement vs. duck typing

Using `ABC` and `@abstractmethod` means Python raises `TypeError` if you
try to instantiate a subclass without implementing all abstract methods.
This is useful for developers -- they get a clear error message instead of a
cryptic `AttributeError` deep in the solver.

The downside is importing `abc`, which is a minor conceptual addition. But
`ABC` is in the standard library, and "class must implement these methods"
is a universally understood concept. The alternative (no ABC, just
document what to override) risks silent failures.

### `@staticmethod` vs. regular methods

Making the abstract methods static means they cannot access `self`, which
prevents them from accidentally depending on instance state. This is good
for purity but slightly unusual for developers who expect methods to use
`self`.

An alternative is regular methods that happen not to use `self`. Linters
would flag this. The `@staticmethod` makes the intent explicit: "this is a
function that lives on the class for organizational purposes."

If this proves confusing for developers, the methods can be changed to
regular methods with a `self` parameter that is simply unused. The generic
machinery calls them via the instance either way.

### One model per simulation run

This architecture assumes only one model class is active at a time. There
is no mechanism for mixed-object particle sets (e.g., some particles are
DroguedDrifters and others are SparBuoys). This is a deliberate constraint
from the project requirements. Supporting mixed objects would require
per-particle dispatch in `_rhs_batch`, which is a fundamentally different
architecture.

### Convenience methods on subclass vs. on base class

The sketch puts `full_trajectory` and `steady_state` on `DroguedDrifter`
rather than on `LagrangianMechanicsModel`. This means each model writes
its own convenience methods, which could lead to inconsistency.

An alternative is putting generic versions on the base class:

```python
class LagrangianMechanicsModel(ABC):
    def steady_state(self, depth_levels, U_profiles, V_profiles, **kw):
        from .solve import steady_state_drift
        return steady_state_drift(self, self.physics, depth_levels,
                                  U_profiles, V_profiles,
                                  backend=self.backend, **kw)
```

This would give every model the same convenience API for free. The
model-specific part (like converting stereographic to spherical for the
`full_trajectory` Dataset) would still live on the subclass.

This is a reasonable middle ground: put `steady_state` on the base class
(since its return format is model-independent: drift velocity arrays), but
leave `full_trajectory` on the subclass (since the xr.Dataset format is
model-specific). **This decision can be deferred to implementation time.**
