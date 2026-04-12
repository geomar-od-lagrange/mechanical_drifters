# Architecture v5: class-based multi-object generalization

Extends the current single-model codebase to support multiple types of
drifting objects, each derived via Lagrangian mechanics. The motivating
use case is adding a SparBuoy alongside the existing DroguedDrifter, but
the design handles any M*qdd = F model.

**Constraint:** only one object class is simulated at a time. No
mixed-object particle sets.

## Core observation

The current machinery does most of the work generically:

1. `_build_packer` inspects a lambdified function's signature and maps
   parameter names to NamedTuple fields by name. It hardcodes
   `DrifterPhysics` and `EOMState`, but generalizing it to accept any
   pair of NamedTuples is a one-parameter change.

2. `_make_qdd_func` wraps a raw lambdified callable into a
   `qdd_func(physics, state)` evaluator. The wrapping logic is the same
   for any model. Only the raw callable and the packer differ.

3. `_rhs_batch` is deeply object-specific. It knows the state vector
   layout, how to compute drogue depth from generalized coordinates, and
   how to query velocity at specific depths. A SparBuoy would need a
   completely different `_rhs_batch`.

The design splits along this seam: **generic EOM machinery** (caching,
lambdification, packing, evaluation) lives in shared code;
**object-specific knowledge** (symbolic derivation, state layout, depth
queries, RHS assembly) lives in per-model subclasses of a base class.

## The base class

A plain Python class (no ABC, no metaclass). Subclass it, set four class
attributes, override four methods.

**Why not ABC?** A plain class with `__init__`-time validation gives
equally clear errors (`TypeError: SparBuoy must set class attribute
'n_q'`) without the `abc` import, without `@abstractmethod` decorator
stacking, and without forcing decisions about `@staticmethod` vs regular
methods. With 2-3 models in one repo, convention beats mechanism.

**Why regular methods, not `@staticmethod`?** The model-specific methods
(`_rhs_batch`, `_derive_symbolic`, etc.) don't use `self` in the current
DroguedDrifter, but a future model might need instance state in one of
them. Regular methods are familiar, need no decorator, and keep the door
open. The cost is a linter warning about unused `self` that can be
suppressed once.

```python
# base.py

import re
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


class LagrangianMechanicsModel:
    """Base class for drifting-object models derived via Lagrangian mechanics.

    Subclass this to define a new drifting object. You must provide:

    1. Class attributes: ``Physics``, ``State``, ``n_q``,
       ``_drift_velocity_indices``.
    2. Methods: ``default_physics``, ``_derive_symbolic``, ``_rhs_batch``,
       ``_max_depth``.

    The base class provides ODE integration (``steady_state_batch``)
    and a Parcels kernel factory (``make_kernel``).

    See ``DroguedDrifter`` for a complete example.
    """

    # --- Class attributes (override in subclass) ---

    Physics = None   # NamedTuple class for physical constants
    State = None     # NamedTuple class for per-timestep state + forcing
    n_q = None       # number of generalized coordinates

    # Indices into the state vector [q0, ..., q_{n-1}, qdot0, ..., qdot_{n-1}]
    # that are the "drift velocity" output.
    # DroguedDrifter: (4, 5) = (xd, yd).
    _drift_velocity_indices = None

    # --- Constructor ---

    def __init__(self, physics=None, *, backend="numpy"):
        # Validate that the subclass set the required class attributes.
        for attr in ("Physics", "State", "n_q", "_drift_velocity_indices"):
            if getattr(type(self), attr, None) is None:
                raise TypeError(
                    f"{type(self).__name__} must set class attribute {attr!r}"
                )

        if physics is None:
            physics = self.default_physics()
        self.physics = physics
        self.backend = backend

        from .eom import _make_qdd_func

        self._qdd_func = _make_qdd_func(self, backend)

    # --- Properties ---

    @property
    def state_size(self):
        """Total state vector length (2 * n_q for standard layouts)."""
        return 2 * self.n_q

    @property
    def _cache_path(self):
        """Path to the pickled symbolic derivation cache.

        Auto-derived from the class name:
        DroguedDrifter -> data/eom_cache_drogued_drifter.pkl
        SparBuoy       -> data/eom_cache_spar_buoy.pkl

        Override on the subclass if a different path is needed.
        """
        name = type(self).__name__
        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
        return Path(__file__).resolve().parent / "data" / f"eom_cache_{snake}.pkl"

    # --- Methods to override ---

    def default_physics(self):
        """Return the default Physics instance for this model."""
        raise NotImplementedError

    def _derive_symbolic(self):
        """Derive symbolic M and F from the Lagrangian.

        Use sympy to:
        1. Define generalized coordinates and physical parameters.
        2. Write down T, V, and the Lagrangian L = T - V.
        3. Compute generalized forces Q from non-conservative forces.
        4. Apply the Euler-Lagrange equations.
        5. Extract M and F such that M * qdd = F.

        Symbol names in the returned ``args`` tuple must exactly match
        field names in ``self.Physics`` and ``self.State``.

        Returns:
            Tuple ``(M_static, F_static, args)``.
        """
        raise NotImplementedError

    def _rhs_batch(self, Y, sample_uv):
        """Compute dY/dt for N particles.

        This is the complete right-hand side of the ODE. It queries
        velocities via ``sample_uv``, builds the State NamedTuple,
        evaluates accelerations via ``self._qdd_func``, guards NaN, and
        packs derivatives. Everything in one method, no sub-dispatch.

        Args:
            Y: State array, shape ``(N, state_size)``.
            sample_uv: Callable ``sample_uv(z) -> (U, V)`` where z is
                ``(N,)`` and U, V are ``(N,)`` arrays in m/s.

        Returns:
            dY: Derivative array, shape ``(N, state_size)``.
        """
        raise NotImplementedError

    def _max_depth(self, physics):
        """Maximum depth [m, positive] to sample from the fieldset.

        The Parcels coupling uses this to decide how many depth levels to
        extract from the ocean model data.

        Args:
            physics: Physics NamedTuple.

        Returns:
            float, positive depth in meters.
        """
        raise NotImplementedError

    # --- Provided by the base class ---

    def steady_state_batch(
        self,
        sample_uv,
        *,
        t_span=(0, 120),
        y0=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Compute steady-state drift velocities for N particles.

        Stacks N particles into a single ``(state_size*N,)`` ODE system
        and integrates to steady state.

        Args:
            sample_uv: Velocity sampler ``sample_uv(z) -> (U, V)``.
                Returns ``(N,)`` arrays for ``(N,)`` input.
            t_span: Integration window ``(t_start, t_end)`` in seconds.
            y0: Initial internal state, shape ``(N, state_size)``.
                If None, cold-start from zeros (equilibrium at rest).
            atol, rtol: ODE solver tolerances.

        Returns:
            Tuple ``(drift_vel, Y_final, max_accel)`` where:
            - drift_vel: ``(N, len(_drift_velocity_indices))``
            - Y_final: ``(N, state_size)`` internal state (for warm-start)
            - max_accel: scalar convergence diagnostic
        """
        ss = self.state_size

        if y0 is not None:
            y0_arr = np.asarray(y0, dtype=float).reshape(-1, ss)
            N = y0_arr.shape[0]
            y0_flat = y0_arr.ravel()
        else:
            # Determine N by probing the sampler
            probe = sample_uv(np.array([0.0]))
            N = len(probe[0])
            y0_flat = np.zeros(N * ss)

        def rhs_flat(t, y_flat):
            Y = y_flat.reshape(N, ss)
            dY = self._rhs_batch(Y, sample_uv)
            return dY.ravel()

        sol = solve_ivp(rhs_flat, t_span, y0_flat, atol=atol, rtol=rtol)
        Y_final = sol.y[:, -1].reshape(N, ss)

        # Convergence diagnostic: max drift acceleration at final state
        dY_final = self._rhs_batch(Y_final, sample_uv)
        idx = list(self._drift_velocity_indices)
        max_accel = float(np.max(np.abs(dY_final[:, idx])))

        drift_vel = Y_final[:, idx]
        return drift_vel, Y_final, max_accel

    def make_kernel(self):
        """Create a Parcels-compatible kernel for this model.

        Returns a ``(particles, fieldset)`` function suitable for
        ``pset.execute(kernels=[...])``.  Uses ``self.physics`` and
        ``self.backend``.
        """
        from .parcels import make_kernel

        return make_kernel(self)
```

### What the base class provides vs. requires

**Requires from each subclass** (4 class attributes + 4 methods):

| Piece | Kind | Purpose |
|---|---|---|
| `Physics` | class attribute | NamedTuple class for physical constants |
| `State` | class attribute | NamedTuple class for per-timestep state + forcing |
| `n_q` | class attribute | Number of generalized coordinates |
| `_drift_velocity_indices` | class attribute | Which state components are the drift velocity |
| `default_physics()` | method | Sensible default Physics instance |
| `_derive_symbolic()` | method | Sympy derivation → `(M_static, F_static, args)` |
| `_rhs_batch(Y, sample_uv)` | method | Complete vectorized RHS, ~20 lines |
| `_max_depth(physics)` | method | Deepest depth to sample for Parcels |

**Provides for free:**

| Piece | Purpose |
|---|---|
| `__init__(physics, backend)` | Validate attributes, store physics, build qdd evaluator |
| `state_size` | Property: `2 * n_q` |
| `_cache_path` | Property: auto-derived from class name |
| `steady_state_batch(sample_uv, ...)` | Batch ODE integration to steady state |
| `make_kernel()` | Parcels kernel factory |

Each required method answers a direct question about the model's physics.
None exist solely to serve a generic decomposition.

### What is NOT in the base class

The base class is the contract plus the solver. Everything else lives in
separate modules:

- Symbolic caching, lambdification, packing → `eom.py`
- Velocity profile interpolation → `velocity.py`
- Parcels profile extraction and position update → `parcels.py`
- Coordinate transforms → `coords.py`
- Stokes drift → `stokes.py`
- Model-specific output formatting → on the subclass
- Single-particle scalar ODE path → on the subclass

## Why `_rhs_batch` is the single override, not a decomposition

The right-hand side of the ODE has a common structure across models:

1. Query depths from the generalized coordinates.
2. Interpolate velocity at those depths.
3. Assemble the State NamedTuple.
4. Evaluate generalized accelerations.
5. Guard NaN/inf.
6. Pack derivatives.

An alternative design decomposes this into three abstract methods
(`z_query`, `build_state`, `pack_derivatives`) and provides a generic
`_rhs_batch` in the solver. That reduces repetition but adds three levels
of indirection in the hot path and three more methods to the contract.

V5 instead makes `_rhs_batch` itself the model-specific method. Each
model writes ~20 lines of straight-line code covering all six steps.
The duplicated boilerplate across models:

- NaN guard: 3 lines, identical
- `sample_uv(z)` call pattern: ~2 lines per body
- Derivative packing: ~3 lines, same structure

Total: ~10 lines of structural repetition per model. With 2-3 models,
this is not worth abstracting. The benefit is that each model's RHS is
completely readable top-to-bottom with zero indirection.

## Generic `eom.py`

The current `eom.py` contains both generic machinery (caching,
lambdification, packing, evaluation) and DroguedDrifter-specific code
(`DrifterPhysics`, `EOMState`, `_derive_symbolic`). V5 keeps only the
generic part, parameterized on the model instance.

### `_build_packer` — parameterized on types

The current `_build_packer(raw_func)` hardcodes `DrifterPhysics._fields`
and `EOMState._fields`. V5 takes the types as arguments:

```python
def _build_packer(raw_func, physics_type, state_type):
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
```

One parameter added. Body unchanged.

### `_load_or_derive` — parameterized on model

```python
def _load_or_derive(model):
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
```

### `_cache_key` — hashes the passed function

```python
def _cache_key(derive_fn):
    source = inspect.getsource(derive_fn)
    key_data = source + sp.__version__ + str(sys.version_info[:2])
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]
```

### Caching strategy

Dict keyed by model class name. All instances of the same model class
share compiled callables. No `lru_cache` (model instances aren't
hashable), no weak refs (2-3 models).

```python
_CALLABLE_CACHE = {}  # class_name -> (qdd_raw, M_raw, F_raw, pack_eom_args)

def _get_eom_callables(model):
    key = type(model).__name__
    if key not in _CALLABLE_CACHE:
        M_static, F_static, qdd_exprs, args = _load_or_derive(model)
        m_exprs = tuple(M_static[i, j]
                        for i in range(model.n_q) for j in range(i, model.n_q))
        f_exprs = tuple(F_static[i] for i in range(model.n_q))

        qdd_raw = sp.lambdify(args, qdd_exprs, modules="numpy", cse=True)
        M_raw = sp.lambdify(args, m_exprs, modules="numpy", cse=True)
        F_raw = sp.lambdify(args, f_exprs, modules="numpy", cse=True)

        pack_eom_args = _build_packer(qdd_raw, model.Physics, model.State)
        _CALLABLE_CACHE[key] = (qdd_raw, M_raw, F_raw, pack_eom_args)

    return _CALLABLE_CACHE[key]


_QDD_CACHE = {}  # (class_name, backend) -> qdd_func

def _make_qdd_func(model, backend="numpy"):
    key = (type(model).__name__, backend)
    if key not in _QDD_CACHE:
        qdd_raw, _, _, pack_eom_args = _get_eom_callables(model)

        if backend == "numpy":
            raw = qdd_raw
        elif backend == "numba":
            from numba import njit
            raw = njit(qdd_raw)
            _n_args = len(model.Physics._fields) + len(model.State._fields)
            _dummy_args = tuple(
                np.ones(1) if i >= len(model.Physics._fields) else 1.0
                for i in range(_n_args)
            )
            raw(*_dummy_args)
        else:
            raise ValueError(f"Unknown backend {backend!r}")

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
```

### Public evaluation functions

```python
def eval_qdd(model, physics, state, *, backend="numpy"):
    """Evaluate generalized accelerations qdd = M^{-1}F."""
    return _make_qdd_func(model, backend)(physics, state)

def eval_M(model, physics, state):
    """Evaluate the mass matrix M."""
    _, M_raw, _, pack_eom_args = _get_eom_callables(model)
    # ... same assembly logic as current M_func, using model.n_q ...

def eval_F(model, physics, state):
    """Evaluate the force vector F."""
    _, _, F_raw, pack_eom_args = _get_eom_callables(model)
    # ... same assembly logic as current F_func, using model.n_q ...
```

### What moves out of `eom.py`

- `DrifterPhysics` → `models/drogued_drifter.py`
- `EOMState` → `models/drogued_drifter.py`
- `_derive_symbolic()` → `DroguedDrifter._derive_symbolic()`
- `_sym_norm()` → moves with `_derive_symbolic`
- `_CACHE_PATH` → replaced by `model._cache_path` property
- Backward-compat coord re-exports → removed

### What stays in `eom.py`

- `_build_packer` (parameterized on types)
- `_cache_key` (parameterized on function)
- `_load_or_derive` (parameterized on model)
- `_get_eom_callables` (parameterized on model)
- `_make_qdd_func` (parameterized on model + backend)
- `eval_qdd`, `eval_M`, `eval_F` (parameterized on model)

## `DroguedDrifter` as a subclass

Complete implementation. The `_derive_symbolic` body is the same as the
current `eom._derive_symbolic` — moved, not changed.

```python
# models/drogued_drifter.py

from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel
from ..coords import _spherical_to_uv, _uv_to_spherical


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
    """Horizontal added mass of the drogue: m_tilde_d = C_perp_d * rho * w_d^2 * h_d."""
    return C_perp_d * rho * w_d**2 * h_d

def buoy_horizontal_added_mass(*, rho, d_b, h_b, C_perp_b=1.0):
    """Horizontal added mass of the buoy: m_tilde_b = C_perp_b * rho * pi/4 * d_b^2 * h_b."""
    return C_perp_b * rho * np.pi / 4 * d_b**2 * h_b

def drogue_horizontal_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2):
    """Horizontal drag coefficient of the drogue: k_d = 0.5 * rho * C_D_d * w_d * h_d."""
    return 0.5 * rho * C_D_d * w_d * h_d

def buoy_horizontal_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0):
    """Horizontal drag coefficient of the buoy: k_b = 0.5 * rho * C_D_b * d_b * h_b."""
    return 0.5 * rho * C_D_b * d_b * h_b


# ---------- Internal state vector layout ----------
# [x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo]
IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)


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

    def __init__(self, physics=None, *, backend="numpy", **kwargs):
        """Create a DroguedDrifter.

        Accepts either a DrifterPhysics instance or individual keyword
        parameters::

            DroguedDrifter()                              # defaults
            DroguedDrifter(my_physics)                    # Physics instance
            DroguedDrifter(l=5.0, k_d=200.0)             # override defaults
            DroguedDrifter(backend="numba")               # numba backend
        """
        if physics is None and kwargs:
            defaults = self.default_physics()._asdict()
            defaults.update(kwargs)
            physics = DrifterPhysics(**defaults)
        super().__init__(physics, backend=backend)

    def default_physics(self):
        """Callies et al. (2017) drifter at rho=1025 kg/m^3."""
        return DrifterPhysics(
            m_b=1.0, m_d=2.7, m_hat_d=1.0,
            m_tilde_d=101.0, m_tilde_b=1.9,
            l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        )

    # --- Symbolic derivation (the physics) ---

    def _derive_symbolic(self):
        """Derive M, F for the buoy-pole-drogue system in stereographic coords.

        The pole direction is parameterized via stereographic projection
        from the south pole, avoiding the azimuthal singularity at
        theta=pi (drogue hanging straight down).

        Returns:
            (M_static, F_static, args) where M is 4x4, F is 4x1, and args
            is the ordered symbol tuple matching Physics + State field names.
        """
        # This is the current eom._derive_symbolic body, moved here unchanged.

        def _sym_norm(vec):
            return sp.sqrt(vec.dot(vec))

        t = dynamicsymbols._t

        x, y = dynamicsymbols("x y")
        u_st, v_st = dynamicsymbols("u_st v_st")

        m_b, m_d, l, g = sp.symbols("m_b m_d l g", positive=True)
        m_hat_d = sp.Symbol("m_hat_d", positive=True)
        m_tilde_d = sp.Symbol("m_tilde_d", positive=True)
        m_tilde_b = sp.Symbol("m_tilde_b", positive=True)
        k_b, k_d = sp.symbols("k_b k_d", positive=True)
        U_b, V_b, U_d, V_d = sp.symbols("U_b V_b U_d V_d", real=True)

        s = u_st**2 + v_st**2
        denom = s + 4
        sin_theta_cos_phi = 4 * u_st / denom
        sin_theta_sin_phi = 4 * v_st / denom
        cos_theta = (s - 4) / denom

        r_b = sp.Matrix([x, y, 0])
        r = l * sp.Matrix([sin_theta_cos_phi, sin_theta_sin_phi, cos_theta])
        r_d = r_b + r

        v_b = r_b.diff(t)
        v_d = r_d.diff(t)
        v_d_h = sp.Matrix([v_d[0], v_d[1], 0])

        u_b_vec = sp.Matrix([U_b, V_b, 0])
        u_d_vec = sp.Matrix([U_d, V_d, 0])

        F_b = -k_b * _sym_norm(v_b - u_b_vec) * (v_b - u_b_vec)
        F_d = -k_d * _sym_norm(v_d_h - u_d_vec) * (v_d_h - u_d_vec)

        T = (
            sp.Rational(1, 2) * m_d * v_d.dot(v_d)
            + sp.Rational(1, 2) * m_tilde_d * v_d_h.dot(v_d_h)
            + sp.Rational(1, 2) * (m_b + m_tilde_b) * v_b.dot(v_b)
        )
        V = (m_d - m_hat_d) * g * r_d[2]
        L = T - V

        q = sp.Matrix([x, y, u_st, v_st])
        qd = q.diff(t)
        qdd = qd.diff(t)

        Q = sp.Matrix([r_b.diff(qi).dot(F_b) + r_d.diff(qi).dot(F_d) for qi in q])
        Q = sp.simplify(Q)

        eoms = sp.Matrix(
            [L.diff(qdj).diff(t) - L.diff(qj) - Qj
             for qj, qdj, Qj in zip(q, qd, Q)]
        )
        eoms = sp.simplify(eoms)
        M, F = sp.simplify(sp.linear_eq_to_matrix(eoms, list(qdd)))

        xd_dyn, yd_dyn = x.diff(t), y.diff(t)
        ud_dyn, vd_dyn = u_st.diff(t), v_st.diff(t)

        u_static, v_static = sp.symbols("u_stereo v_stereo", real=True)
        ud_static, vd_static = sp.symbols("ud_stereo vd_stereo", real=True)
        x_static, y_static = sp.symbols("x_pos y_pos", real=True)
        xd_static, yd_static = sp.symbols("xd yd", real=True)

        subs = {
            x: x_static, y: y_static,
            u_st: u_static, v_st: v_static,
            xd_dyn: xd_static, yd_dyn: yd_static,
            ud_dyn: ud_static, vd_dyn: vd_static,
        }
        M_static = M.subs(subs)
        F_static = F.subs(subs)

        symbol_map = {
            "m_b": m_b, "m_d": m_d, "m_hat_d": m_hat_d,
            "m_tilde_d": m_tilde_d, "m_tilde_b": m_tilde_b,
            "l": l, "g": g, "k_b": k_b, "k_d": k_d,
            "u_stereo": u_static, "v_stereo": v_static,
            "xd": xd_static, "yd": yd_static,
            "ud_stereo": ud_static, "vd_stereo": vd_static,
            "U_b": U_b, "V_b": V_b, "U_d": U_d, "V_d": V_d,
        }
        all_fields = list(DrifterPhysics._fields) + list(EOMState._fields)
        args = tuple(symbol_map[field] for field in all_fields)

        return M_static, F_static, args

    # --- The RHS (the hot path) ---

    def _z_eff(self, u, v):
        """Effective drogue depth from stereographic coordinates."""
        s = u**2 + v**2
        cos_theta = (s - 4) / (s + 4)
        return np.minimum(0.0, self.physics.l * cos_theta)

    def _rhs_batch(self, Y, sample_uv):
        """Vectorized RHS for N particles.

        Args:
            Y: (N, 8) state array.
            sample_uv: callable(z) -> (U, V) for (N,) arrays.

        Returns:
            dY: (N, 8) derivatives.
        """
        N = Y.shape[0]
        u_stereo = Y[:, IU]
        v_stereo = Y[:, IV]
        xd = Y[:, IXD]
        yd = Y[:, IYD]
        ud_stereo = Y[:, IUD]
        vd_stereo = Y[:, IVD]

        # Velocity at buoy (surface) and drogue (effective depth)
        U_b, V_b = sample_uv(np.zeros(N))
        U_d, V_d = sample_uv(self._z_eff(u_stereo, v_stereo))

        state = EOMState(
            u_stereo=u_stereo, v_stereo=v_stereo,
            xd=xd, yd=yd,
            ud_stereo=ud_stereo, vd_stereo=vd_stereo,
            U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
        )

        qdd = self._qdd_func(self.physics, state)

        bad = ~np.isfinite(qdd).all(axis=1)
        if np.any(bad):
            qdd[bad] = 0.0

        dY = np.empty_like(Y)
        dY[:, IX] = xd
        dY[:, IY] = yd
        dY[:, IU] = ud_stereo
        dY[:, IV] = vd_stereo
        dY[:, IXD:] = qdd
        return dY

    def _max_depth(self, physics):
        """Drogue hangs at most one pole-length below surface."""
        return physics.l

    # --- Scalar ODE path (DroguedDrifter-specific) ---

    def _rhs(self, t, y, sample_uv):
        """Scalar RHS for single-particle integration."""
        u_stereo, v_stereo = y[IU], y[IV]
        z_d = float(self._z_eff(np.array([u_stereo]), np.array([v_stereo]))[0])

        U_b, V_b = sample_uv(0.0)
        U_d, V_d = sample_uv(z_d)

        state = EOMState(
            u_stereo, v_stereo, y[IXD], y[IYD], y[IUD], y[IVD],
            U_b, V_b, U_d, V_d,
        )
        qdd = self._qdd_func(self.physics, state)
        return np.array([y[IXD], y[IYD], y[IUD], y[IVD], *qdd])

    # --- DroguedDrifter-specific convenience methods ---

    def get_final_drift_batch(
        self,
        sample_uv,
        *,
        t_span=(0, 120),
        y0=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Compute steady-state drift for N particles.

        Wraps ``steady_state_batch`` with internal-to-public coordinate
        conversion. The public state uses spherical angles (theta, phi)
        instead of internal stereographic (u_stereo, v_stereo).

        Args:
            sample_uv: Velocity sampler ``sample_uv(z) -> (U, V)``.
            t_span: Integration window [s].
            y0: Initial state ``(N, 8)`` in public format
                ``[x, y, theta, phi, xd, yd, thetad, phid]``, or None.
            atol, rtol: ODE solver tolerances.

        Returns:
            ``(xd, yd, Y_public, max_accel)`` where Y_public columns are
            ``[x, y, theta, phi, xd, yd, thetad, phid]``.
        """
        # Convert public y0 (spherical) to internal (stereographic)
        y0_internal = None
        if y0 is not None:
            y0_arr = np.asarray(y0, dtype=float).reshape(-1, 8)
            u0, v0, ud0, vd0 = _spherical_to_uv(
                y0_arr[:, 2], y0_arr[:, 3], y0_arr[:, 6], y0_arr[:, 7],
            )
            y0_internal = np.column_stack([
                y0_arr[:, 0], y0_arr[:, 1],
                u0, v0,
                y0_arr[:, 4], y0_arr[:, 5],
                ud0, vd0,
            ])

        drift_vel, Y_final, max_accel = self.steady_state_batch(
            sample_uv, t_span=t_span, y0=y0_internal, atol=atol, rtol=rtol,
        )

        # Convert internal state to public (spherical) coordinates
        theta, phi, thetad, phid = _uv_to_spherical(
            Y_final[:, IU], Y_final[:, IV],
            Y_final[:, IUD], Y_final[:, IVD],
        )
        Y_public = np.column_stack([
            Y_final[:, IX], Y_final[:, IY],
            theta, phi,
            Y_final[:, IXD], Y_final[:, IYD],
            thetad, phid,
        ])

        return drift_vel[:, 0], drift_vel[:, 1], Y_public, max_accel

    def get_full_solution(
        self,
        sample_uv,
        *,
        t_span,
        x=0.0, y=0.0, theta=np.pi, phi=0.0,
        xd=0.0, yd=0.0, thetad=0.0, phid=0.0,
        t_eval=None, atol=1e-3, rtol=1e-3,
    ):
        """Integrate single-particle trajectory, return xr.Dataset.

        Returns a Dataset with spherical coordinates (theta, phi)
        converted from the internal stereographic representation.
        """
        import xarray as xr
        from scipy.integrate import solve_ivp

        u0, v0, ud0, vd0 = _spherical_to_uv(theta, phi, thetad, phid)
        y0 = [x, y, u0, v0, xd, yd, ud0, vd0]

        sol = solve_ivp(
            lambda t, y_: self._rhs(t, y_, sample_uv),
            t_span, y0, t_eval=t_eval, atol=atol, rtol=rtol,
        )

        theta_arr, phi_arr, thetad_arr, phid_arr = _uv_to_spherical(
            sol.y[IU], sol.y[IV], sol.y[IUD], sol.y[IVD],
        )
        return xr.Dataset(
            {
                "x": ("time", sol.y[IX]),
                "y": ("time", sol.y[IY]),
                "theta": ("time", theta_arr),
                "phi": ("time", phi_arr),
                "xd": ("time", sol.y[IXD]),
                "yd": ("time", sol.y[IYD]),
                "thetad": ("time", thetad_arr),
                "phid": ("time", phid_arr),
            },
            coords={"time": sol.t},
        )

    def get_final_drift(
        self,
        sample_uv,
        *,
        t_span,
        x=0.0, y=0.0, theta=np.pi, phi=0.0,
        xd=0.0, yd=0.0, thetad=0.0, phid=0.0,
    ):
        """Scalar single-particle steady-state drift.

        Returns:
            ``(xd_final, yd_final, max_accel)``
        """
        from scipy.integrate import solve_ivp

        u0, v0, ud0, vd0 = _spherical_to_uv(theta, phi, thetad, phid)
        y0 = [x, y, u0, v0, xd, yd, ud0, vd0]

        sol = solve_ivp(
            lambda t, y_: self._rhs(t, y_, sample_uv),
            t_span, y0,
        )
        y_final = sol.y[:, -1]

        dy_final = self._rhs(0.0, y_final, sample_uv)
        max_accel = float(max(abs(dy_final[IXD]), abs(dy_final[IYD])))

        return float(y_final[IXD]), float(y_final[IYD]), max_accel
```

### Design note: `sample_uv` is a parameter, not stored on `self`

The current `DroguedDrifter` stores `self._sample_uv` at construction
time and has a save/restore hack in `get_final_drift_batch` for the
Parcels path (which rebuilds the sampler each timestep). V5 passes
`sample_uv` to each solve call instead. This eliminates the mutable
state, eliminates the hack, and makes the velocity source explicit at
every call site.

## Generic `parcels.py`

The current `parcels_v4.py` is parameterized on a `DroguedDrifter`
instance (`dd`) but only uses `dd.physics.l` for depth computation and
`dd.get_final_drift_batch(sample_uv=...)` for the solve. V5 replaces
both with model-agnostic access:

- `dd.physics.l` → `model._max_depth(model.physics)`
- `dd.get_final_drift_batch(sample_uv=...)` → `model.steady_state_batch(sample_uv)`

```python
# parcels.py

import numpy as np
from .velocity import make_profile_sampler

_DEG2M = 1852.0 * 60.0


def _extract_profiles(particles, fieldset, max_depth):
    """Extract velocity profiles and build a depth interpolator.

    Samples the fieldset at each depth level from the surface down to
    ``max_depth``.  Converts Parcels' degree-based velocities to m/s
    on spherical grids.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet with a ``UV`` VectorField.
        max_depth: Maximum depth [m, positive] to sample.

    Returns:
        Callable ``sample_uv(z) -> (U, V)``.
    """
    lat = np.asarray(particles.lat)
    lon = np.asarray(particles.lon)
    time = particles.time
    N = len(lat)

    is_spherical = fieldset.U.grid._mesh == "spherical"

    all_depths = np.asarray(fieldset.U.grid.depth, dtype=float)
    cutoff = min(
        np.searchsorted(all_depths, max_depth, side="right") + 1,
        len(all_depths),
    )
    depth_levels = all_depths[: max(cutoff, 2)]
    D = len(depth_levels)

    U_profiles = np.empty((D, N))
    V_profiles = np.empty((D, N))
    for iz, z_level in enumerate(depth_levels):
        z_arr = np.full(N, z_level)
        u, v = fieldset.UV.eval(time, z_arr, lat, lon)[:2]
        if is_spherical:
            cos_lat = np.cos(np.deg2rad(lat))
            u = u * _DEG2M * cos_lat
            v = v * _DEG2M
        U_profiles[iz] = u
        V_profiles[iz] = v

    depth_up = -depth_levels[::-1]
    U_profiles = U_profiles[::-1]
    V_profiles = V_profiles[::-1]

    return make_profile_sampler(depth_up, U_profiles, V_profiles)


def _position_update(particles, xd_ms, yd_ms, fieldset):
    """Euler-forward position update.

    Converts drift velocities (m/s) to degree displacements on spherical
    grids, or applies them directly on flat grids.
    """
    is_spherical = fieldset.U.grid._mesh == "spherical"
    if is_spherical:
        lat = np.asarray(particles.lat)
        cos_lat = np.cos(np.deg2rad(lat))
        particles.dlon += xd_ms / (_DEG2M * cos_lat) * particles.dt
        particles.dlat += yd_ms / _DEG2M * particles.dt
    else:
        particles.dlon += xd_ms * particles.dt
        particles.dlat += yd_ms * particles.dt


def make_kernel(model):
    """Create a Parcels kernel for any LagrangianMechanicsModel.

    Args:
        model: LagrangianMechanicsModel instance.

    Returns:
        Kernel function ``(particles, fieldset)`` for ``pset.execute``.
    """
    physics = model.physics
    max_depth = model._max_depth(physics)

    def _kernel(particles, fieldset):
        sample_uv = _extract_profiles(particles, fieldset, max_depth)
        drift_vel, _, _ = model.steady_state_batch(sample_uv)
        _position_update(
            particles, drift_vel[:, 0], drift_vel[:, 1], fieldset,
        )

    return _kernel


def make_dd_kernel(dd):
    """Backward-compatible alias: create a kernel for a DroguedDrifter."""
    return make_kernel(dd)
```

## What a SparBuoy looks like

A spar buoy has 3 generalized coordinates (x, y, theta) and two bodies
(float at surface, keel at depth). Adding it requires one file, no
changes to any generic module.

```python
# models/spar_buoy.py

from typing import NamedTuple
import numpy as np
from ..base import LagrangianMechanicsModel


class SparBuoyPhysics(NamedTuple):
    m_float: float      # float mass [kg]
    m_keel: float       # keel mass [kg]
    l_keel: float       # keel depth below surface [m]
    g: float            # gravitational acceleration [m/s^2]
    k_float: float      # float drag coefficient [kg/m]
    k_keel: float       # keel drag coefficient [kg/m]


class SparBuoyState(NamedTuple):
    theta: float | np.ndarray
    xd: float | np.ndarray
    yd: float | np.ndarray
    thetad: float | np.ndarray
    U_float: float | np.ndarray
    V_float: float | np.ndarray
    U_keel: float | np.ndarray
    V_keel: float | np.ndarray


class SparBuoy(LagrangianMechanicsModel):
    """A spar buoy with a keel mass."""

    Physics = SparBuoyPhysics
    State = SparBuoyState
    n_q = 3                            # x, y, theta
    _drift_velocity_indices = (3, 4)   # xd, yd

    def default_physics(self):
        return SparBuoyPhysics(
            m_float=5.0, m_keel=10.0, l_keel=5.0,
            g=9.81, k_float=20.0, k_keel=50.0,
        )

    def _derive_symbolic(self):
        """Derive M, F for the spar buoy system."""
        # ... sympy derivation of the spar buoy Lagrangian ...
        # Returns (M_static, F_static, args) with 3x3 M.

    def _rhs_batch(self, Y, sample_uv):
        N = Y.shape[0]
        theta = Y[:, 2]
        xd, yd, thetad = Y[:, 3], Y[:, 4], Y[:, 5]

        z_keel = np.minimum(0.0, -self.physics.l_keel * np.cos(theta))

        U_f, V_f = sample_uv(np.zeros(N))
        U_k, V_k = sample_uv(z_keel)

        state = SparBuoyState(
            theta=theta, xd=xd, yd=yd, thetad=thetad,
            U_float=U_f, V_float=V_f,
            U_keel=U_k, V_keel=V_k,
        )

        qdd = self._qdd_func(self.physics, state)
        bad = ~np.isfinite(qdd).all(axis=1)
        if np.any(bad):
            qdd[bad] = 0.0

        dY = np.empty_like(Y)
        dY[:, :3] = Y[:, 3:]
        dY[:, 3:] = qdd
        return dY

    def _max_depth(self, physics):
        return physics.l_keel
```

No changes to `base.py`, `eom.py`, `velocity.py`, or `parcels.py`.
SparBuoy immediately inherits `steady_state_batch` and `make_kernel`.

## Module layout

```
src/drogued_drifters/
    __init__.py            Public API re-exports
    base.py                LagrangianMechanicsModel
    eom.py                 Generic: caching, lambdification, packing,
                             qdd/M/F evaluation
    coords.py              Stereographic <-> spherical transforms
    velocity.py            Profile interpolation (make_profile_sampler)
    parcels.py             Generic Parcels coupling
    stokes.py              Stokes drift profiles

    models/
        __init__.py
        drogued_drifter.py DroguedDrifter + DrifterPhysics + EOMState +
                             _derive_symbolic + drag helpers
        spar_buoy.py       (future)

    data/
        eom_cache_drogued_drifter.pkl
```

### Dependency graph

```
base.py                    (numpy, scipy -- imports eom inside __init__)
coords.py                  (numpy)
velocity.py                (numpy)
stokes.py                  (numpy)

eom.py                     (sympy, numpy -- reads model attrs at call time)

parcels.py                 (imports velocity)

models/drogued_drifter.py  (imports base, coords)
models/spar_buoy.py        (imports base)
```

No circular dependencies. `base.py` imports `eom._make_qdd_func` inside
`__init__`, not at module level. `eom.py` reads `model._derive_symbolic`,
`model.Physics`, etc. at call time via the model instance. Model modules
import `base` but not `eom` or `parcels`.

## Public API

```python
# __init__.py

# Primary model
from .models.drogued_drifter import (
    DroguedDrifter,
    DrifterPhysics,
    EOMState,
    drogue_horizontal_added_mass,
    buoy_horizontal_added_mass,
    drogue_horizontal_drag_coeff,
    buoy_horizontal_drag_coeff,
)

# Base class (for building new models)
from .base import LagrangianMechanicsModel

# EOM evaluation (takes model instance as first argument)
from .eom import eval_qdd, eval_M, eval_F

# Utilities
from .coords import _uv_to_spherical, _spherical_to_uv
from .velocity import make_profile_sampler
from .stokes import compute_stokes_profile

# Parcels
from .parcels import make_kernel, make_dd_kernel
```

## Data flow

### Parcels path (production)

```
pset.execute(kernel)
  |
  v
_kernel(particles, fieldset)                    # from parcels.make_kernel
  |
  +-- _extract_profiles(particles, fieldset, max_depth)
  |     -> sample_uv callable
  |
  +-- model.steady_state_batch(sample_uv)       # on base class
  |     |
  |     +-- model._rhs_batch(Y, sample_uv)      # on subclass
  |     |     |
  |     |     +-- sample_uv(z) calls             [inline]
  |     |     +-- self._qdd_func(physics, state) [cached]
  |     |     +-- NaN guard + pack derivatives   [inline]
  |     |
  |     `-- solve_ivp(rhs_flat, ...)
  |
  +-- _position_update(particles, xd, yd, fieldset)
```

### Standalone use

```python
from drogued_drifters import DroguedDrifter
from drogued_drifters.velocity import make_profile_sampler
import numpy as np

dd = DroguedDrifter()

z = np.linspace(-20, 0, 50)
U = 0.5 * np.exp(z / 5.0)
V = np.zeros_like(U)

# Single-particle trajectory
sample_uv = lambda depth: (np.interp(depth, z, U), np.interp(depth, z, V))
ds = dd.get_full_solution(sample_uv, t_span=(0, 120))
ds.xd.plot()

# Batch steady state (N=100 particles)
U_batch = np.tile(U[:, np.newaxis], (1, 100))
V_batch = np.tile(V[:, np.newaxis], (1, 100))
sample_uv_batch = make_profile_sampler(z, U_batch, V_batch)
xd, yd, Y, max_accel = dd.get_final_drift_batch(sample_uv_batch)
```

### Direct EOM study

```python
from drogued_drifters import DroguedDrifter, DrifterPhysics, EOMState
from drogued_drifters.eom import eval_qdd, eval_M, eval_F

dd = DroguedDrifter()
physics = dd.physics
state = EOMState(u_stereo=0.1, v_stereo=0.0, xd=0.0, yd=0.0,
                 ud_stereo=0.0, vd_stereo=0.0,
                 U_b=0.5, V_b=0.0, U_d=0.1, V_d=0.0)

qdd = eval_qdd(dd, physics, state)
M = eval_M(dd, physics, state)
F = eval_F(dd, physics, state)
assert np.allclose(M @ qdd, F)
```

## Migration path

### Phase 1: Module restructuring

1. Create `base.py` with `LagrangianMechanicsModel`.
2. Create `models/__init__.py`.
3. Create `models/drogued_drifter.py`:
   - Move `DrifterPhysics`, `EOMState` from `eom.py`.
   - Move `_derive_symbolic`, `_sym_norm` from `eom.py`.
   - Move `DroguedDrifter` class from `drifter.py`, refactored as
     subclass with `_rhs_batch(Y, sample_uv)` signature.
   - Move drag helpers and state index constants from `drifter.py`.
4. Modify `eom.py`: parameterize `_build_packer`, `_load_or_derive`,
   `_get_eom_callables`, `_make_qdd_func` on model instance.
   Add `eval_qdd`, `eval_M`, `eval_F`.
5. Rename `parcels_v4.py` → `parcels.py`. Parameterize
   `_extract_profiles` on `max_depth`. Add generic `make_kernel(model)`.
6. Update `__init__.py`.
7. Run tests, fix imports.

### Phase 2: Test migration

8. Update all test imports.
9. Update test fixtures: solve calls now take `sample_uv` as a
   positional parameter instead of it being stored on the instance.
10. Run full test suite.

### Phase 3: Notebook migration

11. Update example notebooks: add `sample_uv` argument to solve calls.
12. Re-run all notebooks with papermill.

### Phase 4: Cleanup

13. Delete `drifter.py`.
14. Invalidate EOM cache (path changed from `eom_cache.pkl` to
    `eom_cache_drogued_drifter.pkl`).

### Phase 5: SparBuoy (when ready)

15. Create `models/spar_buoy.py`.
16. Test with the generic machinery.

Phases 1-2 are the substantive work. Each phase is independently
committable and testable. The existing test suite stays valid throughout
(only import paths and `sample_uv` argument position change).

## What does NOT change

- **Symbolic derivation body.** `_derive_symbolic()` is identical, just
  moved to the model module.
- **Pickle cache format.** Same keys, same invalidation logic.
- **`_build_packer` logic.** Same signature-inspection approach,
  parameterized on types instead of hardcoded.
- **ODE solver.** `scipy.integrate.solve_ivp`.
- **Backend handling.** `_make_qdd_func(model, backend)` with optional
  numba JIT.
- **Velocity interpolation.** `velocity.py` unchanged.
- **Stokes drift.** `stokes.py` unchanged.
- **Coordinate convention.** z-up internally, stereographic for
  DroguedDrifter internals, spherical in public API.
- **Coordinate transforms.** `coords.py` unchanged.
- **Parcels profile extraction.** Same depth-level loop over
  `fieldset.UV.eval()`.
- **Parcels position update.** Same Euler-forward, same spherical/flat
  detection.
- **Test physics coverage.** Existing physics tests apply to the moved
  code with import-path changes only.
