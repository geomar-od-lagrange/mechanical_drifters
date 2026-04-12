# Architecture v3: multi-object Lagrangian models

Extends proposal B ([architecture-v2.md](architecture-v2.md)) to support
more than one type of drifting object, each derived via Lagrangian
mechanics. The motivating use case is adding a SparBuoy alongside the
existing DroguedDrifter, but the design should handle any M*qdd = F model
without requiring framework-level abstractions.

**Constraint:** only one object class is simulated at a time. No
mixed-object particle sets.

## Core observation

The current machinery already does most of the work generically:

1. `_build_packer` inspects a lambdified function's signature and maps
   parameter names to NamedTuple fields *by name*. It does not hardcode
   `DrifterPhysics` or `EOMState` -- it takes them as implicit inputs
   through the field-name matching. Generalizing it to accept *any* pair
   of NamedTuples is a one-line change.

2. `_make_qdd_func` wraps a raw lambdified callable into a
   `qdd_func(physics, state)` evaluator. The wrapping logic (pack args,
   call raw, reshape output) is the same for any model. Only the raw
   callable and the packer differ.

3. `_rhs_batch` is the one piece that is deeply object-specific. It
   knows the state vector layout (which indices are positions, which are
   velocities), how to compute drogue depth from generalized
   coordinates, and how to query velocity at specific depths. A SparBuoy
   would need a completely different `_rhs_batch`.

The design therefore splits along this seam: **generic EOM machinery**
(derivation, caching, lambdification, packing) lives in shared code;
**object-specific knowledge** (state layout, depth queries, RHS
assembly) lives in per-model modules.

## Abstraction boundary: ModelSpec, not a class hierarchy

The right abstraction is a plain data object -- a `ModelSpec` -- that
carries everything the generic machinery needs to know about a specific
Lagrangian model. No Protocol, no ABC, no registration. Just a
NamedTuple (or dataclass) that bundles the model-specific pieces.

Why not a Protocol/ABC?

- There is no polymorphic dispatch. Only one model is active at a time.
  The "generic" code does not call methods through an interface -- it
  receives concrete functions and data structures as arguments.
- A Protocol would force each model to implement the same method names,
  which is artificial when the models have fundamentally different
  physics.
- This is research code with two (maybe three) models. Convention beats
  mechanism.

Why not a registry?

- Registries add indirection for the benefit of plugin discovery. We
  have two models in one repo. Import paths are fine.

## The ModelSpec

```python
from typing import NamedTuple

class ModelSpec(NamedTuple):
    """Everything the generic machinery needs from a Lagrangian model."""

    name: str
    # --- Types ---
    physics_type: type             # NamedTuple class (e.g. DrifterPhysics)
    state_type: type               # NamedTuple class (e.g. EOMState)
    # --- State vector ---
    n_q: int                       # number of generalized coordinates
    state_size: int                # total state vector length (2 * n_q)
    velocity_indices: tuple[int, ...]   # indices of qdot in state vector
    accel_slice: slice             # where qdd goes in dY (= slice(n_q, 2*n_q))
    # --- Symbolic derivation ---
    derive_symbolic: callable      # () -> (M_static, F_static, args)
    # --- Depth query ---
    z_query: callable              # (physics, Y_batch) -> dict[str, z_array]
    # --- RHS assembly ---
    build_state: callable          # (Y_batch, velocity_dict) -> state NamedTuple
    pack_derivatives: callable     # (Y_batch, qdd_batch) -> dY_batch
    # --- Cache ---
    cache_path: Path               # path to model-specific pickle cache
```

Each field is documented below.

### `physics_type` and `state_type`

The NamedTuple classes themselves. `_build_packer` needs these to map
lambda parameter names to struct fields. Currently it hardcodes
`DrifterPhysics._fields` and `EOMState._fields`. Instead, it receives
them as arguments:

```python
def _build_packer(raw_func, physics_type, state_type):
    param_names = list(inspect.signature(raw_func).parameters)
    physics_fields = physics_type._fields
    state_fields = state_type._fields
    # ... rest unchanged
```

### `n_q` and `state_size`

The DroguedDrifter has 4 generalized coordinates `[x, y, u_stereo,
v_stereo]` and 4 velocities, so `n_q = 4`, `state_size = 8`. A triple
pendulum might have `n_q = 6`, `state_size = 12`. The solver needs
these to allocate arrays and reshape between flat and structured
representations.

### `velocity_indices`

Indices into the state vector for the generalized velocities. For the
DroguedDrifter: `(4, 5, 6, 7)` (xd, yd, ud_stereo, vd_stereo). The
solver uses these to extract qdot from Y and to write qdd into dY. This
replaces the hardcoded `IX, IY, ...` constants.

### `accel_slice`

The slice of dY where accelerations (qdd) are written. For
DroguedDrifter: `slice(4, 8)`. This is always `slice(n_q, 2*n_q)` for
standard Lagrangian systems where the state vector is `[q, qdot]`, but
making it explicit costs nothing and allows non-standard layouts.

### `derive_symbolic`

The model-specific symbolic derivation function. For DroguedDrifter,
this is the current `_derive_symbolic()`. For SparBuoy, a completely
different function deriving the SparBuoy Lagrangian. Returns the same
`(M_static, F_static, args)` triple -- this is the contract.

The args tuple must use symbol names that match field names in
`physics_type` and `state_type`, enabling `_build_packer` to map them.

### `z_query`

How to get depth queries from the generalized coordinates. This is the
generalization of `_z_eff`. Different models query velocity at different
depths.

```python
def drifter_z_query(physics, Y):
    """Return depth queries for DroguedDrifter.

    Args:
        physics: DrifterPhysics.
        Y: (N, 8) state array.

    Returns:
        dict mapping query names to (N,) depth arrays.
        DroguedDrifter needs: {"buoy": zeros(N), "drogue": z_eff(N,)}.
    """
    u, v = Y[:, 2], Y[:, 3]
    s = u**2 + v**2
    cos_theta = (s - 4) / (s + 4)
    z_eff = np.minimum(0.0, physics.l * cos_theta)
    return {"buoy": np.zeros(N), "drogue": z_eff}
```

A SparBuoy with three bodies might return:

```python
{"float": np.zeros(N), "mid_mass": z_mid, "keel": z_keel}
```

The velocity interpolator is called once per key, and the results are
passed to `build_state`.

### `build_state`

Assembles the model's state NamedTuple from the ODE state array and the
interpolated velocities. This replaces the inline state construction in
`_rhs_batch`.

```python
def drifter_build_state(Y, velocity_dict):
    """Build EOMState from state array and interpolated velocities.

    Args:
        Y: (N, 8) state array.
        velocity_dict: {"buoy": (U, V), "drogue": (U, V)}.

    Returns:
        EOMState instance with (N,) arrays.
    """
    U_b, V_b = velocity_dict["buoy"]
    U_d, V_d = velocity_dict["drogue"]
    return EOMState(
        u_stereo=Y[:, 2], v_stereo=Y[:, 3],
        xd=Y[:, 4], yd=Y[:, 5],
        ud_stereo=Y[:, 6], vd_stereo=Y[:, 7],
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )
```

### `pack_derivatives`

Assembles the derivative array dY from the state array and the computed
accelerations. This replaces the inline derivative packing in
`_rhs_batch`.

```python
def drifter_pack_derivatives(Y, qdd):
    """Pack derivatives for DroguedDrifter.

    Args:
        Y: (N, 8) state array.
        qdd: (N, 4) accelerations.

    Returns:
        dY: (N, 8) derivative array.
    """
    dY = np.empty_like(Y)
    dY[:, :4] = Y[:, 4:]   # d(q)/dt = qdot
    dY[:, 4:] = qdd         # d(qdot)/dt = qdd
    return dY
```

For the standard `[q, qdot]` layout this is always the same, but models
with non-standard layouts (e.g., extra non-dynamic state variables) can
override it.

### `cache_path`

Where to store the pickled symbolic derivation. Each model gets its own
cache file to avoid collisions.

```
data/eom_cache_drogued_drifter.pkl
data/eom_cache_spar_buoy.pkl
```

## Module layout

```
src/drogued_drifters/
    __init__.py               Public API re-exports
    eom.py                    Generic: derivation, caching, lambdification,
                                packing, qdd/M/F evaluation
    model_spec.py             ModelSpec definition
    coords.py                 Stereographic <-> spherical (DroguedDrifter-specific,
                                but reusable math)
    velocity.py               Profile interpolation, Stokes drift (object-agnostic)
    solve.py                  Generic ODE integration
    parcels.py                Generic Parcels coupling

    models/
        __init__.py
        drogued_drifter/
            __init__.py       DroguedDrifterSpec, DrifterPhysics, EOMState,
                                DEFAULT_PHYSICS, DroguedDrifter class
            physics.py        DrifterPhysics NamedTuple, drag helpers, z_eff
            state.py          EOMState NamedTuple
            symbolic.py       _derive_symbolic() for this model
            rhs.py            z_query, build_state, pack_derivatives
            spec.py           Assembles ModelSpec for this model
        spar_buoy/            (future, same structure)
            ...

    data/
        eom_cache_drogued_drifter.pkl
```

### Why `models/` as a sub-package?

Each model has ~4 small files (~50-100 lines each). Keeping them in the
top-level package would pollute it with `drifter_physics.py`,
`drifter_state.py`, `drifter_symbolic.py`, `spar_physics.py`, etc.
The `models/` sub-package groups them cleanly.

The alternative -- one file per model -- would work for small models
(the DroguedDrifter model-specific code is ~250 lines total) but becomes
unwieldy as models grow. The sub-package structure scales and keeps
each file focused.

### Simplification: one module per model

On reflection, the DroguedDrifter-specific code is small enough that it
could all live in a single module `models/drogued_drifter.py` rather
than a sub-package. This is simpler and avoids the 4-file overhead.

**Preferred layout:**

```
src/drogued_drifters/
    __init__.py
    eom.py                    Generic EOM machinery
    model_spec.py             ModelSpec NamedTuple
    coords.py                 Stereographic <-> spherical
    velocity.py               Profile interpolation, Stokes drift
    solve.py                  Generic ODE integration
    parcels.py                Generic Parcels coupling

    models/
        __init__.py
        drogued_drifter.py    All DroguedDrifter-specific code:
                                DrifterPhysics, EOMState, _derive_symbolic,
                                z_query, build_state, pack_derivatives,
                                DEFAULT_PHYSICS, SPEC, DroguedDrifter class,
                                drag/added-mass helpers
        spar_buoy.py          (future)

    data/
        eom_cache_drogued_drifter.pkl
```

This is the layout we should implement. The sub-package version is
there if models become large enough to justify it.

## Generic machinery: what changes

### `eom.py` -- parameterized on ModelSpec

The current `eom.py` (proposal B) or `lagrange_model.py` (current)
has hardcoded references to `DrifterPhysics` and `EOMState`. These
become parameters.

**`_build_packer(raw_func, physics_type, state_type)`** -- already
discussed above. One-line change: accept the types as arguments instead
of importing them.

**`_load_or_derive(spec)`** -- takes a ModelSpec instead of implicitly
knowing the derivation function and cache path.

```python
def _load_or_derive(spec):
    """Load symbolic EOM from cache, or derive from scratch.

    Args:
        spec: ModelSpec with derive_symbolic and cache_path.

    Returns:
        (M_static, F_static, qdd_exprs, args)
    """
    cache_path = spec.cache_path
    key = _cache_key(spec.derive_symbolic)

    if cache_path.exists():
        try:
            cached = pickle.loads(cache_path.read_bytes())
            if cached.get("key") == key:
                return cached["M"], cached["F"], cached["qdd"], cached["args"]
        except Exception as e:
            warnings.warn(f"EOM cache load failed: {e}", stacklevel=2)

    M_static, F_static, args = spec.derive_symbolic()
    qdd_vec = M_static.LUsolve(F_static)
    qdd_exprs = tuple(qdd_vec[i] for i in range(spec.n_q))
    # ... cache write ...
    return M_static, F_static, qdd_exprs, args
```

**`_cache_key(derive_fn)`** -- hashes the source of the passed function
instead of hardcoding `_derive_symbolic`.

**`_get_eom_callables(spec)`** -- cached on `spec.name` (since
ModelSpec is a NamedTuple containing callables which are not hashable
by value, use the name as cache key).

```python
@functools.lru_cache()
def _get_eom_callables(model_name):
    """Internal: cached by model name. Use get_eom_callables(spec) instead."""
    ...

def get_eom_callables(spec):
    """Load/derive EOM, lambdify, build packer. Cached per model name."""
    return _get_eom_callables_impl(spec)
```

Actually, the cleanest approach is to store the spec in a module-level
dict keyed by name, and cache callables in that dict:

```python
_CALLABLE_CACHE = {}  # model_name -> (qdd_raw, M_raw, F_raw, pack_eom_args)

def get_eom_callables(spec):
    if spec.name not in _CALLABLE_CACHE:
        M_static, F_static, qdd_exprs, args = _load_or_derive(spec)
        # ... lambdify, build packer ...
        pack_eom_args = _build_packer(qdd_raw, spec.physics_type, spec.state_type)
        _CALLABLE_CACHE[spec.name] = (qdd_raw, M_raw, F_raw, pack_eom_args)
    return _CALLABLE_CACHE[spec.name]
```

**`make_qdd_func(spec, backend="numpy")`** -- the factory is
parameterized on the spec.

```python
_QDD_CACHE = {}  # (model_name, backend) -> qdd_func

def make_qdd_func(spec, backend="numpy"):
    key = (spec.name, backend)
    if key not in _QDD_CACHE:
        qdd_raw, _, _, pack_eom_args = get_eom_callables(spec)
        # ... wrap with batch detection, optional numba ...
        _QDD_CACHE[key] = qdd_func
    return _QDD_CACHE[key]
```

**`eval_qdd(spec, physics, state, *, backend="numpy")`** -- public
entry point.

```python
def eval_qdd(spec, physics, state, *, backend="numpy"):
    return make_qdd_func(spec, backend)(physics, state)
```

**`eval_M(spec, physics, state)` and `eval_F(spec, physics, state)`** --
same pattern.

Note: for the DroguedDrifter, `spec` is always the same object. The
public API (in `models/drogued_drifter.py`) provides convenience
wrappers that hide it:

```python
from ..eom import eval_qdd as _eval_qdd

def qdd_func(physics, state, *, backend="numpy"):
    return _eval_qdd(SPEC, physics, state, backend=backend)
```

### `solve.py` -- generic over ModelSpec

The solver becomes model-agnostic. The key function is `_rhs_batch`,
which currently hardcodes the DroguedDrifter state layout.

**Generic `_rhs_batch`:**

```python
def _rhs_batch(Y, spec, physics, depth_levels, U_profiles, V_profiles, qdd_func):
    """Vectorized RHS for N particles, generic over model type.

    Args:
        Y: (N, spec.state_size) state array.
        spec: ModelSpec.
        physics: Physics NamedTuple (type matches spec.physics_type).
        depth_levels: (D,) depth levels for velocity interpolation.
        U_profiles, V_profiles: (D, N) velocity profiles.
        qdd_func: callable(physics, state) -> (N, n_q) accelerations.

    Returns:
        dY: (N, spec.state_size) derivatives.
    """
    N = Y.shape[0]

    # 1. Query depths from generalized coordinates
    z_dict = spec.z_query(physics, Y)

    # 2. Interpolate velocity at each depth query
    velocity_dict = {}
    for name, z_arr in z_dict.items():
        U, V = interpolate_profile(z_arr, depth_levels, U_profiles, V_profiles)
        velocity_dict[name] = (U, V)

    # 3. Build model-specific state NamedTuple
    state = spec.build_state(Y, velocity_dict)

    # 4. Evaluate accelerations
    qdd = qdd_func(physics, state)

    # 5. Guard NaN/inf
    bad = ~np.isfinite(qdd).all(axis=1)
    if np.any(bad):
        qdd[bad] = 0.0

    # 6. Pack derivatives (model-specific layout)
    return spec.pack_derivatives(Y, qdd)
```

This is the heart of the generalization. Steps 1, 3, and 6 are
model-specific (delegated to spec callables). Steps 2, 4, and 5 are
generic.

**`steady_state_drift`** becomes:

```python
def steady_state_drift(
    spec,
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
    drift_velocity_indices=None,   # which state indices are the "drift velocity"
):
    ...
```

The `drift_velocity_indices` parameter tells the solver which
components of the state vector to return as the "drift velocity". For
DroguedDrifter, these are `(4, 5)` (xd, yd -- the buoy translation
velocities). For a model where the primary output is the velocity of
a different body, this would be different indices.

Alternatively, `drift_velocity_indices` could live on the ModelSpec.
This is cleaner -- the model knows which velocities are the "answer":

```python
class ModelSpec(NamedTuple):
    ...
    drift_velocity_indices: tuple[int, ...]  # indices of the "answer" in state vector
```

**`full_trajectory`** is model-specific in its xr.Dataset construction
(variable names, coordinate transforms for display). Two options:

1. **The solver returns raw arrays; the model wraps them.** The solver
   returns `(t, Y)` and the model's convenience function converts to an
   xr.Dataset with model-specific variable names and optional coordinate
   transforms. This is the cleanest split.

2. **The solver takes a `format_output` callable.** Over-engineering.

Option 1 is right. `solve.py` provides:

```python
def integrate(spec, physics, depth_levels, U, V, *, t_span, y0, t_eval, ...):
    """Integrate the ODE. Returns (t_array, Y_array)."""
```

Each model's convenience function wraps this:

```python
# In models/drogued_drifter.py
def full_trajectory(physics, depth_levels, U, V, **kw):
    t, Y = solve.integrate(SPEC, physics, depth_levels, U, V, **kw)
    # Convert to xr.Dataset with theta, phi, etc.
    ...
```

### `parcels.py` -- generic over ModelSpec

The Parcels coupling has three concerns:

1. **Profile extraction** -- querying the fieldset at depth levels. This
   is model-agnostic except for *how deep* to sample. The model spec
   can provide a `max_depth(physics)` function, or `extract_profiles`
   can take `max_depth` as a parameter.

2. **Steady-state solve** -- calls `steady_state_drift`. Already generic
   via ModelSpec.

3. **Position update** -- Euler-forward in lat/lon or x/y. This is
   model-agnostic (it just needs drift velocity components).

```python
def make_kernel(spec, physics=None, *, backend="numpy"):
    """Create a Parcels kernel for any Lagrangian model.

    Args:
        spec: ModelSpec instance.
        physics: Physics NamedTuple, or None for model defaults.
        backend: "numpy" or "numba".

    Returns:
        Kernel function (particles, fieldset).
    """
    if physics is None:
        physics = spec.default_physics

    def _kernel(particles, fieldset):
        depth_levels, U, V = extract_profiles(
            particles, fieldset, max_depth=spec.max_depth(physics),
        )
        xd, yd, _, _ = steady_state_drift(
            spec, physics, depth_levels, U, V, backend=backend,
        )
        position_update(particles, xd, yd, fieldset)

    return _kernel
```

This requires two more fields on ModelSpec:

```python
class ModelSpec(NamedTuple):
    ...
    default_physics: object          # default physics instance (e.g. DEFAULT_PHYSICS)
    max_depth: callable              # (physics) -> float, deepest depth to sample
```

For DroguedDrifter: `max_depth = lambda physics: physics.l` (pole
length determines how deep to sample).

**Backward compatibility:** `make_dd_kernel` becomes a thin wrapper:

```python
def make_dd_kernel(physics=None, *, backend="numpy"):
    from .models.drogued_drifter import SPEC
    return make_kernel(SPEC, physics, backend=backend)
```

## Revised ModelSpec

Consolidating the discussion above:

```python
class ModelSpec(NamedTuple):
    """Specification for a Lagrangian drifting-object model.

    Bundles everything the generic EOM/solver/Parcels machinery needs
    to work with a specific model.  Each model defines one SPEC
    instance as a module-level constant.
    """

    # Identity
    name: str                          # unique name, used as cache key

    # Types
    physics_type: type                 # NamedTuple class for physics constants
    state_type: type                   # NamedTuple class for per-step state+forcing

    # State vector geometry
    n_q: int                           # number of generalized coordinates
    state_size: int                    # total state vector length (usually 2*n_q)
    drift_velocity_indices: tuple      # which state components are the "answer"

    # Symbolic derivation
    derive_symbolic: callable          # () -> (M_static, F_static, args)
    cache_path: object                 # pathlib.Path to pickle cache

    # RHS assembly (all take batch inputs)
    z_query: callable                  # (physics, Y) -> dict[str, ndarray]
    build_state: callable              # (Y, velocity_dict) -> state NamedTuple
    pack_derivatives: callable         # (Y, qdd) -> dY

    # Parcels / convenience
    default_physics: object            # default physics instance
    max_depth: callable                # (physics) -> float
```

## DroguedDrifter model definition

Here is the complete DroguedDrifter as a ModelSpec:

```python
# models/drogued_drifter.py

from pathlib import Path
from typing import NamedTuple
import numpy as np

from ..model_spec import ModelSpec


class DrifterPhysics(NamedTuple):
    m_b: float
    m_d: float
    m_hat_d: float
    m_tilde_d: float
    m_tilde_b: float
    l: float
    g: float
    k_b: float
    k_d: float


class EOMState(NamedTuple):
    u_stereo: ...
    v_stereo: ...
    xd: ...
    yd: ...
    ud_stereo: ...
    vd_stereo: ...
    U_b: ...
    V_b: ...
    U_d: ...
    V_d: ...


DEFAULT_PHYSICS = DrifterPhysics(
    m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
    l=3.0, g=9.81, k_b=12.0, k_d=154.0,
)


def _derive_symbolic():
    """Derive M, F for the buoy-pole-drogue system in stereographic coords."""
    # ... current _derive_symbolic body, unchanged ...


def _z_query(physics, Y):
    N = Y.shape[0]
    u, v = Y[:, 2], Y[:, 3]
    s = u**2 + v**2
    cos_theta = (s - 4) / (s + 4)
    z_eff = np.minimum(0.0, physics.l * cos_theta)
    return {"buoy": np.zeros(N), "drogue": z_eff}


def _build_state(Y, velocity_dict):
    U_b, V_b = velocity_dict["buoy"]
    U_d, V_d = velocity_dict["drogue"]
    return EOMState(
        u_stereo=Y[:, 2], v_stereo=Y[:, 3],
        xd=Y[:, 4], yd=Y[:, 5],
        ud_stereo=Y[:, 6], vd_stereo=Y[:, 7],
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )


def _pack_derivatives(Y, qdd):
    dY = np.empty_like(Y)
    dY[:, :4] = Y[:, 4:]  # qdot
    dY[:, 4:] = qdd        # qdd
    return dY


def _max_depth(physics):
    return physics.l


SPEC = ModelSpec(
    name="drogued_drifter",
    physics_type=DrifterPhysics,
    state_type=EOMState,
    n_q=4,
    state_size=8,
    drift_velocity_indices=(4, 5),    # xd, yd
    derive_symbolic=_derive_symbolic,
    cache_path=Path(__file__).resolve().parent.parent / "data" / "eom_cache_drogued_drifter.pkl",
    z_query=_z_query,
    build_state=_build_state,
    pack_derivatives=_pack_derivatives,
    default_physics=DEFAULT_PHYSICS,
    max_depth=_max_depth,
)
```

The `DroguedDrifter` convenience class lives in the same module:

```python
class DroguedDrifter:
    """Convenience wrapper for interactive use."""

    def __init__(self, physics=None, *, backend="numpy"):
        self.physics = physics or DEFAULT_PHYSICS
        self.backend = backend
        self.spec = SPEC

    def steady_state(self, depth_levels, U_profiles, V_profiles, **kw):
        from ..solve import steady_state_drift
        return steady_state_drift(
            self.spec, self.physics, depth_levels, U_profiles, V_profiles,
            backend=self.backend, **kw,
        )

    def full_trajectory(self, depth_levels, U, V, **kw):
        from ..solve import integrate
        t, Y = integrate(
            self.spec, self.physics, depth_levels, U, V,
            backend=self.backend, **kw,
        )
        return self._to_dataset(t, Y)

    def _to_dataset(self, t, Y):
        """Convert raw arrays to xr.Dataset with spherical angles."""
        import xarray as xr
        from ..coords import _uv_to_spherical
        theta, phi, thetad, phid = _uv_to_spherical(
            Y[:, 2], Y[:, 3], Y[:, 6], Y[:, 7],
        )
        return xr.Dataset(
            {"x": ("time", Y[:, 0]), "y": ("time", Y[:, 1]),
             "theta": ("time", theta), "phi": ("time", phi),
             "xd": ("time", Y[:, 4]), "yd": ("time", Y[:, 5]),
             "thetad": ("time", thetad), "phid": ("time", phid)},
            coords={"time": t},
        )
```

## What a SparBuoy looks like

To add a new model, create `models/spar_buoy.py` with:

1. A `SparBuoyPhysics` NamedTuple with its own fields.
2. A `SparBuoyState` NamedTuple with its own fields.
3. A `_derive_symbolic()` for the spar buoy Lagrangian.
4. `_z_query`, `_build_state`, `_pack_derivatives` for its geometry.
5. A `SPEC = ModelSpec(...)` assembling everything.
6. Optionally a `SparBuoy` convenience class.

No changes to `eom.py`, `solve.py`, `velocity.py`, or `parcels.py`.
No registration. No ABC implementation. Just a new file.

## Data flow (generic)

### Parcels path

```
pset.execute(kernel)
  |
  v
_kernel(particles, fieldset)
  |
  +-- extract_profiles(particles, fieldset, max_depth)
  |     -> (depth_levels, U_profiles, V_profiles)
  |
  +-- steady_state_drift(spec, physics, depth_levels, U, V, backend=)
  |     |
  |     +-- make_qdd_func(spec, backend)  -> qdd_fn     [cached]
  |     +-- _rhs_batch(Y, spec, physics, depth_levels, U, V, qdd_fn)
  |     |     |
  |     |     +-- spec.z_query(physics, Y) -> z_dict
  |     |     +-- interpolate_profile(z, ...) for each z in z_dict
  |     |     +-- spec.build_state(Y, velocity_dict) -> state
  |     |     +-- qdd_fn(physics, state) -> accelerations
  |     |     +-- spec.pack_derivatives(Y, qdd) -> dY
  |     |
  |     `-- solve_ivp(rhs_flat, ...)
  |
  +-- position_update(particles, xd, yd, fieldset)
```

### Direct EOM study

```python
from drogued_drifters.models.drogued_drifter import (
    DrifterPhysics, EOMState, SPEC, DEFAULT_PHYSICS,
)
from drogued_drifters.eom import eval_qdd, eval_M, eval_F

physics = DEFAULT_PHYSICS
state = EOMState(u_stereo=0.1, v_stereo=0.0, ...)

qdd = eval_qdd(SPEC, physics, state)
M = eval_M(SPEC, physics, state)
F = eval_F(SPEC, physics, state)
assert np.allclose(M @ qdd, F)
```

## What stays in top-level `__init__.py`

```python
# Backward-compatible re-exports (DroguedDrifter is the default model)
from .models.drogued_drifter import (
    DroguedDrifter,
    DrifterPhysics,
    EOMState,
    DEFAULT_PHYSICS,
    SPEC as DROGUED_DRIFTER_SPEC,
)
from .eom import eval_qdd, eval_M, eval_F
from .coords import uv_to_theta, uv_to_spherical, spherical_to_uv
from .velocity import interpolate_profile, compute_stokes_profile
from .parcels import make_dd_kernel  # backward-compatible name

# For users who want the model-agnostic API:
from .parcels import make_kernel
from .solve import steady_state_drift, integrate
from .model_spec import ModelSpec
```

## Dependency graph

```
model_spec.py                (standalone, no imports)
coords.py                    (standalone, numpy)
velocity.py                  (standalone, numpy)

eom.py                       (imports model_spec)
  |
  v
solve.py                     (imports eom, velocity)
  |
  v
parcels.py                   (imports solve, velocity)

models/drogued_drifter.py    (imports model_spec, coords)
models/spar_buoy.py          (imports model_spec)
```

No circular dependencies. Model modules do not import from `eom.py` or
`solve.py` -- they define data and callables that are *consumed* by
those modules. The dependency arrow flows from generic machinery toward
model specs (at call time, via parameters), not the other way.

## Cost of generalization

### What gets harder

- **Three extra arguments to `_rhs_batch`.** The spec flows through the
  call chain. This is a minor readability cost.

- **Indirection in the RHS.** Instead of inline `z_eff = ...`, the
  generic RHS calls `spec.z_query(physics, Y)`. One level of
  indirection. The callables are plain functions, not method lookups, so
  the performance cost is negligible.

- **ModelSpec has 12 fields.** It is large for a NamedTuple. But it is
  constructed once per model (as a module-level constant) and never
  modified. The fields are all documented and motivated.

### What gets easier

- **Adding a new model.** One file, no framework changes, no base class
  methods to implement.

- **Testing generic machinery.** `eom.py` and `solve.py` can be tested
  with a trivial synthetic ModelSpec (e.g., a 1-DOF harmonic oscillator)
  without involving the full drogued drifter derivation.

- **Understanding the architecture.** The ModelSpec makes the contract
  between generic and specific code explicit. Currently, the contract
  is implicit: scattered hardcoded references to `DrifterPhysics`,
  `EOMState`, `IX`, `IU`, `_z_eff`, etc.

## Trade-offs considered

### Why not just copy-paste for SparBuoy?

The alternative to all this is: copy `eom.py`, `solve.py`, `parcels.py`
into a `spar_buoy/` directory and modify them. No abstraction needed.

This works for two models. It fails at three, because bug fixes need to
be applied in three places. The generic EOM caching, lambdification,
NaN guarding, solve_ivp setup, and Parcels profile extraction are ~300
lines of non-trivial code that should not be duplicated.

### Why not Protocols?

A `LagrangianModel` Protocol with methods `derive_symbolic`,
`z_query`, `build_state`, etc. would work. But:

- It requires each model to be a class (or at least an object with
  methods). A NamedTuple of functions is lighter.
- Protocols shine when you have code that receives an unknown
  implementation at runtime. We always know which model we have -- it is
  a parameter, not a plugin.
- The ModelSpec approach is just a Protocol written as data instead of as
  method signatures. It communicates the same contract with less
  ceremony.

### Why NamedTuple for ModelSpec, not dataclass?

ModelSpec is immutable and constructed once. NamedTuple is the right
tool. Dataclass would work too (with `frozen=True`), but NamedTuple is
consistent with the rest of the codebase (`DrifterPhysics`, `EOMState`
are NamedTuples).

One concern: NamedTuples containing callables are not hashable, so
`ModelSpec` cannot be used as an `lru_cache` key directly. The caching
uses `spec.name` (a string) instead. This is fine -- model names are
unique.

**Actually, use a dataclass.** NamedTuples are positional, and 12
positional fields are error-prone. A frozen dataclass with keyword-only
construction is safer:

```python
@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    physics_type: type
    state_type: type
    n_q: int
    state_size: int
    drift_velocity_indices: tuple
    derive_symbolic: callable
    cache_path: Path
    z_query: callable
    build_state: callable
    pack_derivatives: callable
    default_physics: object
    max_depth: callable
```

This is the right call. Use dataclass for ModelSpec, NamedTuple for
physics/state types. The physics/state types benefit from being tuples
(positional packing for lambda args). ModelSpec does not.

## Migration path

### Phase 1: Extract model-specific code (mechanical)

1. Create `models/drogued_drifter.py`. Move `DrifterPhysics`,
   `EOMState`, `DEFAULT_PHYSICS`, `_derive_symbolic`, drag helpers,
   `DroguedDrifter` class there.
2. Create `model_spec.py` with the `ModelSpec` dataclass.
3. Define `SPEC` in `models/drogued_drifter.py`.
4. Create `coords.py` (move from `lagrange_model.py`).
5. Create `velocity.py` (move `make_profile_sampler` + `stokes.py`).
6. Update all imports. Run tests.

This is the proposal B refactor, extended with ModelSpec.

### Phase 2: Generalize eom.py

7. Parameterize `_build_packer`, `_load_or_derive`, `_get_eom_callables`,
   `_make_qdd_func`, `eval_qdd`, `eval_M`, `eval_F` on ModelSpec.
8. Write `_z_query`, `_build_state`, `_pack_derivatives` for the
   DroguedDrifter spec.
9. Update tests.

### Phase 3: Generalize solve.py

10. Rewrite `_rhs_batch` as a generic function taking ModelSpec.
11. Rewrite `steady_state_drift` as a generic function.
12. Add `integrate()` as the raw solver interface.
13. The DroguedDrifter class delegates to these.
14. Update tests.

### Phase 4: Generalize parcels.py

15. Add `make_kernel(spec, physics, backend)`.
16. `make_dd_kernel` becomes a thin wrapper.
17. Update Parcels tests.

### Phase 5: Add SparBuoy (when ready)

18. Create `models/spar_buoy.py` with its own physics, state,
    derivation, and spec.
19. Test with the same generic machinery.

Phases 1-4 can be done incrementally. Each phase is independently
testable. Phase 5 is the payoff.

## What this does NOT change

- **Velocity protocol.** `interpolate_profile` / `make_profile_sampler`
  are completely model-agnostic. No changes.
- **Stokes drift.** Unchanged.
- **Coordinate transforms.** `coords.py` stays DroguedDrifter-specific
  but lives in its own module. Other models may have their own
  coordinate utilities.
- **Parcels profile extraction.** The depth-level loop over
  `fieldset.UV.eval()` is model-agnostic.
- **ODE solver choice.** Still `scipy.integrate.solve_ivp`.
- **Backend handling.** Still `_make_qdd_func(backend)` with optional
  numba.
- **Test structure.** Existing tests stay, new tests added for generic
  machinery and SparBuoy.
