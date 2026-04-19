Implemented. See [docs/architecture.md](../../docs/architecture.md).

# Plan: model-independence refactor

Make each model self-contained, push Parcels to the boundary, consolidate
the evaluation API, and thin out shared infrastructure.

## Context

Track G gave us a working multi-model architecture. But the base class and
shared modules were shaped around one model (DroguedDrifter) and one
integration path (Parcels). Now that PointSurfaceDrifter exists and more
models may follow, the seams are showing. The TODOs scattered across `src/`
all point at the same family of problems.

## Guiding principles

1. **Models are independent units.** Each model owns its physics types, its
   coordinate helpers, its RHS. No shared utility modules that silently
   couple models. Trivial duplication of small helpers is cheaper than
   coupling.

2. **The base class is a thin contract.** It defines what a model must
   provide and offers one clean evaluation method. It doesn't know about
   Parcels, doesn't extract velocities by index magic, doesn't proliferate
   method variants.

3. **Parcels is an adapter, not a core concern.** `parcels.py` imports
   models, not the other way around. Models have zero Parcels awareness.

4. **Honest names.** "Drift after N seconds", not "steady state". Let
   callers inspect convergence. Readability >> DRY cleverness.

5. **eom.py does one thing well.** It's the sympy-to-numpy pipeline:
   derive, cache, lambdify, evaluate. Strip the over-engineered public
   surface area; keep the useful core.


## What changes

### 1. Delete `coords.py` — inline into `drogued_drifter.py`

The three functions (`_uv_to_theta`, `_uv_to_spherical`,
`_spherical_to_uv`) are only used by DroguedDrifter. Move them into the
model module. Delete `coords.py`.

Tests that import from `mechanical_drifters.coords` update to import from
`mechanical_drifters.models.drogued_drifter`.

### 2. Delete `velocity.py` — inline `make_profile_sampler` into `parcels.py`

`make_profile_sampler` assumes shapes that only make sense in the Parcels
context. Move it into `parcels.py` as a private function. Delete
`velocity.py`.

Tests that import from `mechanical_drifters.velocity` update to import from
`mechanical_drifters.parcels`.

### 3. Remove `make_kernel` from the base class

Delete `LagrangianMechanicsModel.make_kernel()`. The free function
`parcels.make_kernel(model)` is the only entry point. Callers already
use it directly in notebooks and docs:

```python
from mechanical_drifters.parcels import make_kernel
kernel = make_kernel(model)
```

### 4. Remove `_max_depth` from the base class contract

`_max_depth` exists solely for the Parcels coupling. PointSurfaceDrifter
returns a dummy `0.0` to satisfy the contract. Instead, let `parcels.py`
query it:

- Keep `_max_depth` as an **optional** method on models that need depth
  sampling (DroguedDrifter).
- In `parcels.py`, use `getattr(model, '_max_depth', None)` and default
  to `0.0` (surface only) if absent.

This way PointSurfaceDrifter doesn't need the method at all.

### 5. Replace `_drift_velocity_indices` with a `drift_velocity()` method

Instead of index-tuple extraction on the base class, each model implements
a method that returns drift velocity from state. More readable, no index
magic.

### 6. Drop `default_physics` from base class contract

The base `__init__` takes a required `physics` arg. Each model's own
`__init__` can provide defaults however it likes — that's not the base
class's business.

### 7. Rename `steady_state_batch` → `integrate`

Honest name: "integrate the ODE for the given time span". The caller
decides whether the result is converged. A single particle is N=1 —
no separate `integrate_single`.

### 8. Make `integrate` public-coords-in/out

DroguedDrifter overrides `integrate` to convert spherical→stereo on
entry, stereo→spherical on exit. Callers always see spherical coords.
`_to_public_state` / `_from_public_state` are private helpers inside the
model, present for dev clarity.

PointSurfaceDrifter needs no override — internal = public.

### 9. Consolidate DroguedDrifter's evaluation methods

Current DroguedDrifter has 5 entry points:
- `_rhs` (scalar, single particle)
- `_rhs_batch` (vectorized, base contract)
- `get_final_drift` (scalar, single particle, public coords)
- `get_final_drift_batch` (batch, public coords, wraps steady_state_batch)
- `get_full_solution` (single particle, returns xr.Dataset)

Target: 2 entry points:
- `_rhs_batch` (vectorized, base contract)
- inherits `integrate` from base (overridden for coord conversion)

Delete `get_final_drift`, `get_final_drift_batch`, `get_full_solution`,
`_rhs`. A single particle is `integrate(sample_uv, y0=y0[np.newaxis])`.
xr.Dataset output via `to_xarray()`.

### 10. Add `to_xarray()` on base class

Each model declares a `state_names` class attribute (e.g.
`("x", "y", "xd", "yd")`). The base class provides `to_xarray(t, Y)`
that wraps solver output into an xr.Dataset using those names.

### 11. Rename `DrifterPhysics` → `DroguedDrifterPhysics`

Unambiguous when multiple models exist. Same for `EOMState` →
`DroguedDrifterState`.

### 12. Empty `__init__.py`

No re-exports. Users import from the module they need:
`from mechanical_drifters.models.drogued_drifter import DroguedDrifter`.

### 13. Simplify `eom.py`

**Lambdification:** M and F lambdified as full Matrix expressions with
CSE — return numpy arrays directly. No upper-triangle decomposition, no
reassembly loops. qdd stays as tuple-of-scalars with CSE (hot path,
numba-compatible).

**Public surface:** Delete `eval_qdd`, `eval_M`, `eval_F`. Exploration
notebooks use `_get_eom_callables` directly:
```python
qdd_raw, M_raw, F_raw, pack_eom_args = _get_eom_callables(model)
args = pack_eom_args(physics, state)
M_raw(*args)   # (n_q, n_q) numpy array
F_raw(*args)   # (n_q, 1) numpy array
```


## Target interfaces

### `base.py`

```python
class LagrangianMechanicsModel:
    # Subclass contract: provide Physics, State, n_q,
    #   _derive_symbolic, _rhs_batch, drift_velocity, state_names

    Physics = None
    State = None
    n_q = None
    state_names = None  # e.g. ("x", "y", "xd", "yd") — used by to_xarray

    def __init__(self, physics, *, backend="numpy"): ...  # physics required

    @property
    def state_size(self): ...

    @property
    def _cache_path(self): ...

    # --- Override these ---
    def _derive_symbolic(self): raise NotImplementedError
    def _rhs_batch(self, Y, sample_uv): raise NotImplementedError
    def drift_velocity(self, Y): raise NotImplementedError  # (N, state_size) -> (N, 2)

    # --- Provided by base ---
    def integrate(
        self, sample_uv, *, t_span=(0, 120), y0=None,
        t_eval=None, atol=1e-3, rtol=1e-3,
    ):
        # y0 in public coords. Models with internal representations
        # override to convert on the way in/out.
        #
        # Always returns (t, Y, max_acceleration):
        #   t: (T,) time array
        #   Y: (T, N, state_size) in public coords
        #   max_acceleration: scalar (max |d(drift_vel)/dt| at final time)
        #
        # t_eval=None: T=1, only the final state.
        # t_eval given: T=len(t_eval), full trajectory.
        # Single particle: N=1.
        ...

    def to_xarray(self, t, Y):
        # Wraps integrate() output into xr.Dataset using self.state_names.
        # t: (T,), Y: (T, N, state_size)
        ...
```

### `models/drogued_drifter.py`

```python
# Coordinate helpers (inlined from former coords.py)
def _uv_to_theta(u, v): ...
def _uv_to_spherical(u, v, ud, vd): ...
def _spherical_to_uv(theta, phi, thetad, phid): ...

class DroguedDrifterPhysics(NamedTuple):
    m_b: float; m_d: float; m_hat_d: float; m_tilde_d: float
    m_tilde_b: float; l: float; g: float; k_b: float; k_d: float

class DroguedDrifterState(NamedTuple):
    u_stereo: ...; v_stereo: ...; xd: ...; yd: ...
    ud_stereo: ...; vd_stereo: ...
    U_b: ...; V_b: ...; U_d: ...; V_d: ...

def drogue_horizontal_added_mass(*, rho, w_d, h_d, C_perp_d=np.pi / 4): ...
def buoy_horizontal_added_mass(*, rho, d_b, h_b, C_perp_b=1.0): ...
def drogue_horizontal_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2): ...
def buoy_horizontal_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0): ...

IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)

class DroguedDrifter(LagrangianMechanicsModel):
    Physics = DroguedDrifterPhysics
    State = DroguedDrifterState
    n_q = 4
    state_names = ("x", "y", "theta", "phi", "xd", "yd", "thetad", "phid")

    def __init__(self, physics, *, backend="numpy", **kwargs): ...
    def _derive_symbolic(self): ...
    def _z_eff(self, u, v): ...
    def _rhs_batch(self, Y, sample_uv): ...
    def drift_velocity(self, Y): ...       # returns Y[:, [IXD, IYD]]
    def _max_depth(self, physics): ...      # optional, for Parcels

    # integrate() overrides base: converts spherical y0 -> stereo on entry,
    # stereo Y_final -> spherical on exit. Caller always sees spherical.
    def integrate(self, sample_uv, *, t_span=(0, 120), y0=None, t_eval=None, atol=1e-3, rtol=1e-3): ...

    # Private coord conversion (clear internal helpers for devs)
    def _to_public_state(self, Y_internal): ...   # (N,8) stereo -> (N,8) spherical
    def _from_public_state(self, Y_public): ...    # (N,8) spherical -> (N,8) stereo
```

### `models/point_surface_drifter.py`

```python
class PointSurfacePhysics(NamedTuple):
    m: float; m_tilde: float; k: float

class PointSurfaceState(NamedTuple):
    xd: ...; yd: ...; U: ...; V: ...

IX, IY, IXD, IYD = range(4)

class PointSurfaceDrifter(LagrangianMechanicsModel):
    Physics = PointSurfacePhysics
    State = PointSurfaceState
    n_q = 2
    state_names = ("x", "y", "xd", "yd")

    def __init__(self, physics, *, backend="numpy"): ...
    def _derive_symbolic(self): ...
    def _rhs_batch(self, Y, sample_uv): ...
    def drift_velocity(self, Y): ...    # returns Y[:, [IXD, IYD]]
    # No integrate() override needed — internal = public coords
```

### `parcels.py`

```python
def _make_profile_sampler(depth_levels, U_profiles, V_profiles): ...  # from velocity.py
def _extract_profiles(particles, fieldset, max_depth): ...
def _position_update(particles, xd_ms, yd_ms, fieldset): ...

def make_kernel(model):
    # Parcels kernel: integrate, extract final drift velocity, update position.
    # t, Y, _ = model.integrate(sample_uv)  # T=1
    # drift_vel = model.drift_velocity(Y[-1])
    max_depth_fn = getattr(model, '_max_depth', None)
    max_depth = max_depth_fn(model.physics) if max_depth_fn else 0.0
    ...
```

### `eom.py`

```python
def _build_packer(raw_func, physics_type, state_type): ...
def _cache_key(derive_fn): ...
def _load_or_derive(model): ...
def _make_qdd_func(model, backend="numpy"): ...

def _get_eom_callables(model):
    # Returns (qdd_raw, M_raw, F_raw, pack_eom_args)
    # M_raw: lambdified full Matrix, returns (n_q, n_q) array
    # F_raw: lambdified full Matrix, returns (n_q, 1) array
    # qdd_raw: lambdified tuple of scalars with CSE (hot path)
    ...
```

### `stokes.py` (unchanged)

```python
def compute_stokes_profile(surface_u, surface_v, peak_period, depth_levels, g=None): ...
```

### `__init__.py`

```python
# Empty. Users import from the module they need.
```


## Execution order

Changes are ordered to keep tests passing at each step.

1. **Inline coords.py into drogued_drifter.py.** Copy the three functions,
   update all imports in tests/examples/docs, delete `coords.py`.

2. **Inline velocity.py into parcels.py.** Move `make_profile_sampler` as
   `_make_profile_sampler`, update imports, delete `velocity.py`.

3. **Rename Physics/State types.** `DrifterPhysics` →
   `DroguedDrifterPhysics`, `EOMState` → `DroguedDrifterState`. Update all
   references. (The eom.py `_build_packer` and `_cache_key` use NamedTuple
   field names, not class names, so they don't need changes.)

4. **Add `drift_velocity()` method** to both models. Then rewrite
   `steady_state_batch` to call it instead of indexing
   `_drift_velocity_indices`. Then remove `_drift_velocity_indices` from the
   base class contract and both models.

5. **Rename `steady_state_batch` → `integrate`.** Update all callers
   (parcels.py, tests, notebooks, docs).

6. **Make `integrate` public-coords-in/out for DroguedDrifter.** Override
   `integrate` to convert spherical→stereo on entry, stereo→spherical on
   exit. Add `_to_public_state`, `_from_public_state` as private helpers.
   Delete `get_final_drift`, `get_final_drift_batch`, `get_full_solution`,
   `_rhs`. All callers use `integrate` with N=1 for single particles.

7. **Drop `default_physics` from base class contract.** Make `physics` a
   required arg on the base `__init__`. Each model's own `__init__` handles
   its own defaults.

8. **Add `to_xarray()` on base class** and `state_names` class attribute.
   Update notebooks that used `get_full_solution` to use
   `model.to_xarray(...)`.

9. **Remove `make_kernel` from base class** and `_max_depth` from the base
   class contract. Update `parcels.py` to use `getattr`.

10. **Simplify eom.py.** Lambdify M and F as full Matrix expressions (no
    upper-triangle decomposition). Delete `eval_qdd`, `eval_M`, `eval_F`.
    Update the exploration notebook to use `_get_eom_callables` directly.

11. **Empty `__init__.py`.** Remove all re-exports. Update imports
    throughout (tests, notebooks, docs, README).

12. **Update docs.** Rewrite `docs/drifter-model.md`,
    `docs/point-surface-drifter.md`, `docs/architecture.md`,
    `docs/parcels-v4-coupling.md`.

13. **Run all tests and notebooks.** Fix any breakage.


## What stays the same

- `eom.py` internal pipeline (caching, lambdification, packer)
- `stokes.py` (standalone utility, no coupling issues)
- `_derive_symbolic` pattern (sympy derivation is the core value)
- `_rhs_batch` as the central ODE RHS contract
- `parcels.py` as the Parcels adapter module
- `make_kernel(model)` as the Parcels entry point
