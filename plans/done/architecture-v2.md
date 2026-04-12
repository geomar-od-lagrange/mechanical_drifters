# Architecture v2: function-first, Parcels-optimized

Alternative architecture proposal for the `drogued_drifters` package.

## Design principles

1. **Functions, not methods.** The core computation is a pure function
   from (physics, velocity_profile) to (drift_velocity). There is no
   reason for it to live on a class. The class exists to hold defaults
   and wire things together -- make it a thin convenience layer, not the
   load-bearing abstraction.

2. **One velocity protocol.** A velocity profile is `(z_levels, U, V)` --
   three arrays. Not a callback, not a closure, not a protocol with
   scalar/batch modes. Functions that need velocity at arbitrary z
   interpolate from these arrays. The interpolation is a utility, not a
   protocol.

3. **Parcels integration is the primary path.** The standalone
   single-particle path (`get_full_solution`) is a teaching/debugging
   tool. The batch steady-state solver is the production workhorse.
   Optimize the architecture for the batch path.

4. **Backend is a parameter, not architecture.** numpy vs numba changes
   one function pointer inside the EOM evaluator. It should not ripple
   through module boundaries, class interfaces, or factory functions.

5. **No closures as data.** `make_profile_sampler` currently returns a
   closure that captures arrays. Instead, pass the arrays through the
   call chain. Functions are cheaper to inspect, test, and debug than
   closures.

## Module structure

```
src/drogued_drifters/
    __init__.py           Public API re-exports
    eom.py                Symbolic derivation, caching, lambdification, EOM evaluation
    coords.py             Stereographic <-> spherical coordinate transforms
    physics.py            DrifterPhysics, drag/added-mass helpers, z_eff
    solve.py              ODE integration: batch steady-state, single-particle trajectory
    velocity.py           Profile interpolation, Stokes drift
    parcels.py            Parcels-specific: profile extraction, position update, kernel
    data/eom_cache.pkl    Cached symbolic derivation output
```

### Why these boundaries

The current `lagrange_model.py` mixes three concerns: symbolic
derivation, numeric evaluation, and coordinate transforms. The current
`drifter.py` mixes four: physics constants, drag parameterization, ODE
integration, and the velocity adapter. Splitting along concern boundaries
makes each module independently testable and independently
understandable.

The key split:

- **eom.py** owns everything from sympy to callable `qdd(physics, state)`.
  This is the most complex module but has the narrowest interface: it
  exports exactly three evaluation functions and the derivation
  machinery.

- **solve.py** owns the ODE integration. It imports from eom and
  physics but knows nothing about Parcels or velocity sources. Its
  functions take velocity arrays as explicit arguments.

- **velocity.py** owns all velocity-related utilities: profile
  interpolation and Stokes drift. Currently `make_profile_sampler` lives
  in `parcels_v4.py` but it has nothing to do with Parcels -- it is a
  generic depth interpolator. Moving Stokes drift here too groups the
  two main velocity-profile building blocks.

- **parcels.py** is the only module that imports parcels. It knows how
  to extract profiles from a fieldset and how to update particle
  positions. Everything else it delegates to solve.py and velocity.py.

## Detailed design

### `eom.py` -- Equations of motion

#### Keeps from `lagrange_model.py`

- `EOMState` (NamedTuple, 10 fields) -- unchanged
- `_derive_symbolic()` -- unchanged
- `_load_or_derive()` -- unchanged
- `_get_eom_callables()` -- unchanged
- `_build_packer()` -- unchanged
- `_cache_key()`, `_CACHE_PATH` -- unchanged
- `_sym_norm()` -- unchanged

#### Changes

**`_make_qdd_func(backend)` becomes the single factory.** It already
supports both backends. No change to its implementation, but it becomes
the only way to get a qdd evaluator.

**Delete `_qdd_func`.** The module-level convenience wrapper that
hardcodes `"numpy"` is dead weight now that `_make_qdd_func` handles
both backends. All callers pass the backend explicitly.

**Rename public functions for clarity:**

```python
def eval_qdd(physics: DrifterPhysics, state: EOMState, *, backend="numpy"):
    """Evaluate generalized accelerations qdd = M^{-1}F.

    Returns (4,) for scalar input, (N, 4) for batch.
    """
    return _make_qdd_func(backend)(physics, state)

def eval_M(physics: DrifterPhysics, state: EOMState):
    """Evaluate 4x4 mass matrix. Returns (4,4) or (N,4,4)."""
    ...  # current M_func body

def eval_F(physics: DrifterPhysics, state: EOMState):
    """Evaluate 4-element force vector. Returns (4,) or (N,4)."""
    ...  # current F_func body
```

The old names `qdd_func`, `M_func`, `F_func` become deprecated aliases
for one release, then removed.

**Rationale for `eval_*` naming:** `M_func` reads as "M function" which
is a tautology. `eval_M` reads as "evaluate M" which is an action.
Also, `eval_qdd(..., backend="numba")` makes the backend parameter
natural, while `qdd_func(..., backend="numba")` is awkward because the
function is already the func.

### `coords.py` -- Coordinate transforms

Extracts from `lagrange_model.py`:

```python
def uv_to_theta(u, v):
    """Stereographic (u, v) -> polar angle theta."""

def uv_to_spherical(u, v, ud, vd):
    """Stereographic -> (theta, phi, thetad, phid)."""

def spherical_to_uv(theta, phi, thetad, phid):
    """Spherical -> stereographic (u, v, ud, vd)."""
```

**Drop the underscore prefix.** These are not internal implementation
details -- they are the coordinate system of the model. Tests already
use them. They should be importable as public utilities.

**Keep `uv_to_theta`.** It is called by `uv_to_spherical` and is
independently useful for `z_eff` (extracting only the pole angle without
computing the azimuthal direction). The split is justified.

### `physics.py` -- Physical parameters and geometry

```python
from typing import NamedTuple

class DrifterPhysics(NamedTuple):
    """Physical constants for a drogued drifter -- frozen, set once."""
    m_b: float       # buoy dry mass [kg]
    m_d: float       # drogue dry mass [kg]
    m_hat_d: float   # drogue buoyancy correction [kg]
    m_tilde_d: float # drogue added mass [kg]
    m_tilde_b: float # buoy added mass [kg]
    l: float         # pole length [m]
    g: float         # gravitational acceleration [m/s^2]
    k_b: float       # buoy drag coefficient [kg/m]
    k_d: float       # drogue drag coefficient [kg/m]

# Callies et al. defaults at rho=1025 kg/m^3
DEFAULT_PHYSICS = DrifterPhysics(
    m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
    l=3.0, g=9.81, k_b=12.0, k_d=154.0,
)

def drogue_horizontal_added_mass(*, rho, w_d, h_d, C_perp_d=np.pi / 4):
    ...

def buoy_horizontal_added_mass(*, rho, d_b, h_b, C_perp_b=1.0):
    ...

def drogue_horizontal_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2):
    ...

def buoy_horizontal_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0):
    ...

def z_eff(physics: DrifterPhysics, u, v):
    """Effective drogue depth from stereographic coordinates.

    At equilibrium (u=v=0) returns -l. Clamped to <= 0.
    """
    s = u**2 + v**2
    cos_theta = (s - 4) / (s + 4)
    return np.minimum(0.0, physics.l * cos_theta)
```

**`z_eff` moves from `DroguedDrifter._z_eff` to a free function.**
It depends only on physics.l and the stereographic coordinates. No
reason for it to be a method.

**`DEFAULT_PHYSICS` is canonical.** Tests and examples use this instead
of duplicating the NamedTuple construction. The `DroguedDrifter` class
defaults to `DEFAULT_PHYSICS`.

### `velocity.py` -- Velocity profile utilities

```python
def interpolate_profile(z, depth_levels, U_profiles, V_profiles):
    """Linear interpolation in z from pre-sampled velocity profiles.

    Args:
        z: Depth query, scalar or (N,) array, z-up (0=surface, negative=below).
        depth_levels: (D,) sorted ascending (deepest first), z-up.
        U_profiles: (D, N) eastward velocity at each depth for each particle.
        V_profiles: (D, N) northward velocity, same shape.

    Returns:
        (U, V) each shape (N,).
    """
    ...  # Current make_profile_sampler body, but as a function not closure

def compute_stokes_profile(surface_u, surface_v, peak_period, depth_levels, g=None):
    """Deep-water exponential Stokes drift profile.

    Returns (stokes_u, stokes_v) of shape (D, ...).
    """
    ...  # Current stokes.py body, unchanged
```

**`make_profile_sampler` is replaced by `interpolate_profile`.** Instead
of a closure factory that returns a callable, this is a pure function.
The caller passes the profile arrays explicitly. This means the arrays
flow through the call chain as data, not captured in a closure.

**Impact on the ODE integrator:** The batch RHS needs to interpolate
velocity at buoy (z=0) and drogue (z=z_eff) depths. Instead of calling
a closure `sample_uv(z)`, it calls
`interpolate_profile(z, depth_levels, U, V)`. The profile arrays are
passed as parameters to the RHS function, not captured in instance
state.

**Impact on Parcels:** `_extract_profiles` currently returns a
`sample_uv` closure. It now returns `(depth_levels, U_profiles,
V_profiles)` -- plain arrays. The kernel passes these to the solver.

**Stokes drift moves here** from `stokes.py`. This groups all
velocity-profile building blocks in one place: compute a Stokes profile,
then add it to Eulerian profiles, then interpolate the combined profile
at arbitrary depths. One import for the whole chain.

### `solve.py` -- ODE integration

This is the radical restructuring: the solver functions take velocity
profile arrays as explicit arguments, not callbacks.

```python
# State vector layout
IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)

def steady_state_drift(
    physics: DrifterPhysics,
    depth_levels: np.ndarray,    # (D,)
    U_profiles: np.ndarray,      # (D, N)
    V_profiles: np.ndarray,      # (D, N)
    *,
    t_span=(0, 120),
    y0=None,                     # (N, 8) internal stereographic, or None
    atol=1e-3,
    rtol=1e-3,
    backend="numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute steady-state buoy drift for N particles.

    Args:
        physics: Drifter physical parameters.
        depth_levels: Vertical positions (D,), z-up, sorted ascending.
        U_profiles, V_profiles: Velocity profiles (D, N).
        t_span: Integration window [s].
        y0: Initial state (N, 8) in internal coords, or None for cold start.
        atol, rtol: ODE solver tolerances.
        backend: "numpy" or "numba".

    Returns:
        (xd_final, yd_final, Y_final, max_accel)
        xd_final, yd_final: (N,) drift velocities [m/s].
        Y_final: (N, 8) final state in internal coords (pass back as y0
            for warm-starting).
        max_accel: Scalar convergence diagnostic.
    """
```

**The public/internal coordinate split is gone.** `y0` and `Y_final`
are in internal stereographic coordinates. The spherical round-trip
(theta/phi <-> u/v on every call) was pure overhead for the Parcels
path, which never displays angles. Code that needs angles (the
single-particle trajectory viewer) converts explicitly using
`coords.uv_to_spherical`.

```python
def full_trajectory(
    physics: DrifterPhysics,
    depth_levels: np.ndarray,    # (D,)
    U_profile: np.ndarray,       # (D,) -- single particle
    V_profile: np.ndarray,       # (D,)
    *,
    t_span,
    y0=None,                     # (8,) internal, or None
    t_eval=None,
    atol=1e-3,
    rtol=1e-3,
    backend="numpy",
) -> xr.Dataset:
    """Integrate single-particle trajectory over t_span.

    Returns xr.Dataset with time coordinate and variables:
    x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo.

    For display, convert angles with:
        theta, phi, thetad, phid = coords.uv_to_spherical(
            ds.u_stereo, ds.v_stereo, ds.ud_stereo, ds.vd_stereo)
    """
```

**Why internal coords in the xr.Dataset?** The current public API
converts to spherical angles for the Dataset, hiding the actual
representation. This makes round-tripping lossy and forces everyone
through the conversion. Users studying the EOM want the native
coordinates; users wanting theta/phi call a one-liner. Internal coords
also make warm-starting trivial: just pass `ds.isel(time=-1).values`.

```python
def _rhs_batch(
    Y: np.ndarray,               # (N, 8)
    physics: DrifterPhysics,
    depth_levels: np.ndarray,
    U_profiles: np.ndarray,
    V_profiles: np.ndarray,
    qdd_func,                    # callable(physics, state) -> (N, 4)
) -> np.ndarray:
    """Vectorized RHS for N particles.

    All parameters are explicit. No instance state.
    """
```

**The RHS is a free function with all dependencies as parameters.** No
`self._sample_uv`, no `self._qdd_func`, no `self.physics`. This is the
core gain: the hot path is a pure function from (state, physics,
velocity) to (derivatives). It can be tested with synthetic inputs, JIT-
compiled, or called from any context.

### `parcels.py` -- Parcels coupling

```python
_DEG2M = 1852.0 * 60.0

def extract_profiles(particles, fieldset, drogue_depth):
    """Sample velocity profiles from fieldset at particle positions.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet with UV VectorField.
        drogue_depth: Pole length [m] (determines depth extent).

    Returns:
        (depth_levels, U_profiles, V_profiles)
        depth_levels: (D,) z-up ascending.
        U_profiles: (D, N) in m/s.
        V_profiles: (D, N) in m/s.
    """

def position_update(particles, xd_ms, yd_ms, fieldset):
    """Euler-forward position update. Mutates particles.dlon, dlat."""

def make_dd_kernel(physics=None, *, backend="numpy"):
    """Create a Parcels kernel for drogued-drifter advection.

    Args:
        physics: DrifterPhysics instance, or None for defaults.
        backend: "numpy" or "numba".

    Returns:
        Kernel function(particles, fieldset) for pset.execute().
    """
    if physics is None:
        physics = DEFAULT_PHYSICS

    qdd_fn = _make_qdd_func(backend)

    def _kernel(particles, fieldset):
        depth_levels, U, V = extract_profiles(
            particles, fieldset, physics.l
        )
        xd, yd, _, _ = steady_state_drift(
            physics, depth_levels, U, V,
            backend=backend,
        )
        position_update(particles, xd, yd, fieldset)

    return _kernel
```

**`DroguedDrifter` is not needed for the Parcels path.** The kernel
factory takes `physics` (a NamedTuple) and `backend` (a string)
directly. No class construction, no method calls, no instance state.
The kernel closure captures only immutable data (physics constants,
backend string).

**`extract_profiles` returns arrays, not a closure.** The profile data
flows as plain arrays through `steady_state_drift`, which passes them
to `_rhs_batch`, which passes them to `interpolate_profile`. No
closures in the data path.

**`extract_profiles` takes `drogue_depth` instead of `dd`.** It no
longer needs a `DroguedDrifter` instance -- just the pole length to
know how deep to sample.

### `DroguedDrifter` class -- thin convenience wrapper

```python
class DroguedDrifter:
    """Convenience wrapper for interactive use and single-particle studies.

    For Parcels integration, use make_dd_kernel() directly.
    """

    def __init__(self, physics=None, *, backend="numpy"):
        self.physics = physics or DEFAULT_PHYSICS
        self.backend = backend

    def full_trajectory(self, depth_levels, U_profile, V_profile, **kw):
        """Delegate to solve.full_trajectory."""
        return full_trajectory(
            self.physics, depth_levels, U_profile, V_profile,
            backend=self.backend, **kw,
        )

    def steady_state(self, depth_levels, U_profiles, V_profiles, **kw):
        """Delegate to solve.steady_state_drift."""
        return steady_state_drift(
            self.physics, depth_levels, U_profiles, V_profiles,
            backend=self.backend, **kw,
        )
```

**The class is 15 lines.** It holds physics + backend and delegates
to free functions. No velocity callbacks, no adapters, no `_sample_uv`,
no `_rhs`, no `_solve`, no `_z_eff`. All of those are now standalone
functions that the class simply calls.

**No `get_uv` / `sample_uv` parameters.** Velocity is passed as
arrays to each method call, not stored on the instance. This eliminates
the adapter, the dual-protocol confusion, and the save/restore hack in
`get_final_drift_batch`.

**Backward compatibility:** The old constructor signature
(`m_b=, m_d=, ..., get_uv=, sample_uv=`) can be supported via a
transitional `__init__` that builds `DrifterPhysics` from kwargs and
warns about deprecation. But for pre-alpha, just break it.

### `__init__.py` -- Public API

```python
from .physics import (
    DrifterPhysics,
    DEFAULT_PHYSICS,
    drogue_horizontal_added_mass,
    buoy_horizontal_added_mass,
    drogue_horizontal_drag_coeff,
    buoy_horizontal_drag_coeff,
    z_eff,
)
from .eom import EOMState, eval_qdd, eval_M, eval_F
from .coords import uv_to_theta, uv_to_spherical, spherical_to_uv
from .solve import steady_state_drift, full_trajectory
from .velocity import interpolate_profile, compute_stokes_profile
from .parcels import make_dd_kernel

# Convenience class (optional import)
from .solve import DroguedDrifter
```

**Everything useful is a top-level import.** No more
`from drogued_drifters.drifter import DroguedDrifter` or
`from drogued_drifters.stokes import compute_stokes_profile`.

## Data flow diagrams

### Parcels integration (production path)

```
pset.execute(kernel)
  |
  v
_kernel(particles, fieldset)
  |
  +-- extract_profiles(particles, fieldset, physics.l)
  |     -> (depth_levels, U_profiles, V_profiles)        # plain arrays
  |
  +-- steady_state_drift(physics, depth_levels, U, V, backend=backend)
  |     |
  |     +-- _make_qdd_func(backend)  -> qdd_fn            # cached
  |     +-- _rhs_batch(Y, physics, depth_levels, U, V, qdd_fn)
  |     |     |
  |     |     +-- z_eff(physics, u, v)  -> z_d             # drogue depth
  |     |     +-- interpolate_profile(0, ...)  -> U_b, V_b # buoy velocity
  |     |     +-- interpolate_profile(z_d, ...)-> U_d, V_d # drogue velocity
  |     |     +-- qdd_fn(physics, state) -> accelerations
  |     |     `-- pack derivatives -> dY
  |     `-- solve_ivp(rhs_flat, ...)
  |
  +-- position_update(particles, xd, yd, fieldset)
```

Every piece of data flows as an explicit argument. No closures, no
instance state, no monkey-patching. The `qdd_fn` is the only function
pointer, and it comes from a cached factory.

### Standalone use (teaching/exploration)

```python
from drogued_drifters import (
    DrifterPhysics, steady_state_drift, interpolate_profile,
    compute_stokes_profile,
)

# Build velocity profile
z = np.linspace(-20, 0, 50)
U = 0.5 * np.exp(z / 5.0)
V = np.zeros_like(U)

# Add Stokes drift
Us, Vs = compute_stokes_profile(0.1, 0.0, 8.0, z)
U = U + Us[:, 0]  # single point, squeeze

# Compute drift
physics = DrifterPhysics(...)
xd, yd, _, _ = steady_state_drift(
    physics,
    depth_levels=z,
    U_profiles=U[:, np.newaxis],  # (D, 1)
    V_profiles=V[:, np.newaxis],
)
```

### Direct EOM study

```python
from drogued_drifters import (
    DrifterPhysics, EOMState, eval_qdd, eval_M, eval_F,
)

physics = DrifterPhysics(...)
state = EOMState(u_stereo=0.1, v_stereo=0.0, ...)

qdd = eval_qdd(physics, state)
M = eval_M(physics, state)
F = eval_F(physics, state)
assert np.allclose(M @ qdd, F)
```

## Backend handling

Backend selection is simple: `_make_qdd_func(backend)` is LRU-cached
and returns a callable. The callable is passed through the call chain
as a function argument:

```
steady_state_drift(..., backend="numba")
  -> qdd_fn = _make_qdd_func("numba")   # cached, JIT warmup on first call
  -> _rhs_batch(Y, ..., qdd_fn)          # qdd_fn is just a parameter
```

No instance attribute mutation, no factory-in-a-factory, no closure
capturing a mutable reference. The backend string flows in, a function
pointer flows out, and that pointer is used until the call returns.

For the Parcels kernel, the backend is captured in the closure at
`make_dd_kernel` time:

```python
def make_dd_kernel(physics=None, *, backend="numpy"):
    physics = physics or DEFAULT_PHYSICS
    def _kernel(particles, fieldset):
        ...
        steady_state_drift(physics, ..., backend=backend)
    return _kernel
```

## What is deleted

| Current | Disposition |
|---------|------------|
| `_adapt_get_uv` | Deleted. No adapter needed -- velocity is arrays. |
| `_qdd_func` (module-level) | Deleted. Use `_make_qdd_func(backend)` directly. |
| `DroguedDrifter._rhs` | Deleted. Replaced by `_rhs_single` in solve.py (or just use `_rhs_batch` with N=1). |
| `DroguedDrifter._rhs_batch` | Extracted to `solve._rhs_batch`, a free function. |
| `DroguedDrifter._z_eff` | Extracted to `physics.z_eff`. |
| `DroguedDrifter._solve` | Inlined into `full_trajectory`. |
| `DroguedDrifter._default_uv` | Deleted. No default velocity -- pass arrays explicitly. |
| `DroguedDrifter._default_sample_uv` | Deleted. Same reason. |
| `DroguedDrifter.get_uv` | Deleted. No velocity callbacks. |
| `DroguedDrifter._sample_uv` | Deleted. No velocity callbacks. |
| `DroguedDrifter._get_final_drift_batch_impl` | Absorbed into `solve.steady_state_drift`. |
| `make_profile_sampler` | Replaced by `velocity.interpolate_profile`. |
| `stokes.py` (module) | Contents move to `velocity.py`. Module deleted. |
| `drifter.py` (module) | Split into `physics.py` and `solve.py`. Module deleted. |
| `lagrange_model.py` (module) | Split into `eom.py` and `coords.py`. Module deleted. |

## What stays

| Current | New location | Changes |
|---------|-------------|---------|
| `DrifterPhysics` | `physics.py` | Add `DEFAULT_PHYSICS` constant |
| `EOMState` | `eom.py` | Unchanged |
| `_derive_symbolic` | `eom.py` | Unchanged |
| `_load_or_derive` | `eom.py` | Unchanged |
| `_get_eom_callables` | `eom.py` | Unchanged |
| `_build_packer` | `eom.py` | Unchanged |
| `_make_qdd_func` | `eom.py` | Unchanged |
| `M_func` -> `eval_M` | `eom.py` | Rename only |
| `F_func` -> `eval_F` | `eom.py` | Rename only |
| `qdd_func` -> `eval_qdd` | `eom.py` | Add backend kwarg |
| `_uv_to_theta` | `coords.py` | Drop underscore |
| `_uv_to_spherical` | `coords.py` | Drop underscore |
| `_spherical_to_uv` | `coords.py` | Drop underscore |
| Drag/added-mass helpers | `physics.py` | Unchanged |
| `compute_stokes_profile` | `velocity.py` | Unchanged |
| `_extract_profiles` | `parcels.py` as `extract_profiles` | Returns arrays instead of closure |
| `_position_update` | `parcels.py` as `position_update` | Unchanged |
| `DDAdvectEE` | Inlined into kernel closure | 3 lines |
| `make_dd_kernel` | `parcels.py` | Takes physics+backend, not dd instance |

## Migration path

This is pre-alpha. No deprecation period needed.

1. **Create `coords.py`, `physics.py`, `velocity.py`.** Move code,
   adjust imports in tests. Run tests. These are pure moves.

2. **Create `eom.py`.** Move from `lagrange_model.py`, rename public
   functions. Update `__init__.py`. Run tests.

3. **Create `solve.py`.** Rewrite ODE integration as free functions
   taking explicit velocity arrays. This is the biggest change. Port
   existing tests to the new signatures.

4. **Rewrite `parcels.py`.** New `make_dd_kernel` signature. Update
   Parcels tests.

5. **Shrink `DroguedDrifter`.** Make it a 15-line wrapper in
   `solve.py`. Update example notebooks.

6. **Delete old modules.** Remove `lagrange_model.py`, `drifter.py`,
   `stokes.py`.

Steps 1-2 can be done with purely mechanical moves. Step 3 is the
substantive refactor. Steps 4-5 follow from 3. Step 6 is cleanup.

## Trade-offs and risks

**Loss of the scalar `get_uv` convenience.** Users who want to pass
a function `f(z) -> (U, V)` must evaluate it on a depth grid first.
This is one extra line in notebooks (`U = f(z)`) but removes an
entire protocol (callbacks, adapters, scalar/batch modes).

**`interpolate_profile` called per RHS evaluation.** Currently the
closure `sample_uv` captures the arrays once and the searchsorted
runs on cached references. The free function receives the arrays as
arguments each call. The overhead is negligible (numpy array passing
is a pointer copy), but worth verifying there is no measurable
regression.

**Larger import surface.** Six modules instead of four. But each is
smaller and self-contained. Actual cognitive load decreases because
you can understand `coords.py` without understanding `eom.py`.

**Breaking all existing imports.** Every `from drogued_drifters.drifter
import DroguedDrifter` breaks. For pre-alpha research code this is
acceptable. The notebooks are co-located and easy to update.
