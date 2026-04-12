# Revised architecture for `drogued_drifters`

Addresses maintainer feedback from [api-review.md](api-review.md) and
findings from [review-architecture.md](review-architecture.md). The goal
is the simplest structure that supports three use cases: direct EOM
study, standalone drift computation, and Parcels integration.

## Design principles

1. **Functions over classes.** The core computation is stateless: physics
   parameters in, drift velocity out. `DroguedDrifter` becomes a thin
   convenience shell, not the primary interface.

2. **One velocity protocol.** `sample_uv(z) -> (U, V)` everywhere.
   No adapters, no `get_uv`, no dual interface.

3. **Clean parameter plumbing.** Backend choice is a parameter passed
   through the call chain. No monkey-patching, no closure tricks, no
   mutable instance attributes for dispatch.

4. **Parcels code sees only Parcels concerns.** Profile extraction,
   depth convention, position update. No physics internals.

5. **c-order `(t, z, y, x)` for positional args** where applicable.

## Module map (after)

```
src/drogued_drifters/
  __init__.py            Public API re-exports
  eom.py                 Symbolic derivation, caching, lambdification, EOM evaluation
  coords.py              Stereographic <-> spherical coordinate transforms
  drifter.py             DroguedDrifter class, ODE integration, parameterization helpers
  velocity.py            Velocity profile interpolation (make_profile_sampler)
  parcels_v4.py          Parcels-specific: profile extraction, position update, kernel
  stokes.py              Stokes drift profiles (unchanged)
  data/eom_cache.pkl     Cached derivation (unchanged)
```

### What moves where

| Current location | Destination | What |
|---|---|---|
| `lagrange_model.py` (all) | `eom.py` | Symbolic derivation, caching, `DrifterPhysics`, `EOMState`, `_build_packer`, `_make_qdd_func`, `M_func`, `F_func`, `qdd_func` |
| `lagrange_model.py` | `coords.py` | `_uv_to_theta`, `_uv_to_spherical`, `_spherical_to_uv` |
| `parcels_v4.py` | `velocity.py` | `make_profile_sampler` |
| `drifter.py` | `drifter.py` | `DroguedDrifter`, drag/added-mass helpers, ODE integration |

The rename from `lagrange_model.py` to `eom.py` reflects its actual
role: it provides numeric evaluation of the equations of motion, not the
symbolic model itself (which is a cached artifact). The name
`lagrange_model` suggests the sympy derivation is the module's purpose;
`eom` says "here are the callable equations of motion."

Coordinate transforms go to `coords.py` because they are pure math
utilities used by both `drifter.py` and `parcels_v4.py`, and have
nothing to do with EOM evaluation.

`make_profile_sampler` moves to `velocity.py` because it is a
general-purpose depth interpolation utility. It is useful standalone
(the integration tests already use it without Parcels) and should not
live in a Parcels-specific module. The Parcels module imports it.

---

## Module details

### `eom.py` --- equations of motion

Everything currently in `lagrange_model.py` minus the coordinate
transforms.

#### What stays

- `DrifterPhysics` (NamedTuple, 9 fields) -- unchanged
- `EOMState` (NamedTuple, 10 fields) -- unchanged
- `_derive_symbolic()` -- unchanged
- `_load_or_derive()` -- unchanged
- `_get_eom_callables()` -- unchanged
- `_build_packer()` -- unchanged
- `_sym_norm()` -- unchanged
- `_cache_key()` -- unchanged
- `_CACHE_PATH` -- unchanged

#### What changes

**`_make_qdd_func(backend)` stays but becomes the sole factory.** It
already handles both backends correctly. No changes to its
implementation.

**Delete `_qdd_func` (the module-level convenience wrapper).** It calls
`_make_qdd_func("numpy")` on every invocation, creating a new function
each time (the LRU cache is on `_get_eom_callables`, not on
`_make_qdd_func`). Callers should use `_make_qdd_func("numpy")` or the
public `qdd_func` instead.

**`qdd_func` becomes a thin public wrapper:**

```python
def qdd_func(physics: DrifterPhysics, state: EOMState, *, backend="numpy"):
    """Evaluate generalized accelerations qdd = M^{-1}F.

    Public entry point for direct EOM evaluation. See M_func and F_func
    for the mass matrix and force vector separately.

    Args:
        physics: DrifterPhysics instance.
        state: EOMState instance (scalar or batch).
        backend: "numpy" (default) or "numba".

    Returns:
        (4,) array for scalar input, (N, 4) for batch.
    """
    return _make_qdd_func(backend)(physics, state)
```

This wires `qdd_func` into the backend story. Currently `qdd_func` is
exported but hardcoded to numpy. Adding `backend=` makes it usable for
both backends without needing `DroguedDrifter`.

**`M_func` and `F_func` stay unchanged.** They are numpy-only, which is
fine -- the mass matrix and force vector are diagnostic tools, not hot
path. No need for a numba path.

#### Why keep `_build_packer`

`_build_packer` exists because the lambdified functions have 19
positional parameters whose order is determined by sympy, not by us. The
packer inspects the lambda signature and maps `DrifterPhysics` +
`EOMState` field names to positional slots. This prevents silent
mis-ordering of 19 arguments. The alternative -- manually maintaining a
positional tuple of 19 values -- is strictly worse. Keep it.

#### Why keep `_make_qdd_func` as a factory

The factory pattern is justified because:

1. The numba path needs to JIT-compile once, then reuse the compiled
   function. `_make_qdd_func("numba")` does this (compile + warmup),
   returning a closure that calls the compiled function. Calling it
   again gets the same compiled function via the LRU cache on
   `_get_eom_callables`.

2. The returned `qdd_func(physics, state)` signature is what the ODE
   RHS needs. The factory converts from raw lambdified args to
   struct-based args once, rather than on every call.

The factory should be LRU-cached on `backend` so repeated calls with the
same backend return the same function object:

```python
@functools.lru_cache()
def _make_qdd_func(backend="numpy"):
    ...
```

This fixes the current waste where `qdd_func` and `_qdd_func` both call
`_make_qdd_func("numpy")` every time, each creating a new closure.

---

### `coords.py` --- coordinate transforms

Three functions, moved verbatim from `lagrange_model.py`:

```python
def _uv_to_theta(u, v) -> theta
def _uv_to_spherical(u, v, ud, vd) -> (theta, phi, thetad, phid)
def _spherical_to_uv(theta, phi, thetad, phid) -> (u, v, ud, vd)
```

#### Is `_uv_to_theta` justified?

Yes. `_uv_to_theta` is used in two places:

1. Inside `_uv_to_spherical` (which calls it for the theta component).
2. In tests that only need theta (4 tests in
   `test_numerical_edge_cases.py`).

It is not used in the hot path independent of `_uv_to_spherical`. But
the split is harmless -- `_uv_to_theta` is 4 lines, and having it
separate makes the test code cleaner. Keep as-is.

All three functions stay private (underscore prefix). They are internal
plumbing for converting between the public spherical API and the
internal stereographic representation.

---

### `velocity.py` --- velocity profile interpolation

```python
def make_profile_sampler(depth_levels, U_profiles, V_profiles):
    """Build a sample_uv(z) interpolator from pre-sampled velocity profiles.

    Args:
        depth_levels: (D,) array, z-up ascending (deepest first, e.g. [-20, -10, 0]).
        U_profiles: (D, N) eastward velocity at each depth for each particle.
        V_profiles: (D, N) northward velocity, same shape.

    Returns:
        Callable sample_uv(z) -> (U, V) where z is scalar or (N,) array.
    """
```

Moved verbatim from `parcels_v4.py`. The defensive `.copy()` (from
release-polish S2) should be applied here.

This module is a natural home for future velocity-related utilities
(e.g., analytical profile builders for testing, or a Stokes+current
composite sampler).

---

### `drifter.py` --- ODE integration and DroguedDrifter

#### Drag / added-mass helpers (stay, unchanged)

```python
def drogue_horizontal_added_mass(*, rho, w_d, h_d, C_perp_d=pi/4) -> float
def buoy_horizontal_added_mass(*, rho, d_b, h_b, C_perp_b=1.0) -> float
def drogue_horizontal_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2) -> float
def buoy_horizontal_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0) -> float
```

These are public, self-contained, and well-tested. Keep as-is. They will
be exercised in a new example notebook (see "Examples" section below).

#### State vector layout (stays)

```python
IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)
```

#### What gets deleted

- **`_adapt_get_uv`** -- gone entirely. The dual `get_uv`/`sample_uv`
  interface is eliminated.

- **`_default_uv` (instance method)** -- gone. Only
  `_default_sample_uv` survives.

- **`get_uv` constructor parameter** -- gone.

#### Default velocity sampler

The `_default_sample_uv` static method stays for testing convenience.
It becomes a module-level function so it can be used without
instantiating `DroguedDrifter`:

```python
def _default_sample_uv(z):
    """Step-function velocity for testing: +1 m/s at surface, -1 m/s below."""
    z_arr = np.asarray(z, dtype=float)
    scalar = z_arr.ndim == 0
    z_arr = np.atleast_1d(z_arr)
    U = np.where(z_arr == 0.0, 1.0, -1.0)
    V = np.where(z_arr == 0.0, 1.0, -1.0)
    if scalar:
        return float(U[0]), float(V[0])
    return U, V
```

#### `DroguedDrifter` class (rewired)

The class stays but becomes a thin shell: it owns `DrifterPhysics`, a
`sample_uv` callable, and a backend choice. All computation is done by
standalone internal functions.

```python
class DroguedDrifter:
    """Drogued drifter simulator.

    Thin wrapper around the EOM solver and ODE integrator. Owns the
    physical parameters, velocity sampler, and backend choice.
    """

    def __init__(
        self,
        *,
        m_b=1.0,
        m_d=2.7,
        m_hat_d=1.0,
        m_tilde_d=101.0,
        m_tilde_b=1.9,
        l=3.0,
        k_b=12.0,
        k_d=154.0,
        g=9.81,
        sample_uv=None,
        backend="numpy",
    ):
        self.physics = DrifterPhysics(
            m_b=m_b, m_d=m_d, m_hat_d=m_hat_d,
            m_tilde_d=m_tilde_d, m_tilde_b=m_tilde_b,
            l=l, g=g, k_b=k_b, k_d=k_d,
        )
        self.backend = backend
        self._qdd_func = _make_qdd_func(backend)
        self._sample_uv = sample_uv if sample_uv is not None else _default_sample_uv
```

Key changes:

- **No `get_uv` parameter.** One velocity protocol only.
- **`backend` is set at construction time and immutable.** No mutation by
  external code.
- **`_qdd_func` is set once at construction, not mutable.** The Parcels
  module never touches it.

#### Public methods

**`get_full_solution`** -- signature unchanged except `get_uv`-related
defaults are gone. Uses `self._sample_uv` for velocity queries (same as
the batch path). The scalar `_rhs` calls `sample_uv(np.array([z]))` and
unpacks.

```python
def get_full_solution(
    self,
    *,
    t_span,
    x=0.0, y=0.0,
    theta=np.pi, phi=0.0,
    xd=0.0, yd=0.0,
    thetad=0.0, phid=0.0,
    t_eval=None,
    atol=1e-3, rtol=1e-3,
) -> xr.Dataset:
```

**`get_final_drift`** -- signature unchanged.

```python
def get_final_drift(
    self,
    *,
    t_span,
    x=0.0, y=0.0,
    theta=np.pi, phi=0.0,
    xd=0.0, yd=0.0,
    thetad=0.0, phid=0.0,
) -> tuple[float, float, float]:
    # Returns (xd_final, yd_final, max_accel)
```

**`get_final_drift_batch`** -- the `sample_uv` override parameter stays.
This is needed by the Parcels kernel, which builds a new sampler each
timestep. But the save/restore trick on `self._sample_uv` is eliminated
in favor of passing `sample_uv` as a parameter to the internal
functions.

```python
def get_final_drift_batch(
    self,
    *,
    sample_uv=None,
    t_span=(0, 120),
    y0=None,
    atol=1e-3, rtol=1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Returns (xd_final, yd_final, Y_final, max_accel)
```

#### Internal methods -- rewired as functions

The key change: `_rhs`, `_rhs_batch`, and `_z_eff` become module-level
functions that take `physics`, `qdd_func`, and `sample_uv` as explicit
parameters. The class methods delegate to these.

```python
def _z_eff(l, u, v):
    """Effective drogue depth from stereographic coordinates.

    Args:
        l: Pole length [m].
        u, v: Stereographic coordinates, scalar or (N,) array.

    Returns:
        z_eff: Drogue depth [m], shape (N,), clamped <= 0.
    """
    s = u**2 + v**2
    cos_theta = (s - 4) / (s + 4)
    return np.minimum(0.0, l * cos_theta)


def _rhs(t, y, *, physics, qdd_func, sample_uv):
    """Scalar ODE RHS for solve_ivp.

    Args:
        t: Time [s].
        y: State vector of length 8.
        physics: DrifterPhysics.
        qdd_func: Callable (physics, state) -> (4,) array.
        sample_uv: Callable sample_uv(z) -> (U, V).

    Returns:
        dy/dt, length 8.
    """
    ...


def _rhs_batch(Y, *, physics, qdd_func, sample_uv):
    """Vectorized RHS for N particles.

    Args:
        Y: (N, 8) state array.
        physics: DrifterPhysics.
        qdd_func: Callable (physics, state) -> (N, 4) array.
        sample_uv: Callable sample_uv(z) -> (U, V).

    Returns:
        dY/dt, shape (N, 8).
    """
    ...
```

The `DroguedDrifter` methods become:

```python
class DroguedDrifter:

    def _rhs(self, t, y):
        return _rhs(t, y, physics=self.physics,
                    qdd_func=self._qdd_func, sample_uv=self._sample_uv)

    def _rhs_batch(self, Y, sample_uv=None):
        return _rhs_batch(Y, physics=self.physics,
                          qdd_func=self._qdd_func,
                          sample_uv=sample_uv or self._sample_uv)
```

This eliminates the save/restore trick in `get_final_drift_batch` --
the `sample_uv` override is threaded through directly:

```python
def get_final_drift_batch(self, *, sample_uv=None, t_span=(0, 120),
                          y0=None, atol=1e-3, rtol=1e-3):
    uv = sample_uv if sample_uv is not None else self._sample_uv
    # ... rest of implementation uses uv directly
```

#### What this achieves

1. **Testable without the class.** `_rhs`, `_rhs_batch`, `_z_eff` can
   be unit-tested by passing explicit `physics`, `qdd_func`, and
   `sample_uv` arguments. No class instantiation needed.

2. **No mutable state.** `sample_uv` is passed as a parameter, not
   swapped on the instance and restored in a finally block.

3. **Clean parameter plumbing.** `qdd_func` flows from
   `_make_qdd_func(backend)` through the constructor to the RHS
   functions. No re-wiring by external code.

---

### `parcels_v4.py` --- Parcels coupling

After the refactor, this module contains only Parcels-specific code:

```python
_DEG2M = 1852.0 * 60.0


def _extract_profiles(particles, fieldset, drogue_depth):
    """Sample fieldset at depth levels, return sample_uv callable.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet with UV VectorField.
        drogue_depth: Pole length [m] (determines depth range to sample).

    Returns:
        Callable sample_uv(z) -> (U, V).
    """
    ...  # Same logic as current _extract_profiles, but takes
    #      drogue_depth instead of dd. Imports make_profile_sampler
    #      from velocity.py.


def _position_update(particles, xd_ms, yd_ms, fieldset):
    """Euler-forward position update. Unchanged."""
    ...


def DDAdvectEE(particles, fieldset, *, dd):
    """Advect particles using drogued-drifter steady-state drift.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet.
        dd: DroguedDrifter instance.
    """
    sample_uv = _extract_profiles(particles, fieldset, dd.physics.l)
    xd_ms, yd_ms, _, _ = dd.get_final_drift_batch(sample_uv=sample_uv)
    _position_update(particles, xd_ms, yd_ms, fieldset)


def make_dd_kernel(dd):
    """Closure factory for Parcels kernel. Unchanged."""
    def _kernel(particles, fieldset):
        DDAdvectEE(particles, fieldset, dd=dd)
    return _kernel
```

Key change: `_extract_profiles` takes `drogue_depth` (a scalar) instead
of `dd` (the whole DroguedDrifter instance). This removes the Parcels
module's dependency on the `DroguedDrifter` class internals -- it only
needs a number. `DDAdvectEE` still takes `dd` because it calls
`get_final_drift_batch`, but `_extract_profiles` is now independently
testable with just a depth value.

#### Why the closure stays

Parcels v4 alpha requires `types.FunctionType` for kernels. This is a
Parcels constraint, not our design choice. `make_dd_kernel` returns a
plain closure. If Parcels relaxes this, `functools.partial` would
replace it.

---

### `stokes.py` --- unchanged

No changes needed. Already isolated, well-documented, well-tested.

---

### `__init__.py` --- public API

```python
from .drifter import DroguedDrifter
from .eom import DrifterPhysics, EOMState, M_func, F_func, qdd_func
from .stokes import compute_stokes_profile
```

Changes:
- `compute_stokes_profile` is now re-exported (maintainer feedback 6).
- Import source changes from `lagrange_model` to `eom`.
- `qdd_func` remains exported, now with `backend=` parameter.

---

## Velocity protocol

### The contract

```python
def sample_uv(z: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return (U, V) ocean current velocity [m/s] at depth z.

    Args:
        z: Depth [m], positive upward (0 = surface, negative = below).
           Scalar or (N,) array.

    Returns:
        (U, V): Eastward and northward velocity [m/s].
            Scalar if z is scalar, (N,) arrays if z is (N,).
    """
```

This is the only velocity interface. All code paths -- scalar ODE,
batch ODE, Parcels kernel -- go through it.

### Adapting scalar callbacks (user-side)

Users who have a `get_uv(*, t, x, y, z)` callback can wrap it
themselves:

```python
def my_sample_uv(z):
    z_arr = np.atleast_1d(np.asarray(z, float))
    U = np.array([my_get_uv(t=0, x=0, y=0, z=float(zi))[0] for zi in z_arr])
    V = np.array([my_get_uv(t=0, x=0, y=0, z=float(zi))[1] for zi in z_arr])
    if np.asarray(z).ndim == 0:
        return float(U[0]), float(V[0])
    return U, V
```

We do not provide this adapter in the package. The `get_uv` protocol
was always a convenience for idealized tests, and the adapter was
broken for realistic use (it hardcoded `t=0, x=0, y=0`). Users with
real data will construct `sample_uv` directly from their data source.

---

## Backend handling

### Current problems

1. `_qdd_func` (module-level convenience) always uses numpy, creating a
   new closure each call.
2. `qdd_func` (public) is hardcoded to numpy.
3. `_make_qdd_func` is not cached -- repeated calls with the same
   backend create redundant closures (and re-do numba warmup).

### After

`_make_qdd_func` is LRU-cached on `backend`:

```python
@functools.lru_cache()
def _make_qdd_func(backend="numpy"):
    ...  # existing implementation
```

This means:
- `_make_qdd_func("numpy")` compiles once, returns the same closure
  thereafter.
- `_make_qdd_func("numba")` JIT-compiles and warms up once, reuses
  thereafter.

`DroguedDrifter.__init__` calls `_make_qdd_func(backend)` at
construction time. Multiple `DroguedDrifter` instances with the same
backend share the same compiled function.

The public `qdd_func(physics, state, *, backend="numpy")` calls
`_make_qdd_func(backend)` on each invocation, which hits the LRU cache.

**No backend parameter on `M_func` or `F_func`.** These are diagnostic
functions, not hot path. numpy is fine.

---

## Dependency graph (after)

```
stokes.py          (standalone)
coords.py          (standalone, numpy only)
eom.py             (sympy, numpy, optional numba)
velocity.py        (numpy only)

           eom.py  coords.py  velocity.py
              \       |        /
               v      v       v
              drifter.py
                   |
                   v
              parcels_v4.py  (imports drifter, velocity)
```

Each arrow is a direct import. There are no circular dependencies.
`parcels_v4.py` imports from `drifter.py` (for `DroguedDrifter`) and
`velocity.py` (for `make_profile_sampler`). It does not import from
`eom.py` directly.

---

## Impact on examples

### Existing notebooks

| Notebook | Current import | After |
|---|---|---|
| `01_synthetic_flow_profiles` | `from drogued_drifters.drifter import DroguedDrifter` | Same (module path unchanged) |
| `02_sheared_jet_parcels` | `from drogued_drifters.drifter import DroguedDrifter; from drogued_drifters.parcels_v4 import make_dd_kernel` | Same |
| `03_drogued_drifter_in_wave_orbitals` | `from drogued_drifters import DroguedDrifter; from drogued_drifters.stokes import compute_stokes_profile` | `compute_stokes_profile` can now also be imported from top-level |
| `02_derive_effective_currents` | `from drogued_drifters.stokes import compute_stokes_profile` | Same, or `from drogued_drifters import compute_stokes_profile` |
| `04_run_simulations`, `06_run_short_simulations` | `from drogued_drifters.drifter import DroguedDrifter; from drogued_drifters.parcels_v4 import make_dd_kernel` | Same |

The only breaking change for notebooks is the removal of the `get_uv`
constructor parameter. Notebook `03_drogued_drifter_in_wave_orbitals`
uses `DroguedDrifter(sample_uv=...)`, which is fine. The idealized
notebooks that use default velocity are fine. Any notebook passing
`get_uv=` needs to switch to `sample_uv=`.

### New example: EOM study

A new notebook exercises `DrifterPhysics`, `EOMState`, `qdd_func`,
`M_func`, `F_func`, and the drag/added-mass helpers:

```python
from drogued_drifters import DrifterPhysics, EOMState, qdd_func, M_func, F_func
from drogued_drifters.drifter import (
    drogue_horizontal_added_mass,
    buoy_horizontal_added_mass,
    drogue_horizontal_drag_coeff,
    buoy_horizontal_drag_coeff,
)
```

This addresses maintainer feedback 5 (exercise `M_func`, `F_func`,
drag helpers in an example).

---

## Impact on tests

### Test changes required

1. **Import paths.** Tests importing from `lagrange_model` need to
   update to `eom` and `coords`. Mechanical change.

2. **`_qdd_func` references.** Tests currently import `_qdd_func` from
   `lagrange_model`. After deletion, they should use
   `_make_qdd_func("numpy")` or the public `qdd_func`. Since the tests
   are testing the physics (not the convenience wrapper), this is
   straightforward.

3. **`get_uv` in tests.** Several tests in `test_drogued_drifter.py`
   construct `DroguedDrifter(get_uv=...)`. These need to switch to
   `sample_uv=`. The test helper `_make_const_uv` returns a scalar
   `get_uv` callback -- it should be replaced with the existing
   `_step_sampler` pattern (which already returns `sample_uv`).

4. **`make_profile_sampler` imports.** Tests in
   `test_drifter_parcels.py` and `test_integration_full_chain.py`
   import from `parcels_v4`. After the move, they import from
   `velocity`.

### Test structure (no change)

The test file structure stays the same. The `conftest.py` with
`DEFAULT_PHYSICS` and shared fixtures stays. No new test files are
needed for the refactor itself -- the existing tests cover the
physics, and the refactor only changes how the pieces are wired.

---

## Migration path

The refactor is entirely internal. There is one external-facing
breaking change: the `get_uv` constructor parameter is removed. All
other changes are import-path renames.

### Suggested order

1. **Create `coords.py`** -- move the three coordinate transform
   functions. Update imports in `lagrange_model.py`, `drifter.py`,
   tests. This is a pure move with no logic changes.

2. **Create `velocity.py`** -- move `make_profile_sampler`. Update
   imports in `parcels_v4.py`, tests.

3. **Rename `lagrange_model.py` to `eom.py`.** Update all imports.
   (Or: create `eom.py` as an alias that re-exports from
   `lagrange_model.py`, then migrate imports incrementally. Either
   works for pre-alpha.)

4. **Cache `_make_qdd_func`.** Add `@functools.lru_cache()` decorator.
   Delete `_qdd_func`. Update tests.

5. **Add `backend=` to public `qdd_func`.** One-line change.

6. **Kill the adapter.** Delete `_adapt_get_uv`, delete `get_uv`
   constructor parameter, delete `_default_uv`. Move
   `_default_sample_uv` to module level. Update `DroguedDrifter.__init__`
   and tests.

7. **Extract `_rhs`, `_rhs_batch`, `_z_eff` to module-level functions.**
   Add explicit `physics`, `qdd_func`, `sample_uv` parameters. Class
   methods become one-line delegates. Eliminate the save/restore in
   `get_final_drift_batch`.

8. **Slim down `_extract_profiles`** to take `drogue_depth` instead of
   `dd`.

9. **Update `__init__.py`** to export `compute_stokes_profile`.

10. **New example notebook** for EOM study.

Steps 1-3 are mechanical moves. Steps 4-8 are the real refactoring.
Step 9-10 are polish. Each step is independently committable and
testable.

---

## What this does NOT change

- **Symbolic derivation.** `_derive_symbolic()` and the pickle cache
  are untouched.
- **`_build_packer`.** The signature-inspection packer stays.
- **`DrifterPhysics` and `EOMState`.** NamedTuples stay as-is.
- **`stokes.py`.** Unchanged.
- **Parcels kernel pattern.** `make_dd_kernel` + `DDAdvectEE` closure
  stays. The Parcels-facing API is unchanged.
- **ODE integration.** `solve_ivp` usage stays. No change to solver
  choice, tolerances, or state vector layout.
- **Coordinate convention.** z-up, stereographic internal, spherical
  public.

---

## Trade-offs considered

### Why not delete `DroguedDrifter` entirely?

The maintainer considered this. Arguments for deletion: the class is
just a bag of parameters plus method dispatch. Arguments against:

1. **The constructor provides Callies et al. defaults.** Users write
   `DroguedDrifter()` and get a working drifter. Without the class,
   they would need to construct `DrifterPhysics(m_b=1.0, m_d=2.7, ...)`
   explicitly -- 9 parameters with no defaults.

2. **The Parcels kernel needs something to close over.** `make_dd_kernel`
   captures `dd` in a closure. If there is no class, it captures a
   (physics, qdd_func, sample_uv) triple, which is equivalent but less
   readable.

3. **The class is already thin after the refactor.** Constructor + 3
   public methods that delegate to module-level functions. There is
   nothing left to simplify.

The right answer is: keep the class as a convenience, but make all the
real computation accessible as standalone functions underneath. Users
who want the class get the class. Users who want functions get functions.
The class is not load-bearing for any logic.

### Why not separate `eom_symbolic.py` and `eom_numeric.py`?

The symbolic derivation and numeric evaluation are currently in one
file. Splitting them would put `_derive_symbolic`, `_load_or_derive`,
`_cache_key`, `_sym_norm` in one file and the rest in another.

Against: the only consumer of the symbolic output is the numeric
lambdification in the same file. Splitting creates two tightly coupled
files instead of one cohesive file. The symbolic code is ~130 lines;
the numeric code is ~180 lines. Together they are manageable.

Keep as one file.

### Why move `make_profile_sampler` out of `parcels_v4.py`?

It has no Parcels dependency. Tests already use it standalone. It is
useful for anyone building velocity profiles from data (e.g., the
integration tests, the synthetic flow notebook). Keeping it in the
Parcels module forces non-Parcels users to import from a Parcels-named
module, which is confusing.

### Why not add `backend=` to `M_func` and `F_func`?

These functions are used for inspection and verification, not for ODE
integration. The mass matrix and force vector are computed once per
study, not 10,000 times per simulation. Numba overhead (import, JIT)
is not justified. Keep them numpy-only.
