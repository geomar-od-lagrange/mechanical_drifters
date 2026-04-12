# Architecture review

## 1. Architecture overview

### Module map

```
src/drogued_drifters/
  __init__.py          Re-exports DroguedDrifter, DrifterPhysics, M_func, F_func
  lagrange_model.py    Symbolic derivation, caching, lambdification, coordinate transforms
  drifter.py           DroguedDrifter class, ODE integration, parameterization helpers
  parcels_v4.py        Parcels kernel, profile sampler, make_dd_kernel factory
  stokes.py            Stokes drift profile computation (standalone utility)
  data/eom_cache.pkl   Cached symbolic derivation output
```

### Dependency flow

```
stokes.py  (standalone, no internal deps)

lagrange_model.py  (sympy, numpy, scipy -- no internal deps)
       |
       v
drifter.py  (imports DrifterPhysics, EOMState, _qdd_func, coordinate transforms)
       |
       v
parcels_v4.py  (imports DroguedDrifter, and reaches into lagrange_model for numba path)
```

### Key abstractions

- **`DrifterPhysics`** (NamedTuple): frozen physical parameters (9 fields)
- **`EOMState`** (NamedTuple): per-timestep state + forcing (10 fields)
- **`DroguedDrifter`**: main simulator, owns physics + velocity callback + ODE integration
- **`_qdd_func`**: module-level function evaluating generalized accelerations
- **`make_profile_sampler`**: closure factory for depth-interpolated velocity profiles
- **`DDAdvectEE`**: Parcels kernel function
- **`make_dd_kernel`**: factory that captures `dd` in a closure for Parcels

## 2. Strengths

**Clean physics / Parcels separation.** The core physics (`lagrange_model.py`,
`drifter.py`) has zero Parcels dependency. Physics can be tested, used, and
reasoned about without Parcels installed. This is the most important structural
property of the codebase and it is solid.

**Derivation chain integrity.** The sympy derivation in `_derive_symbolic` goes
from Lagrangian to EOM to lambdified functions with no manual algebraic steps.
The cache with source-hash invalidation is pragmatic and well-implemented.

**Signature-based argument packing.** `_build_packer` inspects the lambdified
function's parameter names and maps them to `DrifterPhysics` / `EOMState`
fields by name. This eliminates positional ordering bugs -- a real risk with 19
parameters. The tests in `test_lagrange_physics.py` (`test_packer_covers_all_struct_fields`,
`test_packer_arg_order_matches_lambda`) are exactly the right tests for this.

**NamedTuples for physics and state.** `DrifterPhysics` and `EOMState` are
frozen, lightweight, and provide named access. Good choice for data that is
constructed once and passed through.

**Test coverage at physics boundaries.** Positive-definiteness of M, quadratic
drag scaling, coordinate round-trips, batch-vs-scalar consistency -- these test
the physics contract, not implementation details.

**`stokes.py` is perfectly isolated.** No internal dependencies, clean
vectorized API, good docs. Nothing to change here.

## 3. Findings

### Critical

#### C1. Numba backend breaks the abstraction boundary

**File:** `parcels_v4.py:168-201` (the TODO at line 175)

**Problem:** The `numba` branch of `make_dd_kernel` reaches into
`lagrange_model._get_eom_callables()` to get `qdd_raw`, wraps it with
`njit`, and mutates `dd._qdd_func` as a side effect of creating a kernel.
The `numpy` branch does none of this -- it just wraps `DDAdvectEE` in a
closure. This asymmetry reveals that backend selection is at the wrong layer.

Concrete consequences:

1. **Mutation as side effect.** Calling `make_dd_kernel(dd, backend="numba")`
   permanently mutates `dd._qdd_func`. If you later call
   `make_dd_kernel(dd, backend="numpy")`, `dd` still uses the numba function.
   There is no way to undo it, and the caller has no reason to expect that a
   kernel factory mutates the drifter instance.

2. **Wrong layer.** The Parcels integration layer (`parcels_v4.py`) now knows
   about `_get_eom_callables`, `pack_eom_args`, and the internal signature of
   the lambdified function. These are `lagrange_model` internals.

3. **Incomplete coverage.** The numba path only accelerates `_qdd_func`. But
   `DroguedDrifter._rhs` (the scalar path used by `get_final_drift` and
   `get_full_solution`) also calls `self._qdd_func`. Since the mutation
   happens through `make_dd_kernel`, the scalar path never benefits unless the
   user happens to call `make_dd_kernel` first -- an invisible ordering
   dependency.

**Suggested fix:** Backend selection belongs in `lagrange_model.py` or on
`DroguedDrifter.__init__`. Two clean options:

**(a)** `_get_eom_callables` gains a `backend` parameter and returns the
appropriate `qdd_raw` (numpy or njit-wrapped). `_qdd_func` dispatches based
on the backend. `DroguedDrifter.__init__` accepts `backend=` and passes it
through. `make_dd_kernel` loses its `backend` parameter entirely.

**(b)** A `_numba.py` module (as sketched in
[numba-acceleration.md](numba-acceleration.md) Option A) that provides
`qdd_func_numba` as a drop-in replacement. `DroguedDrifter.__init__`
accepts `qdd_func=` and defaults to the numpy version. The Parcels layer
never touches it.

Option (a) is simpler for the current scope. Option (b) is more extensible
if other backends (JAX, C codegen) are ever considered.

**Depends on:** nothing. This is the first thing to fix.

---

### Important

#### I1. `DroguedDrifter` has two incompatible velocity interfaces

**Files:** `drifter.py:158-162` (`get_uv` callback), `drifter.py:231-286`
(`_rhs_batch` with `sample_uv`)

**Problem:** The scalar path (`_rhs`, `get_final_drift`, `get_full_solution`)
uses `self.get_uv(t=t, x=x, y=y, z=z)` -- a callback taking keyword
arguments `(t, x, y, z)` and returning a 2-tuple of scalars.

The batch path (`_rhs_batch`, `get_final_drift_batch`) uses
`sample_uv(z)` -- a positional callable taking a depth array and returning
`(N,)` arrays. It has no `t`, `x`, or `y` parameters because those are
handled externally by the Parcels kernel.

These are not two views of the same abstraction. They are two different
contracts that happen to be on the same class. The Parcels kernel uses the
batch path exclusively; the scalar `get_uv` callback is only used by the
standalone `_rhs` / `get_full_solution` path. This means:

1. **You cannot use `get_full_solution` with a `sample_uv`-style velocity
   profile.** The scalar path always queries `self.get_uv`, which knows
   nothing about the pre-sampled profiles that `make_profile_sampler` builds.

2. **The `get_uv` constructor parameter is dead weight in the Parcels path.**
   It is set, stored, but never called.

3. **Testing the scalar path with realistic velocity fields requires writing
   a `get_uv` callback by hand**, even though `make_profile_sampler` already
   provides the profile in a different format.

**Suggested fix:** Unify around a single velocity protocol. The batch
`sample_uv(z) -> (U, V)` interface is the better one (it's what the ODE
hot path actually needs). For the scalar path, `_rhs` can call
`sample_uv(np.array([z]))` and unpack. The `get_uv` constructor parameter
becomes a convenience adapter:

```python
def __init__(self, ..., get_uv=None, sample_uv=None):
    if sample_uv is not None:
        self._sample_uv = sample_uv
    elif get_uv is not None:
        self._sample_uv = _adapt_getuv(get_uv)
    else:
        self._sample_uv = self._default_sample_uv
```

This also means `get_final_drift_batch` no longer needs `sample_uv` as
a parameter -- it uses `self._sample_uv`, same as the scalar path.

**Depends on:** nothing, but doing this before C1 simplifies the kernel.

#### I2. `_qdd_func` is a module-level function stored as an instance attribute

**File:** `drifter.py:156`

**Problem:** `self._qdd_func = _qdd_func` copies a module-level function
reference onto the instance. This exists solely so that the numba path in
`parcels_v4.py` can replace it with a JIT-compiled version via
`dd._qdd_func = _qdd_func_numba`.

This is the plumbing that makes C1 possible. It's not a monkey-patch in
the worst sense (it uses an instance attribute, not class-level patching),
but it makes the function dispatch opaque -- there's no way to inspect
which backend a `DroguedDrifter` is using.

**Suggested fix:** Resolves naturally when C1 is fixed. If backend
selection moves to `__init__`, `_qdd_func` becomes a constructor-time
binding based on the backend parameter, not a mutable slot.

**Depends on:** C1.

#### I3. `DDAdvectEE` does too much

**File:** `parcels_v4.py:57-136`

**Problem:** `DDAdvectEE` is a single function that:
1. Extracts velocity profiles from the fieldset (lines 89-116)
2. Converts depth conventions (lines 119-123)
3. Builds the profile sampler (line 124)
4. Runs the drifter ODE (line 127)
5. Converts velocity units (lines 113-115, 131-136)
6. Updates particle positions (lines 131-136)

This makes it hard to test the profile extraction separately from the ODE,
and hard to swap the position update scheme (e.g., upgrading from Euler to
RK4).

**Suggested fix:** Factor into `_extract_profiles(particles, fieldset, dd)
-> sample_uv` and `_position_update(particles, xd_ms, yd_ms, fieldset)`.
The kernel becomes:

```python
def DDAdvectEE(particles, fieldset, *, dd):
    sample_uv = _extract_profiles(particles, fieldset, dd)
    xd_ms, yd_ms, _, _ = dd.get_final_drift_batch(sample_uv=sample_uv)
    _position_update(particles, xd_ms, yd_ms, fieldset)
```

Each piece is independently testable. The profile extraction tests
currently in `test_drifter_parcels.py` already test `make_profile_sampler`
in isolation, so factoring it out makes that separation real.

**Depends on:** nothing.

#### I4. `_DEFAULT_PHYSICS` is duplicated across three test files

**Files:** `tests/test_drogued_drifter.py:306`,
`tests/test_lagrange_physics.py:22`, `tests/test_numerical_edge_cases.py:22`

**Problem:** The same `DrifterPhysics(m_b=1.0, m_d=2.7, ...)` NamedTuple
is defined identically in three test files. If the default parameters
change, all three must be updated.

**Suggested fix:** Either:
- Make `DroguedDrifter().physics` the canonical default (the constructor
  already uses these values), and use `DroguedDrifter().physics` in tests.
- Add `DEFAULT_PHYSICS` as a module-level constant in `drifter.py` or
  `lagrange_model.py`.

**Depends on:** nothing.

---

### Consider

#### S1. Pickle cache is fragile across Python versions

**File:** `lagrange_model.py:244-294`

**Problem:** The EOM cache is a pickled dict of sympy objects. Pickle
compatibility across Python minor versions and sympy versions is not
guaranteed. The cache key only hashes the source of `_derive_symbolic` and
`sp.__version__`, not the Python version or pickle protocol. A stale cache
from a different Python version could load without error but produce wrong
results if sympy's internal representation changed.

**Mitigation already present:** The `lru_cache` on `_get_eom_callables`
means this only matters on cold start. The source-hash key provides
reasonable invalidation. For pre-alpha, this is fine.

**If it ever matters:** Add `sys.version_info[:2]` and
`pickle.HIGHEST_PROTOCOL` to the cache key.

#### S2. `make_profile_sampler` closure captures mutable arrays

**File:** `parcels_v4.py:36-54`

**Problem:** `make_profile_sampler` converts inputs to numpy arrays but
does not copy them. If the caller mutates the original arrays after
creating the sampler, the sampler's behavior changes silently.

In practice this is safe because `DDAdvectEE` creates fresh arrays each
call. But the function's API contract doesn't prevent misuse.

**If it ever matters:** Add `.copy()` to the array conversions on lines
37-38.

#### S3. `__init__.py` exports `M_func` and `F_func` but not `_qdd_func`

**File:** `__init__.py:2`

**Problem:** `M_func` and `F_func` are diagnostic/inspection functions
(evaluate mass matrix and force vector separately). They are exported as
public API. `_qdd_func` (the function that actually computes accelerations)
is private. This is fine -- but the public API currently has no way to
evaluate `qdd` without going through `DroguedDrifter`. If someone wants
to study the EOM without the ODE wrapper, they need `_qdd_func`.

**If it ever matters:** Consider exporting `qdd_func` (without underscore)
or documenting that `M_func` + `F_func` + `np.linalg.solve` is the
intended external interface for direct EOM evaluation.

#### S4. Coordinate conversion is hidden inside `DroguedDrifter`

**File:** `drifter.py:288-417` (`get_final_drift_batch`)

**Problem:** The public interface of `get_final_drift_batch` accepts and
returns `(theta, phi, thetad, phid)` -- spherical coordinates. Internally,
it converts to stereographic `(u, v, ud, vd)` for the ODE, then converts
back. The kernel (`DDAdvectEE`) uses `get_final_drift_batch` with `y0=None`
(cold start), so it never passes or receives angular state.

If warm-starting is implemented (per docs and backlog), the kernel will
need to store and restore `Y_final` per particle. That state is in spherical
coordinates (the public format). But the ODE works in stereographic. Every
warm-start call pays for two coordinate conversions that are pure overhead.

**If warm-start happens:** Allow `get_final_drift_batch` to accept and
return internal (stereographic) state directly, bypassing the spherical
round-trip. A `raw=True` flag or a separate `_get_final_drift_batch_internal`
method would work.

#### S5. No conftest.py or test fixtures for common setup

**Files:** all test files

**Problem:** Each test file independently creates `DroguedDrifter()` instances,
`EOMState` objects, and step samplers. There is no shared fixture for common
test setup. This is fine at the current scale (7 test files), but shared
fixtures for `dd = DroguedDrifter()` and `default_physics` would reduce
boilerplate and make I4 unnecessary.

## 4. Recommended order of operations

1. **C1 (numba abstraction boundary)** -- the only critical finding. Fix
   this first because it also resolves I2.

2. **I3 (factor DDAdvectEE)** -- independent of C1, can be done in
   parallel. Makes the kernel testable in parts.

3. **I1 (unify velocity interface)** -- simplifies the `DroguedDrifter` API
   and eliminates the dual-callback confusion. Can be done after C1 since
   the `sample_uv` protocol is the one that survives.

4. **I4 (deduplicate test constants)** -- quick cleanup, do alongside any
   of the above.

5. **S1-S5** -- address opportunistically or defer.
