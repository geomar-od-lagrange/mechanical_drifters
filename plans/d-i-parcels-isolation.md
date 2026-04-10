# D-I: Parcels Isolation

## Goal

Move all Parcels-coupling code out of `drifter.py` into a new
`parcels_v4.py` module.  Replace the private `_get_corner_data_Agrid` call
with grid-agnostic `Field.eval()` calls, so the bridge works unchanged on
A-grids, C-grids, curvilinear grids, and (in principle) unstructured grids.
Drop warm-state caching entirely (simple, correct; revisit via
particle-based state once Parcels exposes particle identity to
`vector_interp_method`).  The resulting module should be clean enough to
show to Parcels developers as a reference for a future profile-sampling API.

## Current state

`drifter.py` contains two module-level functions that are purely
Parcels-facing:

| Function | Lines | Role |
|---|---|---|
| `make_profile_sampler` | 19–57 | Builds `sample_uv(z)` closure from `(D, N)` velocity arrays |
| `make_dd_velocity_interpolator` | 61–216 | Factory returning a `vector_interp_method` callback |

The callback (`_interpolator`) does three things:
1. **Profile extraction** — loops over D depth levels, calling
   `_get_corner_data_Agrid` (private, A-grid-only) to get 2×2 corners, then
   manually does bilinear + temporal interpolation.
2. **DD model call** — `make_profile_sampler` → `dd.get_final_drift_batch`.
3. **Unit conversion** — m/s → deg/s when `spherical=True`.

### What's wrong

- **Grid lock-in.**  `_get_corner_data_Agrid` hard-codes the A-grid data
  layout.  C-grids, curvilinear grids, and unstructured grids are
  impossible.  The current code also reuses U's spatial indices `(yi, xi)`
  for V — correct on A-grids where U and V share a grid, wrong on C-grids
  where they are staggered.
- **Private API dependency.**  `_get_corner_data_Agrid` is an undocumented
  Parcels internal that can break without notice.
- **Warm-state bug.**  Cache validation only checks particle count `N`.  If
  particles are deleted OOB and the array is reorganised such that `N` is
  unchanged but particles at certain indices are different, stale ODE state
  is silently reused.  Fixing this properly requires per-particle identity
  (particle IDs or custom variables), which `vector_interp_method` doesn't
  receive — `VectorField.eval()` accepts `particles` but doesn't forward it
  to the callback.  Rather than work around this with fragile heuristics,
  drop warm-starting for now and cold-start from equilibrium every call.

## Design

### Module layout after the change

```
src/drogued_drifters/
├── __init__.py          # unchanged
├── drifter.py           # DroguedDrifter + make_profile_sampler (physics only, no Parcels imports)
├── lagrange_model.py    # symbolic derivation + lambdified callables
├── stokes.py            # Stokes drift model
└── parcels_v4.py        # NEW — all Parcels coupling lives here
```

### What moves, what stays

| Item | From | To | Rationale |
|---|---|---|---|
| `make_dd_velocity_interpolator` | `drifter.py` | `parcels_v4.py` | Parcels-specific; depends on Parcels imports |
| `make_profile_sampler` | `drifter.py` | stays in `drifter.py` | Generic depth-interpolation utility; used by the Parcels bridge but also useful standalone |
| `_get_corner_data_Agrid` import | `drifter.py` | deleted | Replaced by `Field.eval()` |

No re-exports or backward-compat shims in `drifter.py`.  Import path
changes from:
```python
from drogued_drifters.drifter import make_dd_velocity_interpolator
```
to:
```python
from drogued_drifters.parcels_v4 import make_dd_velocity_interpolator
```

### Grid-agnostic profile extraction via `Field.eval()`

The key change.  Instead of extracting corner data and manually
interpolating, we call Parcels' public `Field.eval(time, z, y, x)` API.
This method internally delegates to `grid.search()` + the field's native
`_interp_method`, which handles A-grids, C-grids, curvilinear, and
unstructured grids transparently.

**Current (A-grid only):**
```python
from parcels.interpolators._xinterpolators import _get_corner_data_Agrid

for iz in range(D):
    zi_arr = np.full(N, iz, dtype=np.int32)
    corner_U = _get_corner_data_Agrid(field_U.data, ti, zi_arr, yi, xi, ...)
    corner_V = _get_corner_data_Agrid(field_V.data, ti, zi_arr, yi, xi, ...)
    # manual time interpolation ...
    # manual bilinear interpolation ...
```

**New (grid-agnostic):**
```python
# No private imports needed.

depth_levels = np.asarray(field_U.grid.depth, dtype=float)  # Parcels z-down

for iz, z_level in enumerate(depth_levels):
    z_arr = np.full(N, z_level)
    U_profiles[iz] = field_U.eval(time, z_arr, lat, lon)
    V_profiles[iz] = field_V.eval(time, z_arr, lat, lon)
```

**Why this is correct for staggered (C) grids:**  On C-grids, U and V live
at different horizontal positions.  `field_U.eval()` uses U's grid and U's
interpolator; `field_V.eval()` uses V's grid and V's interpolator.  Each
field handles its own staggering.  The current code uses the *same*
`(yi, xi)` indices for both — it only works because A-grids share the grid.

**Why `Field.eval()` works from inside `vector_interp_method`:**  Our
callback replaces the *VectorField*-level interpolation.  Calling
individual `Field.eval()` invokes each field's own scalar interpolator
(`XLinear`, `CGrid_Velocity`, etc.) — no recursion, no shared mutable
state.  We pass `particles=None`, so no element-index caching or particle
state updates.

**Depth convention handling:**  `Field.eval(time, z, ...)` expects `z` in
Parcels convention (positive downward).  `field.grid.depth` returns values
in that convention.  We sample at those native depth levels, then negate and
reverse for `make_profile_sampler` (which expects z-up, ascending):

```python
depth_up = -depth_levels[::-1]      # e.g., [0, 5, 20] → [-20, -5, 0]
U_profiles = U_profiles[::-1]
V_profiles = V_profiles[::-1]
sample_uv = make_profile_sampler(depth_up, U_profiles, V_profiles)
```

**Edge case — no Z axis:**  Some fields are 2D (surface only).
`grid.depth` returns `np.zeros(1)` in that case.  The profile has a single
level at z=0, so `sample_uv(z)` returns the same velocity at all depths.
The drifter model degrades gracefully: buoy and drogue see identical
currents, so the pole hangs vertical and drift velocity = current velocity.
No special handling needed.

### Unstructured grids (UxGrid)

Out of scope for the current dataset, but the design accommodates them:

- `Field.eval()` works identically on unstructured grids.  `UxGrid.search`
  uses a spatial hash for lateral location and `_search_1d_array` for
  vertical.
- `UxGrid.depth` exists (returns `self.z.values`), marked v3-compat like
  XGrid's.
- The `grid_positions` dict keys differ (`"FACE"` instead of `"X"`/`"Y"`),
  but our new code never touches `grid_positions` — it uses physical
  coordinates exclusively.

The only limitation: per-depth-level `Field.eval()` calls redo the lateral
spatial hash query D times.  For structured grids this is a cheap binary
search; for unstructured grids the hash is more expensive.  If performance
matters for unstructured grids in the future, we can batch all D×N points
into a single `Field.eval()` call:

```python
z_tiled = np.repeat(depth_levels, N)
lat_tiled = np.tile(lat, D)
lon_tiled = np.tile(lon, D)
time_tiled = np.broadcast_to(np.atleast_1d(time), D * N)
all_U = field_U.eval(time_tiled, z_tiled, lat_tiled, lon_tiled)
U_profiles = all_U.reshape(D, N)
```

This is a single search + interpolation pass.  Note in the plan; implement
later if needed.

### Performance trade-off

The current code does **1** horizontal search (by Parcels, before calling
our callback) and reuses the indices for all D depths.  The new code does
**2×D** horizontal searches (one per field per depth level).

For typical parameters (D=20 depth levels, N=50–100 particles, structured
grid with ~300×150 cells), each `_search_1d_array` is O(log 300) ≈ 9
comparisons.  Total overhead: 2×20×100×9 ≈ 36k comparisons — negligible
compared to the ODE solve in `get_final_drift_batch` which dominates the
runtime.

This is the "go the slow (grid-agnostic) way" decision from the code review
remarks.

### Warm-starting: dropped for now

The current `warm_state` dict caches the full ODE state `(N, 8)` between
Parcels timesteps so the solver can pick up near the previous equilibrium.
The cache validation is broken (checks only particle count), and fixing it
properly requires per-particle identity.

**Decision:** Drop warm-starting entirely.  Every interpolator call
cold-starts from equilibrium (`y0=None`).  The ODE converges quickly
(typical `t_span=(0, 120)` is sufficient), so the performance cost is
small.  The code becomes much simpler — no `warm_state` parameter, no cache
management, no validation logic.

**Future path:** Store drifter state (theta, phi, xd, yd, thetad, phid) as
Parcels particle variables via a custom `DDParticle` class.  The
interpolator would read/write those variables per particle.  This requires
either:
- Parcels forwarding `particles` to `vector_interp_method` (upstream ask), or
- Capturing the `ParticleSet` in the interpolator closure (works but
  unconventional).

This is deferred to the upstream engagement.  When implemented, it also
gives us drifter state in Parcels output for free (diagnostics,
restartability).

### Public API of `parcels_v4.py`

```python
def make_dd_velocity_interpolator(
    dd: DroguedDrifter,
    *,
    spherical: bool = False,
) -> Callable:
    """Create a Parcels v4 vector interpolator for drogued-drifter drift velocity.

    Uses ``Field.eval()`` for grid-agnostic profile extraction.
    Works with A-grids, C-grids, curvilinear grids, and (in principle)
    unstructured grids.

    Each call cold-starts the ODE from equilibrium (pole vertical, at rest).

    Args:
        dd: DroguedDrifter instance.
        spherical: If True, convert m/s drift output to deg/s for
            Parcels ``mesh="spherical"`` convention.

    Returns:
        Callable matching the Parcels ``vector_interp_method`` signature:
        ``(particle_positions, grid_positions, vectorfield) -> (u, v, w)``.

    Usage::

        from drogued_drifters.parcels_v4 import make_dd_velocity_interpolator

        dd = DroguedDrifter()
        fieldset = FieldSet.from_sgrid_conventions(ds, mesh="spherical")
        fieldset.UV.vector_interp_method = make_dd_velocity_interpolator(
            dd, spherical=True,
        )
    """
```

Simpler signature than the current function: `warm_state` parameter
removed.  Import path changes from `drogued_drifters.drifter` to
`drogued_drifters.parcels_v4`.

## Implementation steps

### Step 1: Create `parcels_v4.py` with the new implementation

Write `src/drogued_drifters/parcels_v4.py`:

- `make_dd_velocity_interpolator(dd, *, spherical)` — factory

The `_interpolator` closure inside the factory:

```python
def _interpolator(particle_positions, grid_positions, vectorfield):
    lat = particle_positions["lat"]
    lon = particle_positions["lon"]
    time = particle_positions["time"]
    N = len(lat)

    field_U = vectorfield.U
    field_V = vectorfield.V
    depth_levels = np.asarray(field_U.grid.depth, dtype=float)
    D = len(depth_levels)

    # Grid-agnostic profile extraction
    U_profiles = np.empty((D, N))
    V_profiles = np.empty((D, N))
    for iz, z_level in enumerate(depth_levels):
        z_arr = np.full(N, z_level)
        U_profiles[iz] = field_U.eval(time, z_arr, lat, lon)
        V_profiles[iz] = field_V.eval(time, z_arr, lat, lon)

    # Convert to z-up ascending for make_profile_sampler
    depth_up = -depth_levels[::-1]
    U_profiles = U_profiles[::-1]
    V_profiles = V_profiles[::-1]

    sample_uv = make_profile_sampler(depth_up, U_profiles, V_profiles)

    # Cold-start from equilibrium every call
    xd_ms, yd_ms, _, max_accel = dd.get_final_drift_batch(
        sample_uv=sample_uv,
    )

    # deg/s conversion for spherical mesh
    if spherical:
        cos_lat = np.cos(np.deg2rad(lat))
        u = xd_ms / (_DEG2M * cos_lat)
        v = yd_ms / _DEG2M
    else:
        u = xd_ms
        v = yd_ms

    return (u, v, np.zeros_like(u))
```

### Step 2: Remove Parcels code from `drifter.py`

- Delete `make_dd_velocity_interpolator` (lines 60–216).
- Delete the comment at line 18 about Parcels depth convention (no longer
  relevant to this file).
- Keep `make_profile_sampler` (lines 19–57) — it's physics, not Parcels.

### Step 3: Update imports and call sites

- `examples/idealized_flow/02_sheared_jet_parcels.ipynb`: change import
  from `drogued_drifters.drifter` to `drogued_drifters.parcels_v4`.
  Remove `dd_warm_state = {}` and `warm_state=dd_warm_state` from the call.
- `tests/test_drifter_parcels.py`: update `make_dd_velocity_interpolator`
  import.  Remove warm-state tests.  `make_profile_sampler` import stays
  unchanged (still from `drifter`).

### Step 4: Add integration tests with a synthetic FieldSet

Add tests to `tests/test_drifter_parcels.py` that construct a real Parcels
`FieldSet` and exercise the new `_interpolator` end-to-end:

1. **Uniform-flow test:**  Create a FieldSet with constant U=0.5, V=0 at
   all depths.  Call `VectorField.eval()` via the DD interpolator.  The
   drifter should return U ≈ 0.5 (buoy and drogue see the same current →
   pole vertical → drift = current).

2. **Sheared-flow test:**  Surface U=1.0, bottom U=0.  The drift velocity
   should be between 0 and 1 (intermediate, as in the example notebook).

### Step 5: Run the example notebook

Execute `examples/idealized_flow/02_sheared_jet_parcels.ipynb` with
papermill to verify the full pipeline still works with the new
implementation.

### Step 6: Clean up

- Remove `/tmp/_inspect_uxgrid.py` and any other temp files.
- Run black, check tests pass.

## Test plan

| Test | What it validates |
|---|---|
| Existing `test_make_profile_sampler_*` (8 tests) | `make_profile_sampler` still works after moving the Parcels code out |
| Existing `test_make_dd_velocity_interpolator_*` (4 tests) | Updated to import from `parcels_v4`; warm-state tests dropped, signature tests updated |
| **New:** `test_uniform_flow_field_eval` | Grid-agnostic profile extraction produces correct values on a real FieldSet |
| **New:** `test_sheared_flow_drift_velocity` | Full pipeline: FieldSet → profile extraction → DD model → drift velocity |
| Example notebook execution | End-to-end integration with Parcels advection loop |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `field.grid.depth` is marked "v3 compat, may be removed" | It's the only public way to get depth levels.  If removed, extract from `field.grid._ds["depth"].values` or accept `depth_levels` as a parameter. |
| `Field.eval()` changes in future Parcels alpha | `eval()` is the primary public API for field sampling.  Much more stable than `_get_corner_data_Agrid`. |
| Performance regression from repeated grid searches | Measured: search cost is negligible vs ODE solve.  Batched single-call optimisation available if needed. |
| No warm-starting → slower convergence | The ODE converges from equilibrium in ~120 s of simulated time.  Acceptable for correctness-first implementation.  Profile separately if it becomes a bottleneck. |

## Out of scope

- **Parcels upstream PR.**  This plan produces the clean module to show
  them; the actual engagement is a separate effort.
- **Unstructured grid testing.**  The design supports it, but we have no
  unstructured dataset to test against.  The batched `Field.eval()` optimisation
  for spatial-hash performance is deferred.
- **Warm-starting.**  Deferred until Parcels forwards `particles` to
  `vector_interp_method` (upstream ask) or we switch to a kernel-based
  architecture with a custom `DDParticle` class.  At that point, drifter
  state (theta, phi, etc.) would live on the particles and appear in output
  automatically — solving warm-starting, diagnostics, and restartability in
  one shot.
