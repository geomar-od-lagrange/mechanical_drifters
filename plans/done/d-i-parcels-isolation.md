# D-I: Parcels Isolation

> **Status: implemented.** The authoritative documentation is
> [`docs/parcels-v4-coupling.md`](../docs/parcels-v4-coupling.md).
> This plan is kept as a historical record.  Notable deviation from
> plan: `functools.partial(DDAdvectEE, dd=dd)` does not work — Parcels
> v4 alpha checks `isinstance(f, types.FunctionType)` and rejects
> `partial` objects.  A `make_dd_kernel(dd)` closure factory is used
> instead.

## Goal

Move all Parcels-coupling code out of `drifter.py` into a new
`parcels_v4.py` module.  Replace the private `_get_corner_data_Agrid`
interpolator approach with a **kernel** that calls `fieldset.UV.eval()` per
depth level — grid-agnostic (A, C, curvilinear, unstructured in principle),
handles vector rotation on C-grids correctly, and auto-detects
spherical/flat mesh.  Drop warm-state caching (revisit via particle-based
state later).  The resulting module should be clean enough to show to
Parcels developers as a reference for a future profile-sampling API.

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
- **Missing vector rotation.**  On C-grids and curvilinear grids, the
  VectorField interpolator (`CGrid_Velocity`) applies a Jacobian-based
  rotation from grid-aligned to geographic (east-north) coordinates.
  Calling individual `field_U.eval()` / `field_V.eval()` bypasses this
  rotation — the returned velocities would be in grid coordinates, not
  east-north.
- **Private API dependency.**  `_get_corner_data_Agrid` is an undocumented
  Parcels internal that can break without notice.
- **Warm-state bug.**  Cache validation only checks particle count `N`.
  Fixing this properly requires per-particle identity, which the
  `vector_interp_method` callback doesn't receive.  Drop warm-starting
  entirely for now.
- **Manual spherical conversion.**  The interpolator duplicates the m/s →
  deg/s logic that Parcels' built-in velocity interpolators already handle
  (via `grid._mesh == "spherical"` check).

### Why a kernel, not a custom interpolator

The previous approach replaced `fieldset.UV.vector_interp_method` with a
custom callback, then called individual `Field.eval()` inside it.  This
has two problems:

1. Individual `Field.eval()` uses the scalar interpolator (`XLinear`), which
   does NOT apply the C-grid vector rotation.  The rotation lives in
   `CGrid_Velocity` (the VectorField-level interpolator), which we replaced.
2. The spherical conversion also lives in the VectorField interpolator.  Our
   custom callback had to re-implement it.

A **kernel** avoids both problems: it calls `fieldset.UV.eval()` at each
depth level using the **default** VectorField interpolator, which handles
rotation and unit conversion correctly for all grid types.  The DD model
operates in m/s internally, so the kernel converts from the interpolator's
output units (deg/s for spherical mesh) to m/s, runs the physics, and
converts back for the position update.

Trade-off: we replace `AdvectionEE` with our own kernel instead of plugging
into it.  Since the DD model already finds the steady-state drift velocity,
Euler forward is the natural scheme.  Upgrade to RK4 later if needed.

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

### What moves, what stays, what's dropped

| Item | From | To | Rationale |
|---|---|---|---|
| `make_dd_velocity_interpolator` | `drifter.py` | deleted | Replaced by `DDAdvectEE` kernel |
| `make_profile_sampler` | `drifter.py` | `parcels_v4.py` | Parcels devs should see the full pipeline in one place |
| `_get_corner_data_Agrid` import | `drifter.py` | deleted | Replaced by `fieldset.UV.eval()` |
| `warm_state` parameter | `drifter.py` | deleted | Dropped entirely; cold-start from equilibrium |

No re-exports or backward-compat shims.

### Grid-agnostic profile extraction via `fieldset.UV.eval()`

The kernel calls `fieldset.UV.eval(time, z_arr, lat, lon)` at each depth
level.  This uses the **default** VectorField interpolator
(`XLinear_Velocity` for A-grids, `CGrid_Velocity` for C-grids), which:

- Handles spatial interpolation (bilinear, C-grid staggering, curvilinear)
- Applies Jacobian-based vector rotation on C-grids
- Converts m/s → deg/s for spherical mesh

The kernel then converts the output back to m/s for the DD physics model.

**Depth limiting:** Only sample depths relevant for the drifter:
surface to `dd.physics.l` (drogue length) + one grid cell margin.  This
reduces D from ~20 (full water column) to ~3–5 for typical drogue depths
(3 m).

**Depth convention:**  `field.grid.depth` returns values in Parcels
convention (positive downward).  After sampling, negate and reverse for
`make_profile_sampler` (which expects z-up, ascending).

**Edge case — no Z axis:**  `grid.depth` returns `np.zeros(1)`.  The
profile has a single level at z=0.  Buoy and drogue see identical currents,
pole hangs vertical, drift = current.  No special handling needed.

### Mesh type auto-detection

The kernel reads `fieldset.U.grid._mesh` (either `"spherical"` or
`"flat"`) to determine whether to apply deg/s ↔ m/s conversion.  This is
the same check all built-in Parcels interpolators use.  No `spherical`
parameter needed in our API.

### Unstructured grids (UxGrid)

Out of scope for the current dataset, but the design accommodates them:

- `fieldset.UV.eval()` works identically on unstructured grids (uses
  `Ux_Velocity` interpolator internally).
- `UxGrid.depth` exists (returns `self.z.values`).
- Per-depth-level `eval()` calls redo the spatial hash query D times.
  Acceptable for small D (depth-limited); batch optimisation available
  later if needed.

### Warm-starting: dropped for now

Every kernel call cold-starts from equilibrium (`y0=None`).  The ODE
converges quickly (typical `t_span=(0, 120)` is sufficient).

**Future path:** Store drifter state (theta, phi, xd, yd, thetad, phid) as
Parcels particle variables on a custom `DDParticle` class.  The kernel has
full access to `particles`, so it can read/write state directly — no
closure capture needed.  This gives warm-starting, diagnostics, and
restartability in one shot.  Implement when the performance cost of
cold-starting becomes measurable.

### Public API of `parcels_v4.py`

```python
def DDAdvectEE(particles, fieldset, *, dd):
    """Parcels kernel: advect particles using drogued-drifter steady-state
    drift velocity (Euler forward).

    Profile extraction uses ``fieldset.UV.eval()`` per depth level,
    leveraging Parcels' native interpolation (handles A-grids, C-grids,
    curvilinear, and unstructured grids, including vector rotation and
    spherical mesh conversion).

    Each call cold-starts the ODE from equilibrium.  Spherical/flat
    mesh is auto-detected from ``fieldset.U.grid._mesh``.

    Only samples depths up to the drogue length plus one grid cell
    to avoid unnecessary work on deep levels.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet.
        dd: DroguedDrifter instance (bind via ``functools.partial``).

    Usage::

        from functools import partial
        from drogued_drifters.parcels_v4 import DDAdvectEE

        dd = DroguedDrifter()
        fieldset = FieldSet.from_sgrid_conventions(ds, mesh="spherical")
        pset = ParticleSet(fieldset=fieldset, pclass=Particle, ...)
        pset.execute(
            kernels=[partial(DDAdvectEE, dd=dd), DeleteOOB],
            dt=DT, runtime=RUNTIME,
        )
    """
```

`dd` is a keyword-only argument bound at the call site via
`functools.partial`.  The kernel is a plain function with an explicit
signature — no wrapper factory.  Parcels v4 kernels are plain callables
(no JIT, no signature introspection), so `partial` works directly.

<!-- TODO (upstream): Propose a ``fieldset.UV.eval_profile(time, lat, lon,
z_levels)`` method to Parcels — grid-agnostic profile sampling as a
first-class concept.  More broadly, Parcels could support truly 2D
particles (no artificial z=0) that are agnostic of the vertical dimension,
treating depth as an extra xarray dim that broadcasts over.  Out of scope
for this repo, but an essential takeaway for Parcels v4 development. -->

## Implementation steps

### Step 1: Create `parcels_v4.py`

Write `src/drogued_drifters/parcels_v4.py`:

```python
import numpy as np

_DEG2M = 1852.0 * 60.0


def make_profile_sampler(depth_levels, U_profiles, V_profiles):
    """Build a fast sample_uv(z) interpolator from pre-sampled profiles.
    (Moved from drifter.py — see existing implementation.)
    """
    ...


def DDAdvectEE(particles, fieldset, *, dd):
    """Parcels kernel: drogued-drifter advection (Euler forward)."""
    lat = np.asarray(particles.lat)
    lon = np.asarray(particles.lon)
    time = particles.time
    N = len(lat)

    is_spherical = fieldset.U.grid._mesh == "spherical"
    drogue_depth = dd.physics.l

    # Depth levels: surface to first level beyond drogue depth (or all if
    # drogue covers the full water column).  At least 2 for interpolation.
    all_depths = np.asarray(fieldset.U.grid.depth, dtype=float)
    cutoff = min(np.searchsorted(all_depths, drogue_depth, side="right") + 1,
                 len(all_depths))
    depth_levels = all_depths[: max(cutoff, 2)]
    D = len(depth_levels)

    # Grid-agnostic profile extraction via default VectorField interpolator
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

    # Convert to z-up ascending for make_profile_sampler
    depth_up = -depth_levels[::-1]
    U_profiles = U_profiles[::-1]
    V_profiles = V_profiles[::-1]

    sample_uv = make_profile_sampler(depth_up, U_profiles, V_profiles)

    # Cold-start from equilibrium
    xd_ms, yd_ms, _, _ = dd.get_final_drift_batch(sample_uv=sample_uv)

    # Position update (Euler forward)
    if is_spherical:
        cos_lat = np.cos(np.deg2rad(lat))
        particles.dlon += xd_ms / (_DEG2M * cos_lat) * particles.dt
        particles.dlat += yd_ms / _DEG2M * particles.dt
    else:
        particles.dlon += xd_ms * particles.dt
        particles.dlat += yd_ms * particles.dt
```

### Step 2: Remove Parcels code from `drifter.py`

- Delete `make_dd_velocity_interpolator` (lines 60–216).
- Delete `make_profile_sampler` (lines 19–57) — moved to `parcels_v4.py`.
- Delete the comment at line 18 about Parcels depth convention.

### Step 3: Update imports and call sites

- `examples/idealized_flow/02_sheared_jet_parcels.ipynb`:
  - Change import to `from drogued_drifters.parcels_v4 import DDAdvectEE`
  - Add `from functools import partial`
  - Remove `dd_warm_state = {}` and the `vector_interp_method` assignment
  - Replace `kernels=[AdvectionEE, DeleteOOB]` with
    `kernels=[partial(DDAdvectEE, dd=dd), DeleteOOB]`
  - Drop separate `fieldset_pp` for DD run (the DD fieldset keeps its
    default interpolator now — no custom one installed)
- `tests/test_drifter_parcels.py`:
  - Update `make_profile_sampler` import: `drogued_drifters.drifter` →
    `drogued_drifters.parcels_v4` (8 tests, unchanged logic).
  - Delete 5 `test_make_dd_velocity_interpolator_*` tests (lines 181–305)
    — the factory no longer exists.
  - Add `DDAdvectEE` integration tests (see Step 4).

### Step 4: Add integration tests with a synthetic FieldSet

Add tests to `tests/test_drifter_parcels.py`:

1. **Uniform-flow test:**  FieldSet with constant U=0.5, V=0 at all depths.
   Run DD kernel for one step.  Drifter should move at ~0.5 m/s (buoy and
   drogue see the same current → pole vertical → drift = current).

2. **Sheared-flow test:**  Surface U=1.0, bottom U=0.  Drift velocity
   should be between 0 and 1.

### Step 5: Run the example notebook

Execute `examples/idealized_flow/02_sheared_jet_parcels.ipynb` with
papermill to verify the full pipeline.

### Step 6: Clean up

- Run black, check tests pass.

## Test plan

| Test | What it validates |
|---|---|
| Existing `test_make_profile_sampler_*` (8 tests) | Import updated to `parcels_v4`; logic unchanged |
| **New:** `test_uniform_flow_dd_kernel` | Kernel + uniform FieldSet → drift ≈ current |
| **New:** `test_sheared_flow_dd_kernel` | Kernel + sheared FieldSet → drift between surface and bottom |
| **New:** `test_dd_kernel_spherical_auto` | Kernel auto-detects `mesh="spherical"` and produces correct deg/s position updates |
| Example notebook execution | End-to-end: FieldSet → kernel → Parcels advection loop → trajectories |

## Performance

The kernel calls `fieldset.UV.eval()` D times per timestep (D = number of
sampled depth levels).  Each call does a full grid search + interpolation.
With depth limiting (D ≈ 3–5 for typical drogue depths), the overhead is
small compared to the ODE solve.

Cold-starting from equilibrium every call adds ~120 s of simulated ODE
integration.  For typical Parcels timesteps (dt = 300–3600 s), this is
acceptable.  Profile if it becomes a bottleneck; warm-starting via particle
variables is the planned optimisation.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `field.grid.depth` marked "v3 compat, may be removed" | Only public way to get depth levels.  Fallback: `field.grid._ds["depth"].values` or accept as parameter. |
| `grid._mesh` is private | Stable — used by every built-in Parcels interpolator.  Single point of access, easy to update if renamed. |
| No warm-starting → slower per-step | ODE converges in ~120 s.  Depth limiting keeps profile extraction cheap.  Measure before optimising. |
| Euler forward only | DD model finds steady state, so the "velocity" is quasi-static.  RK4 available later if needed. |

## Out of scope

- **Parcels upstream engagement.**  This plan produces the clean module;
  the actual PR/discussion is separate.
- **Upstream feature requests** (to document as TODOs in code):
  - `fieldset.UV.eval_profile(time, lat, lon, z_levels)` — grid-agnostic
    profile sampling as a first-class Parcels concept.
  - Truly 2D particles that don't require artificial z=0 placement.
  - Treating depth as an extra xarray dimension for broadcasting when z is
    not a particle coordinate.
- **Unstructured grid testing.**  Design supports it; no test dataset.
- **Warm-starting.**  Deferred to particle-variable approach.
- **RK4.**  Euler forward is sufficient for quasi-static drift velocity.
