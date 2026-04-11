# Parcels v4 coupling

`drogued_drifters.parcels_v4` provides a single Parcels kernel that
advects particles using the drogued-drifter steady-state drift velocity.
All Parcels-specific code is isolated in this module — the core physics
(`drifter.py`, `lagrange_model.py`) has no Parcels dependency.

## Quick start

```python
import numpy as np
from parcels import FieldSet, Particle, ParticleSet, StatusCode

from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.parcels_v4 import make_dd_kernel

dd = DroguedDrifter()
kernel = make_dd_kernel(dd)

def DeleteOOB(particles, fieldset):
    state = np.asarray(particles.state)
    oob = (state == StatusCode.ErrorOutOfBounds) | (state == StatusCode.ErrorThroughSurface)
    if np.any(oob):
        particles.state = np.where(oob, StatusCode.Delete, state)

fieldset = FieldSet.from_sgrid_conventions(ds, mesh="spherical")  # ds: xarray Dataset with U, V
pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=lons, lat=lats, z=[0] * len(lons))
pset.execute(kernels=[kernel, DeleteOOB], dt=300, runtime=86400)
```

## How the kernel works

Each timestep, `DDAdvectEE` does four things:

1. **Extract velocity profiles** at the particle position by calling
   `fieldset.UV.eval(time, z, lat, lon)` once per depth level, from
   the surface down to just past the drogue depth.

2. **Build a fast depth interpolator** (`make_profile_sampler`) from
   the sampled profiles.

3. **Run the drifter ODE** via `dd.get_final_drift_batch(sample_uv=...)`
   to find the steady-state buoy drift velocity. This cold-starts from
   un-sheared equilibrium (pole hanging vertical) every call.

4. **Euler forward position update** using the drift velocity.

### Why `fieldset.UV.eval()` and not individual `field.eval()`

Calling `fieldset.UV.eval()` uses the **VectorField** interpolator,
which:

- Applies C-grid vector rotation (the Jacobian-based transform in
  `CGrid_Velocity`) — individual `Field.eval()` uses the scalar
  interpolator and skips this.
- Handles the spherical m/s → deg/s conversion.
- Works identically across A-grids, C-grids, curvilinear, and
  unstructured grids.

The kernel reads the result in the interpolator's output units, converts
to m/s for the physics, and converts back for the position update.

### Depth limiting

Only depths from the surface to one grid level past the drogue depth are
sampled.  For a typical 3 m drogue on a 20-level grid, this means 3–5
`eval()` calls instead of 20.  At least 2 levels are always sampled so
the profile interpolator has something to work with.

### Depth convention

Parcels stores depth as positive-downward in `field.grid.depth`.  The
drifter physics expects z-up (0 = surface, negative = below).  The
kernel reverses and negates after sampling.

## Why a kernel, not a custom interpolator

An alternative approach is to override
`fieldset.UV.vector_interp_method` with a custom callback that calls
individual `Field.eval()` to extract the velocity profile.  This is
attractive because it plugs into the existing advection kernels
(`AdvectionEE`, `AdvectionRK4`), but it lacks several things:

- **Grid generality.** Profile extraction would need to use private
  internals like `_get_corner_data_Agrid`, hard-coding one grid type.
  C-grids, curvilinear, and unstructured grids would require separate
  code paths.
- **Vector rotation.** Individual `Field.eval()` uses the scalar
  interpolator (`XLinear`), which skips the Jacobian-based C-grid
  vector rotation that lives in `CGrid_Velocity`.  The custom callback
  would have to re-implement this.
- **Spherical conversion.** The callback would also have to duplicate the
  m/s → deg/s logic that Parcels' built-in VectorField interpolators
  already handle.

The kernel approach avoids all of these: it calls `fieldset.UV.eval()`
(the VectorField-level method) and lets Parcels handle interpolation,
rotation, and unit conversion for whatever grid type is present.

**Trade-off:** We own the advection step (Euler forward) instead of
composing with `AdvectionEE`.  Since the DD model finds the steady-state
drift velocity, the "velocity" is quasi-static and Euler forward is the
natural scheme.  Upgrade to RK4 later if needed.

## `make_dd_kernel` — why not `functools.partial`

Parcels v4 alpha checks `isinstance(kernel, types.FunctionType)` in
`Kernel.__init__` and rejects `functools.partial` objects.
`make_dd_kernel(dd)` returns a plain closure instead:

```python
def make_dd_kernel(dd):
    def _kernel(particles, fieldset):
        DDAdvectEE(particles, fieldset, dd=dd)
    return _kernel
```

If a future Parcels release relaxes this check, `partial(DDAdvectEE, dd=dd)`
would work and `make_dd_kernel` could be dropped.

## Spherical vs flat mesh

The kernel auto-detects the mesh type from `fieldset.U.grid._mesh`
(`"spherical"` or `"flat"`).  No parameter is needed.

- **Spherical:** `fieldset.UV.eval()` returns deg/s.  The kernel
  converts to m/s (`u_ms = u_degs * DEG2M * cos(lat)`) before the
  physics, and back to deg/s for the position update.
- **Flat:** velocities are already in m/s throughout.

## Cold start vs warm start

Every kernel call starts the ODE from un-sheared equilibrium (pole vertical, at
rest).  The ODE converges to steady state within the default
`t_span=(0, 120)` seconds for typical ocean conditions.

Warm-starting (reusing the previous pole angle and angular velocity)
would save ODE integration time but requires storing per-particle state.
The planned approach is to add custom particle variables on a
`DDParticle` class — the kernel has full access to `particles` and
can read/write state directly.  This is deferred until cold-start cost
is measurable.

## Error handling

Out-of-bounds particles should be caught by a `DeleteOOB` recovery
kernel placed after the DD kernel in the kernel list:

```python
def DeleteOOB(particles, fieldset):
    state = np.asarray(particles.state)
    oob = (state == StatusCode.ErrorOutOfBounds) | (state == StatusCode.ErrorThroughSurface)
    if np.any(oob):
        particles.state = np.where(oob, StatusCode.Delete, state)
```

## Parcels API dependencies

The kernel relies on two semi-private Parcels attributes:

| Attribute | Used for | Risk | Fallback |
|---|---|---|---|
| `field.grid.depth` | Depth level array | Marked "v3 compat, may be removed" | `field.grid._ds["depth"].values`, or accept depth levels as a parameter |
| `field.grid._mesh` | Spherical/flat auto-detection | Private but stable — used by every built-in Parcels interpolator | Single point of access, easy to update if renamed |

## Upstream considerations

The current approach calls `fieldset.UV.eval()` D times per particle
per timestep.  Each call redoes the full spatial search.  For the Parcels
developers, useful additions would be:

- `fieldset.UV.eval_profile(time, lat, lon, z_levels)` — sample a full
  vertical profile in one call, reusing the horizontal search across
  depth levels.
- Truly 2D particles that don't require an artificial `z=0` coordinate.
- Treating depth as an extra xarray dimension that broadcasts when z is
  not a particle coordinate.
