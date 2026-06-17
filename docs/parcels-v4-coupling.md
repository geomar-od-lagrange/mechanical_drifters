# Parcels v4 coupling

`mechanical_drifters.parcels` provides a Parcels kernel factory that
advects particles using the drift velocity from any
`LagrangianMechanicsModel` subclass. All Parcels-specific code is
isolated in this module -- the core physics (`models/`, `eom.py`) has
no Parcels dependency.

## Quick start

```python
import numpy as np
from parcels import FieldSet, Particle, ParticleSet, StatusCode

from mechanical_drifters.models.drogued_drifter import DroguedDrifter
from mechanical_drifters.parcels import make_kernel

dd = DroguedDrifter()
kernel = make_kernel(dd)

fieldset = FieldSet.from_sgrid_conventions(ds, mesh="spherical")
pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=lons, lat=lats, z=[0] * len(lons))
pset.execute(kernels=[kernel], dt=300, runtime=86400)
```

## How the kernel works

Each timestep, the kernel does four things:

1. **Extract velocity profiles** at the particle position by calling
   `fieldset.UV.eval(time, z, lat, lon)` once per depth level.

2. **Build a fast depth interpolator** (`_make_profile_sampler`) from
   the sampled profiles.

3. **Run the drifter ODE** via `model.integrate(sample_uv)` to find
   the drift velocity. Then extract it with `model.drift_velocity(Y[-1])`.

4. **Euler forward position update** using the drift velocity.

### Depth handling

Models that need depth sampling (e.g. DroguedDrifter) expose a
`_max_depth` attribute. The kernel reads it with
`getattr(model, '_max_depth', 0.0)`, defaulting to 0.0 (surface only)
when absent (PointSurfaceDrifter). `_extract_profiles` samples every
fieldset level from the top of the `depth` axis down to `_max_depth`
(plus one cell), then converts the oceanographic `depth` axis (positive
downward) to the models' $z$-positive-up frame via
`depth_up = -depth[::-1]`.

**Sampling above the surface (air).** Because the conversion is a simple
negation, a model that samples at $z > 0$ (e.g. SparBuoySimple, whose
emergent column feels wind) reads it by encoding the **air column at
negative `depth`** in the fieldset: the negation lands it at $z > 0$
exactly where the model queries, while the water column at positive
`depth` maps to $z \le 0$. One signed-`depth` fieldset thus carries both
media. See [spar-buoy.md](spar-buoy.md) and the
[`02_parcels_wind_and_current`](../examples/spar_buoy/02_parcels_wind_and_current.ipynb)
example.

### Spherical vs flat mesh

The kernel auto-detects the mesh type from `fieldset.U.grid._mesh`.
Spherical meshes get deg/s to m/s conversion; flat meshes use m/s
directly.

## Numba backend

```python
dd = DroguedDrifter(backend="numba")
kernel = make_kernel(dd)
```

`backend="numba"` JIT-compiles the qdd evaluator for ~25x speedup.
