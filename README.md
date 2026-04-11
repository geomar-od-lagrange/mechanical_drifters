# 2025 Drogued Drifters

Lagrangian model for drogued ocean drifters. A drogued drifter is a surface buoy
connected by a rigid pole to a subsurface drogue (drag element). The model
derives equations of motion from a Lagrangian formulation using SymPy, then
integrates numerically with SciPy.

The state vector internally uses stereographic coordinates for the pole
direction: `[x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo]` where
`(x, y)` is the buoy position, `(u_stereo, v_stereo)` are the stereographic
coordinates for the pole direction (equilibrium at origin), and
`(xd, yd, ud_stereo, vd_stereo)` are the time derivatives. The public API
converts to and from spherical angles `(theta, phi)` for user-facing methods.

## Quick start

```python
from drogued_drifters import DroguedDrifter
import numpy as np

dd = DroguedDrifter()  # default Callies et al. geometry

ds = dd.get_full_solution(t_span=(0, 120), t_eval=np.arange(0, 121))
ds.x.plot()  # buoy x position over time
```

Both `get_full_solution` and `get_final_drift` return an `xarray.Dataset` with
time as coordinate and spherical state variables `x, y, theta, phi, xd, yd, thetad, phid`
as data variables (converted from internal stereographic representation). All initial conditions are keyword arguments with sensible
defaults (drogue hanging nearly straight down, at rest).

Custom velocity fields are passed as a callback:

```python
from functools import partial

def my_uv(*, t, z_d, y_b, x_b, ocean_data):
    # look up currents from ocean_data at position and depth
    ...
    return U_b, V_b, U_d, V_d

dd = DroguedDrifter(get_uv=partial(my_uv, ocean_data=my_dataset))
```

## Parcels integration

For Lagrangian particle tracking with [Parcels v4](https://github.com/OceanParcels/parcels),
use the `DDAdvectEE` kernel:

```python
from drogued_drifters.parcels_v4 import make_dd_kernel

dd = DroguedDrifter()
kernel = make_dd_kernel(dd)
pset.execute(kernels=[kernel, DeleteOOB], dt=300, runtime=86400)
```

See [docs/parcels-v4-coupling.md](docs/parcels-v4-coupling.md) for details.

## Setup

```shell
$ pixi install
```

## Run tests

```shell
$ pixi run pytest -v
```

## Examples

### Idealized flow
- [`01_synthetic_flow_profiles`](examples/idealized_flow/01_synthetic_flow_profiles.ipynb) -- synthetic velocity profiles and DD response
- [`02_sheared_jet_parcels`](examples/idealized_flow/02_sheared_jet_parcels.ipynb) -- DD kernel in a sheared jet (Parcels)
- [`03_drogued_drifter_in_wave_orbitals`](examples/idealized_flow/03_drogued_drifter_in_wave_orbitals.ipynb) -- wave orbital effects

### Baltic drifters
- [`00_extract_science_periods`](examples/baltic_drifters/00_extract_science_periods.ipynb) -- clean observed drifter data
- [`01_fetch_cmems_data`](examples/baltic_drifters/01_fetch_cmems_data.ipynb) -- download CMEMS Baltic Sea data
- [`02_derive_effective_currents`](examples/baltic_drifters/02_derive_effective_currents.ipynb) -- Eulerian + Stokes drift
- [`03_validate_effective_current_fields`](examples/baltic_drifters/03_validate_effective_current_fields.ipynb) -- field validation
- [`04_run_simulations`](examples/baltic_drifters/04_run_simulations.ipynb) -- DD and point-particle simulations
- [`05_validation_plots`](examples/baltic_drifters/05_validation_plots.ipynb) -- comparison with observations
- [`06_run_short_simulations`](examples/baltic_drifters/06_run_short_simulations.ipynb) -- 12h re-seeded segments
- [`07_short_horizon_skill`](examples/baltic_drifters/07_short_horizon_skill.ipynb) -- skill scores vs lead time
