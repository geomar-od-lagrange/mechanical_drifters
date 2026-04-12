# 2025 Drogued Drifters

Lagrangian model for drogued ocean drifters. A drogued drifter is an
oceanographic instrument consisting of a GPS-tracked surface buoy connected by a
rigid pole to a subsurface drogue (a cross-shaped drag element). The drogue
anchors the drifter to a target depth so that it measures the current there,
while the buoy experiences surface drag from wind and waves. The model derives
equations of motion from a Lagrangian formulation using SymPy, then integrates
numerically with SciPy. See [docs/drifter-model.md](docs/drifter-model.md) for
the full physics and API reference.

## Quick start

```python
from drogued_drifters import DroguedDrifter
import numpy as np

dd = DroguedDrifter()  # default Callies et al. geometry

ds = dd.get_full_solution(t_span=(0, 120), t_eval=np.arange(0, 121))
ds.x.plot()  # buoy x position over time
```

`get_full_solution` returns an `xarray.Dataset` with time as coordinate and
spherical state variables as data variables (converted from internal
stereographic representation).  `get_final_drift` returns just the steady-state
drift velocity `(xd, yd, max_accel)`.  All initial conditions are keyword
arguments with sensible defaults (drogue hanging straight down, at rest).

Custom velocity fields are passed as a `sample_uv(z)` callable:

```python
import numpy as np

def my_uv(z):
    # z: scalar or (N,) array of depths [m], positive upward
    # look up current from ocean_data at each depth
    ...
    return U, V  # eastward, northward [m/s], same shape as z

dd = DroguedDrifter(sample_uv=my_uv)
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

The DD kernel computes the drift velocity internally and applies its own Euler-forward position update — it does not use Parcels advection schemes like `AdvectionRK4`, which are only relevant for point-particle comparisons.

See [docs/parcels-v4-coupling.md](docs/parcels-v4-coupling.md) for details.

## Setup

Requires Python >= 3.11. Parcels is pinned to a specific development version
(see `pixi.toml` for the exact commit).

This project uses [pixi](https://pixi.sh), a fast package manager for
reproducible environments based on conda-forge. Install pixi first, then:

```shell
$ pixi install
```

## Run tests

```shell
$ pixi run pytest -v
```

## Examples

Notebooks are managed with [jupytext](https://jupytext.readthedocs.io/) and
paired as `.ipynb` + `.md` (Markdown format). The `.md` files are the
diff-friendly source of truth; the `.ipynb` files carry cell outputs for
rendering. Edit either — jupytext keeps them in sync.

### Equations of motion
- [`01_eom_exploration`](examples/eom_study/01_eom_exploration.ipynb) -- mass matrix, force vector, and parameter sensitivity

### Idealized flow
- [`01_synthetic_flow_profiles`](examples/idealized_flow/01_synthetic_flow_profiles.ipynb) -- synthetic velocity profiles and DD response
- [`02_sheared_jet_parcels`](examples/idealized_flow/02_sheared_jet_parcels.ipynb) -- DD kernel in a sheared jet (Parcels)
- [`03_drogued_drifter_in_wave_orbitals`](examples/idealized_flow/03_drogued_drifter_in_wave_orbitals.ipynb) -- wave orbital effects

### Baltic drifters

The Baltic notebooks form a complete validation pipeline. Raw GPS tracks from
six drifters deployed in the Baltic Sea are cleaned to extract science periods
(when drogues were attached). CMEMS Eulerian currents and wave fields are
fetched, then combined with Stokes drift profiles to build effective current
fields. Drogued drifter and point-particle simulations are run in these fields,
and the results are validated against the observed trajectories using separation
distances and short-horizon skill scores.

- [`00_extract_science_periods`](examples/baltic_drifters/00_extract_science_periods.ipynb) -- clean observed drifter data
- [`01_fetch_cmems_data`](examples/baltic_drifters/01_fetch_cmems_data.ipynb) -- download CMEMS Baltic Sea data
- [`02_derive_effective_currents`](examples/baltic_drifters/02_derive_effective_currents.ipynb) -- Eulerian + Stokes drift
- [`03_validate_effective_current_fields`](examples/baltic_drifters/03_validate_effective_current_fields.ipynb) -- field validation
- [`04_run_simulations`](examples/baltic_drifters/04_run_simulations.ipynb) -- DD and point-particle simulations
- [`05_validation_plots`](examples/baltic_drifters/05_validation_plots.ipynb) -- comparison with observations
- [`06_run_short_simulations`](examples/baltic_drifters/06_run_short_simulations.ipynb) -- 12h re-seeded segments
- [`07_short_horizon_skill`](examples/baltic_drifters/07_short_horizon_skill.ipynb) -- skill scores vs lead time
