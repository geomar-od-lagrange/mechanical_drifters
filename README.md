# Mechanical Ocean Drifters

Lagrangian mechanics models for ocean drifters. Each model defines a Lagrangian,
derives equations of motion symbolically with SymPy, and integrates numerically
with SciPy. The package includes:

- **DroguedDrifter** — a surface buoy connected by a rigid pole to a subsurface
  drogue (4 generalized coordinates). The primary model. See
  [docs/drifter-model.md](docs/drifter-model.md).
- **PointSurfaceDrifter** — a point particle at the surface with quadratic drag
  (2 generalized coordinates). Baseline comparison model whose steady-state
  drift equals the surface current. See
  [docs/point-surface-drifter.md](docs/point-surface-drifter.md).

New models can be added by subclassing `LagrangianMechanicsModel` — see
[docs/architecture.md](docs/architecture.md).

## Quick start

```python
from mechanical_drifters import DroguedDrifter
import numpy as np

dd = DroguedDrifter()  # default Callies et al. geometry

def sample_uv(z):
    """Return (U, V) velocity at depth z. z=0 is surface, negative is below."""
    return 0.3, 0.0  # uniform 0.3 m/s eastward

ds = dd.get_full_solution(sample_uv, t_span=(0, 120), t_eval=np.arange(0, 121))
ds.x.plot()  # buoy x position over time
```

`get_full_solution` returns an `xarray.Dataset` with time as coordinate and
spherical state variables as data variables (converted from internal
stereographic representation).  `get_final_drift` returns just the steady-state
drift velocity `(xd, yd, max_accel)`.  All initial conditions are keyword
arguments with sensible defaults (drogue hanging straight down, at rest).

Ocean currents are supplied as a `sample_uv(z)` callable passed to each solve
method:

```python
import numpy as np

def my_uv(z):
    # z: scalar or (N,) array of depths [m], positive upward
    # look up current from ocean_data at each depth
    ...
    return U, V  # eastward, northward [m/s], same shape as z

xd, yd, max_accel = dd.get_final_drift(my_uv, t_span=(0, 120))
```

## Parcels integration

For Lagrangian particle tracking with [Parcels v4](https://github.com/OceanParcels/parcels),
use `make_kernel(model)`:

```python
from mechanical_drifters import DroguedDrifter
from mechanical_drifters.parcels import make_kernel

dd = DroguedDrifter()
kernel = make_kernel(dd)  # works for any LagrangianMechanicsModel
pset.execute(kernels=[kernel, DeleteOOB], dt=300, runtime=86400)
```

The kernel computes the steady-state drift velocity internally and applies its
own Euler-forward position update. See
[docs/parcels-v4-coupling.md](docs/parcels-v4-coupling.md) for details.

## Setup

Requires Python >= 3.11. Parcels is pinned to a specific development version
(see `pyproject.toml` for the exact commit).

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

### Drogued drifter
- [`01_eom_exploration`](examples/drogued_drifter/01_eom_exploration.ipynb) -- mass matrix, force vector, and parameter sensitivity
- [`02_synthetic_flow_profiles`](examples/drogued_drifter/02_synthetic_flow_profiles.ipynb) -- synthetic velocity profiles and DD response
- [`03_sheared_jet_parcels`](examples/drogued_drifter/03_sheared_jet_parcels.ipynb) -- DD kernel in a sheared jet (Parcels)
- [`04_wave_orbitals`](examples/drogued_drifter/04_wave_orbitals.ipynb) -- wave orbital effects

### Point surface drifter
- [`01_surface_tracking`](examples/point_drifter/01_surface_tracking.ipynb) -- steady-state convergence in uniform and sheared flows

### Baltic validation

The Baltic notebooks form a complete validation pipeline. Raw GPS tracks from
six drifters deployed in the Baltic Sea are cleaned to extract science periods
(when drogues were attached). CMEMS Eulerian currents and wave fields are
fetched, then combined with Stokes drift profiles to build effective current
fields. Drogued drifter and point-particle simulations are run in these fields,
and the results are validated against the observed trajectories using separation
distances and short-horizon skill scores.

- [`00_extract_science_periods`](examples/baltic_validation/00_extract_science_periods.ipynb) -- clean observed drifter data
- [`01_fetch_cmems_data`](examples/baltic_validation/01_fetch_cmems_data.ipynb) -- download CMEMS Baltic Sea data
- [`02_derive_effective_currents`](examples/baltic_validation/02_derive_effective_currents.ipynb) -- Eulerian + Stokes drift
- [`03_validate_effective_current_fields`](examples/baltic_validation/03_validate_effective_current_fields.ipynb) -- field validation
- [`04_run_simulations`](examples/baltic_validation/04_run_simulations.ipynb) -- DD and point-particle simulations
- [`05_validation_plots`](examples/baltic_validation/05_validation_plots.ipynb) -- comparison with observations
- [`06_run_short_simulations`](examples/baltic_validation/06_run_short_simulations.ipynb) -- 12h re-seeded segments
- [`07_short_horizon_skill`](examples/baltic_validation/07_short_horizon_skill.ipynb) -- skill scores vs lead time
