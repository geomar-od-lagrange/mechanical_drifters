# 2025 Drogued Drifters

Lagrangian model for drogued ocean drifters. A drogued drifter is a surface buoy
connected by a rigid pole to a subsurface drogue (drag element). The model
derives equations of motion from a Lagrangian formulation using SymPy, then
integrates numerically with SciPy.

The state vector is `[x, y, theta, phi, xd, yd, thetad, phid]` where `(x, y)`
is the buoy position, `theta` is the tether polar angle (theta=pi means drogue
hangs straight down), and `phi` is the azimuthal angle.

## Quick start

```python
from drogued_drifters import DroguedDrifter
import numpy as np

dd = DroguedDrifter()  # default Callies et al. geometry

ds = dd.get_full_solution(t_span=(0, 120), t_eval=np.arange(0, 121))
ds.x.plot()  # buoy x position over time
```

Both `get_full_solution` and `get_final_drift` return an `xarray.Dataset` with
time as coordinate and state variables `x, y, theta, phi, xd, yd, thetad, phid`
as data variables. All initial conditions are keyword arguments with sensible
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

## Setup

```shell
$ pixi install
```

## Run tests

```shell
$ pixi run pytest -v
```

## Notebooks

- [`01_download_baltic_data`](notebooks/01_download_baltic_data.ipynb) -- download CMEMS Baltic Sea velocity data
- [`02_test_with_sheared_flow`](notebooks/02_test_with_sheared_flow.ipynb) -- synthetic sheared flow test case
- [`03_test_with_cmems_data`](notebooks/03_test_with_cmems_data.ipynb) -- simulation with real CMEMS currents
