---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Simulation: drogued drifter and point particle runs

Run drogued drifter and point particle simulations from observed
deployment positions. Six drifters (D298–D303) with 3 m drogues are
simulated in effective currents (Eulerian + Stokes) using
`data/cmems/effective_currents.nc`. Surface and 3 m point particles
provide baselines. All output is saved as zarr for fast downstream use.

## Parameters

```python tags=["parameters"]
CSV_PATH = "data/drifters_science.csv"
EFFECTIVE_CURRENTS_PATH = "data/cmems/effective_currents.nc"
OUTPUT_DIR = "output"
DROGUE_DEPTH = 3.0
DT = 300.0
RUNTIME_HOURS = 288
OUTPUTDT = 3600.0
```

## Imports

```python
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from parcels import FieldSet, Particle, ParticleFile, ParticleSet, StatusCode
from parcels.kernels import AdvectionEE

from mechanical_drifters.models.drogued_drifter import DroguedDrifter
from mechanical_drifters.parcels import make_kernel
```

## Load science observations

Extract deployment position and time as the first record per drifter
within the effective currents time window (2023-04-24 onwards).

```python
df = pd.read_csv(CSV_PATH, parse_dates=["date_UTC"])
drifter_ids = sorted(df["D_number"].unique())

# First record per drifter within effective-currents time window.
# 2023-04-24 is the start date of the effective current fields
# (derived in notebook 02 from CMEMS data availability).
df_window = df[df["date_UTC"] >= "2023-04-24"]
deployments = {}
for d_num in drifter_ids:
    first = df_window[df_window["D_number"] == d_num].iloc[0]
    deployments[d_num] = {
        "lon": first["Longitude"],
        "lat": first["Latitude"],
        "time": first["date_UTC"],
    }

for d_num, dep in deployments.items():
    print(f"  {d_num}: ({dep['lat']:.4f}N, {dep['lon']:.4f}E) at {dep['time']}")
```

## Load effective currents and build FieldSet

The pre-computed effective currents (`U_eff`, `V_eff`) already include
Eulerian + Stokes drift. A z=0 surface layer is prepended by copying
the shallowest level so that Parcels can interpolate from the surface.

```python
ds_eff_raw = xr.open_dataset(EFFECTIVE_CURRENTS_PATH)[["U_eff", "V_eff"]].load()

# Prepend z=0 surface layer (copy of shallowest level)
ds_z0 = ds_eff_raw.isel(depth=0).assign_coords(depth=0.0)
ds_eff = xr.concat([ds_z0, ds_eff_raw], dim="depth").rename(
    {"U_eff": "U", "V_eff": "V"}
).fillna(0.0)

print(ds_eff)
print("depth levels:", ds_eff.depth.values)
```

```python
# Attach sgrid topology attribute required by FieldSet.from_sgrid_conventions
ds_eff["grid"] = xr.DataArray(
    data=0,
    attrs={
        "cf_role": "grid_topology",
        "topology_dimension": 2,
        "node_dimensions": "longitude latitude",
        "face_dimensions": (
            "longitude:longitude (padding: none) "
            "latitude:latitude (padding: none)"
        ),
        "vertical_dimensions": "depth:depth (padding: none)",
        "node_coordinates": "longitude latitude",
    },
)

fieldset = FieldSet.from_sgrid_conventions(ds_eff, mesh="spherical")
print("FieldSet built.")
```

## Set up drogued drifter kernel

The drifter kernel extracts the full velocity profile at each
particle position using `fieldset.UV.eval()` and runs
`model.integrate()` to obtain the drift
velocity. Spherical mesh is auto-detected.

```python
dd = DroguedDrifter()
dd_kernel = make_kernel(dd)
print("DroguedDrifter kernel created.")
```

## Helper kernel and release arrays

```python
def DeleteOOB(particles, fieldset):
    state = np.asarray(particles.state)
    oob = (state == StatusCode.ErrorOutOfBounds) | (state == StatusCode.ErrorThroughSurface)
    if np.any(oob):
        particles.state = np.where(oob, StatusCode.Delete, state)


RUNTIME = RUNTIME_HOURS * 3600
SURFACE_DEPTH = float(ds_eff.depth.isel(depth=0))  # 0.0 after prepending z=0
DROGUE_DEPTH_LEVEL = float(ds_eff.depth.sel(depth=DROGUE_DEPTH, method="nearest"))

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

release_lons = [deployments[d]["lon"] for d in drifter_ids]
release_lats = [deployments[d]["lat"] for d in drifter_ids]
release_times = [np.datetime64(deployments[d]["time"]) for d in drifter_ids]

print(f"Surface depth: {SURFACE_DEPTH} m")
print(f"Drogue depth level: {DROGUE_DEPTH_LEVEL} m")
print(f"Runtime: {RUNTIME_HOURS} h = {RUNTIME} s")
```

## Run drogued drifter simulation

```python
dd_store = str(output_dir / "sim_drogued_drifter.zarr")
shutil.rmtree(dd_store, ignore_errors=True)

pset_dd = ParticleSet(
    fieldset=fieldset,
    pclass=Particle,
    lon=release_lons,
    lat=release_lats,
    z=[SURFACE_DEPTH] * len(drifter_ids),
    time=release_times,
)
pset_dd.execute(
    kernels=[dd_kernel, DeleteOOB],
    dt=DT,
    runtime=RUNTIME,
    output_file=ParticleFile(store=dd_store, outputdt=OUTPUTDT),
    verbose_progress=False,
)
print(f"Saved: {dd_store}")
```

## Run surface point particle simulation

```python
surface_store = str(output_dir / "sim_surface.zarr")
shutil.rmtree(surface_store, ignore_errors=True)

pset_surface = ParticleSet(
    fieldset=fieldset,
    pclass=Particle,
    lon=release_lons,
    lat=release_lats,
    z=[SURFACE_DEPTH] * len(drifter_ids),
    time=release_times,
)
pset_surface.execute(
    kernels=[AdvectionEE, DeleteOOB],
    dt=DT,
    runtime=RUNTIME,
    output_file=ParticleFile(store=surface_store, outputdt=OUTPUTDT),
    verbose_progress=False,
)
print(f"Saved: {surface_store}")
```

## Run drogue-depth point particle simulation

```python
drogue_store = str(output_dir / "sim_3m.zarr")
shutil.rmtree(drogue_store, ignore_errors=True)

pset_drogue = ParticleSet(
    fieldset=fieldset,
    pclass=Particle,
    lon=release_lons,
    lat=release_lats,
    z=[DROGUE_DEPTH_LEVEL] * len(drifter_ids),
    time=release_times,
)
pset_drogue.execute(
    kernels=[AdvectionEE, DeleteOOB],
    dt=DT,
    runtime=RUNTIME,
    output_file=ParticleFile(store=drogue_store, outputdt=OUTPUTDT),
    verbose_progress=False,
)
print(f"Saved: {drogue_store}")
```

## Summary

```python
import os

stores = [
    ("Drogued drifter", dd_store),
    ("Surface point particle", surface_store),
    ("3 m point particle", drogue_store),
]

print("Output file sizes:")
for label, store in stores:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(store)
        for f in filenames
    )
    ds_out = xr.open_zarr(store)
    n_traj = ds_out.sizes["trajectory"]
    n_obs = ds_out.sizes["obs"]
    print(f"  {label}: {n_traj} trajectories x {n_obs} obs, {total / 1024:.1f} kB  -> {store}")
```
