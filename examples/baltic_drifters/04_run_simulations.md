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

<!-- #region papermill={"duration": 0.007833, "end_time": "2026-04-11T15:53:18.149142+00:00", "exception": false, "start_time": "2026-04-11T15:53:18.141309+00:00", "status": "completed"} -->
# Simulation: drogued drifter and point particle runs

Run drogued drifter and point particle simulations from observed
deployment positions. Six drifters (D298–D303) with 3 m drogues are
simulated in effective currents (Eulerian + Stokes) using
`data/cmems/effective_currents.nc`. Surface and 3 m point particles
provide baselines. All output is saved as zarr for fast downstream use.
<!-- #endregion -->

<!-- #region papermill={"duration": 0.002766, "end_time": "2026-04-11T15:53:18.156529+00:00", "exception": false, "start_time": "2026-04-11T15:53:18.153763+00:00", "status": "completed"} -->
## Parameters
<!-- #endregion -->

```python papermill={"duration": 0.008698, "end_time": "2026-04-11T15:53:18.168570+00:00", "exception": false, "start_time": "2026-04-11T15:53:18.159872+00:00", "status": "completed"} tags=["parameters"]
CSV_PATH = "data/drifters_science.csv"
EFFECTIVE_CURRENTS_PATH = "data/cmems/effective_currents.nc"
OUTPUT_DIR = "output"
DROGUE_DEPTH = 3.0
DT = 300.0
RUNTIME_HOURS = 288
OUTPUTDT = 3600.0
```

<!-- #region papermill={"duration": 0.001792, "end_time": "2026-04-11T15:53:18.172383+00:00", "exception": false, "start_time": "2026-04-11T15:53:18.170591+00:00", "status": "completed"} -->
## Imports
<!-- #endregion -->

```python papermill={"duration": 4.758609, "end_time": "2026-04-11T15:53:22.932812+00:00", "exception": false, "start_time": "2026-04-11T15:53:18.174203+00:00", "status": "completed"}
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from parcels import FieldSet, Particle, ParticleFile, ParticleSet, StatusCode
from parcels.kernels import AdvectionEE

from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.parcels_v4 import make_dd_kernel
```

<!-- #region papermill={"duration": 0.001477, "end_time": "2026-04-11T15:53:22.936094+00:00", "exception": false, "start_time": "2026-04-11T15:53:22.934617+00:00", "status": "completed"} -->
## Load science observations

Extract deployment position and time as the first record per drifter
within the effective currents time window (2023-04-24 onwards).
<!-- #endregion -->

```python papermill={"duration": 0.055943, "end_time": "2026-04-11T15:53:22.993548+00:00", "exception": false, "start_time": "2026-04-11T15:53:22.937605+00:00", "status": "completed"}
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

<!-- #region papermill={"duration": 0.001412, "end_time": "2026-04-11T15:53:22.996666+00:00", "exception": false, "start_time": "2026-04-11T15:53:22.995254+00:00", "status": "completed"} -->
## Load effective currents and build FieldSet

The pre-computed effective currents (`U_eff`, `V_eff`) already include
Eulerian + Stokes drift. A z=0 surface layer is prepended by copying
the shallowest level so that Parcels can interpolate from the surface.
<!-- #endregion -->

```python papermill={"duration": 0.236849, "end_time": "2026-04-11T15:53:23.234860+00:00", "exception": false, "start_time": "2026-04-11T15:53:22.998011+00:00", "status": "completed"}
ds_eff_raw = xr.open_dataset(EFFECTIVE_CURRENTS_PATH)[["U_eff", "V_eff"]].load()

# Prepend z=0 surface layer (copy of shallowest level)
ds_z0 = ds_eff_raw.isel(depth=0).assign_coords(depth=0.0)
ds_eff = xr.concat([ds_z0, ds_eff_raw], dim="depth").rename(
    {"U_eff": "U", "V_eff": "V"}
).fillna(0.0)

print(ds_eff)
print("depth levels:", ds_eff.depth.values)
```

```python papermill={"duration": 0.007241, "end_time": "2026-04-11T15:53:23.243830+00:00", "exception": false, "start_time": "2026-04-11T15:53:23.236589+00:00", "status": "completed"}
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

<!-- #region papermill={"duration": 0.001502, "end_time": "2026-04-11T15:53:23.246863+00:00", "exception": false, "start_time": "2026-04-11T15:53:23.245361+00:00", "status": "completed"} -->
## Set up drogued drifter kernel

The `DDAdvectEE` kernel extracts the full velocity profile at each
particle position using `fieldset.UV.eval()` and runs
`DroguedDrifter.get_final_drift_batch` to obtain the steady-state drift
velocity. Spherical mesh is auto-detected.
<!-- #endregion -->

```python papermill={"duration": 0.004067, "end_time": "2026-04-11T15:53:23.252417+00:00", "exception": false, "start_time": "2026-04-11T15:53:23.248350+00:00", "status": "completed"}
dd = DroguedDrifter()
dd_kernel = make_dd_kernel(dd)
print("DroguedDrifter kernel created.")
```

<!-- #region papermill={"duration": 0.001453, "end_time": "2026-04-11T15:53:23.255455+00:00", "exception": false, "start_time": "2026-04-11T15:53:23.254002+00:00", "status": "completed"} -->
## Helper kernel and release arrays
<!-- #endregion -->

```python papermill={"duration": 0.005352, "end_time": "2026-04-11T15:53:23.262233+00:00", "exception": false, "start_time": "2026-04-11T15:53:23.256881+00:00", "status": "completed"}
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

<!-- #region papermill={"duration": 0.001444, "end_time": "2026-04-11T15:53:23.265167+00:00", "exception": false, "start_time": "2026-04-11T15:53:23.263723+00:00", "status": "completed"} -->
## Run drogued drifter simulation
<!-- #endregion -->

```python papermill={"duration": 233.795041, "end_time": "2026-04-11T15:57:17.061716+00:00", "exception": false, "start_time": "2026-04-11T15:53:23.266675+00:00", "status": "completed"}
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

<!-- #region papermill={"duration": 0.001602, "end_time": "2026-04-11T15:57:17.065090+00:00", "exception": false, "start_time": "2026-04-11T15:57:17.063488+00:00", "status": "completed"} -->
## Run surface point particle simulation
<!-- #endregion -->

```python papermill={"duration": 5.738524, "end_time": "2026-04-11T15:57:22.805336+00:00", "exception": false, "start_time": "2026-04-11T15:57:17.066812+00:00", "status": "completed"}
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

<!-- #region papermill={"duration": 0.001569, "end_time": "2026-04-11T15:57:22.808729+00:00", "exception": false, "start_time": "2026-04-11T15:57:22.807160+00:00", "status": "completed"} -->
## Run drogue-depth point particle simulation
<!-- #endregion -->

```python papermill={"duration": 5.6119, "end_time": "2026-04-11T15:57:28.422190+00:00", "exception": false, "start_time": "2026-04-11T15:57:22.810290+00:00", "status": "completed"}
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

<!-- #region papermill={"duration": 0.001653, "end_time": "2026-04-11T15:57:28.425770+00:00", "exception": false, "start_time": "2026-04-11T15:57:28.424117+00:00", "status": "completed"} -->
## Summary
<!-- #endregion -->

```python papermill={"duration": 0.050326, "end_time": "2026-04-11T15:57:28.477722+00:00", "exception": false, "start_time": "2026-04-11T15:57:28.427396+00:00", "status": "completed"}
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
