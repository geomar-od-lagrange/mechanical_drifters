# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.004907, "end_time": "2026-04-02T14:40:18.401097+00:00", "exception": false, "start_time": "2026-04-02T14:40:18.396190+00:00", "status": "completed"}
# # Simulation: drogued drifter and point particle runs
#
# Run drogued drifter and point particle simulations from observed
# deployment positions. Six drifters (D298–D303) with 3 m drogues are
# simulated in effective currents (Eulerian + Stokes) using
# `data/cmems/effective_currents.nc`. Surface and 3 m point particles
# provide baselines. All output is saved as zarr for fast downstream use.

# %% [markdown] papermill={"duration": 0.002168, "end_time": "2026-04-02T14:40:18.406386+00:00", "exception": false, "start_time": "2026-04-02T14:40:18.404218+00:00", "status": "completed"}
# ## Parameters

# %% papermill={"duration": 0.007599, "end_time": "2026-04-02T14:40:18.416154+00:00", "exception": false, "start_time": "2026-04-02T14:40:18.408555+00:00", "status": "completed"} tags=["parameters"]
CSV_PATH = "data/drifters_science.csv"
EFFECTIVE_CURRENTS_PATH = "data/cmems/effective_currents.nc"
OUTPUT_DIR = "output"
DROGUE_DEPTH = 3.0
DT = 300.0
RUNTIME_HOURS = 288
OUTPUTDT = 3600.0

# %% [markdown] papermill={"duration": 0.001608, "end_time": "2026-04-02T14:40:18.419728+00:00", "exception": false, "start_time": "2026-04-02T14:40:18.418120+00:00", "status": "completed"}
# ## Imports

# %% papermill={"duration": 5.507222, "end_time": "2026-04-02T14:40:23.928569+00:00", "exception": false, "start_time": "2026-04-02T14:40:18.421347+00:00", "status": "completed"}
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from parcels import FieldSet, Particle, ParticleFile, ParticleSet, StatusCode
from parcels.kernels import AdvectionEE

from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.parcels_v4 import make_dd_kernel

# %% [markdown] papermill={"duration": 0.001536, "end_time": "2026-04-02T14:40:23.931845+00:00", "exception": false, "start_time": "2026-04-02T14:40:23.930309+00:00", "status": "completed"}
# ## Load science observations
#
# Extract deployment position and time as the first record per drifter
# within the effective currents time window (2023-04-24 onwards).

# %% papermill={"duration": 0.052881, "end_time": "2026-04-02T14:40:23.986199+00:00", "exception": false, "start_time": "2026-04-02T14:40:23.933318+00:00", "status": "completed"}
df = pd.read_csv(CSV_PATH, parse_dates=["date_UTC"])
drifter_ids = sorted(df["D_number"].unique())

# First record per drifter within effective-currents time window
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

# %% [markdown] papermill={"duration": 0.00144, "end_time": "2026-04-02T14:40:23.989340+00:00", "exception": false, "start_time": "2026-04-02T14:40:23.987900+00:00", "status": "completed"}
# ## Load effective currents and build FieldSet
#
# The pre-computed effective currents (`U_eff`, `V_eff`) already include
# Eulerian + Stokes drift. A z=0 surface layer is prepended by copying
# the shallowest level so that Parcels can interpolate from the surface.

# %% papermill={"duration": 0.255775, "end_time": "2026-04-02T14:40:24.246551+00:00", "exception": false, "start_time": "2026-04-02T14:40:23.990776+00:00", "status": "completed"}
ds_eff_raw = xr.open_dataset(EFFECTIVE_CURRENTS_PATH)[["U_eff", "V_eff"]].load()

# Prepend z=0 surface layer (copy of shallowest level)
ds_z0 = ds_eff_raw.isel(depth=0).assign_coords(depth=0.0)
ds_eff = xr.concat([ds_z0, ds_eff_raw], dim="depth").rename(
    {"U_eff": "U", "V_eff": "V"}
).fillna(0.0)

print(ds_eff)
print("depth levels:", ds_eff.depth.values)

# %% papermill={"duration": 0.009123, "end_time": "2026-04-02T14:40:24.258276+00:00", "exception": false, "start_time": "2026-04-02T14:40:24.249153+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.00141, "end_time": "2026-04-02T14:40:24.261169+00:00", "exception": false, "start_time": "2026-04-02T14:40:24.259759+00:00", "status": "completed"}
# ## Set up drogued drifter kernel
#
# The `DDAdvectEE` kernel extracts the full velocity profile at each
# particle position using `fieldset.UV.eval()` and runs
# `DroguedDrifter.get_final_drift_batch` to obtain the steady-state drift
# velocity. Spherical mesh is auto-detected.

# %% papermill={"duration": 0.004172, "end_time": "2026-04-02T14:40:24.266755+00:00", "exception": false, "start_time": "2026-04-02T14:40:24.262583+00:00", "status": "completed"}
dd = DroguedDrifter()
dd_kernel = make_dd_kernel(dd)
print("DroguedDrifter kernel created.")


# %% [markdown] papermill={"duration": 0.001433, "end_time": "2026-04-02T14:40:24.269783+00:00", "exception": false, "start_time": "2026-04-02T14:40:24.268350+00:00", "status": "completed"}
# ## Helper kernel and release arrays

# %% papermill={"duration": 0.005359, "end_time": "2026-04-02T14:40:24.276521+00:00", "exception": false, "start_time": "2026-04-02T14:40:24.271162+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.001424, "end_time": "2026-04-02T14:40:24.279681+00:00", "exception": false, "start_time": "2026-04-02T14:40:24.278257+00:00", "status": "completed"}
# ## Run drogued drifter simulation

# %% papermill={"duration": 366.9431, "end_time": "2026-04-02T14:46:31.224235+00:00", "exception": false, "start_time": "2026-04-02T14:40:24.281135+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.001572, "end_time": "2026-04-02T14:46:31.227453+00:00", "exception": false, "start_time": "2026-04-02T14:46:31.225881+00:00", "status": "completed"}
# ## Run surface point particle simulation

# %% papermill={"duration": 5.593388, "end_time": "2026-04-02T14:46:36.822463+00:00", "exception": false, "start_time": "2026-04-02T14:46:31.229075+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.001626, "end_time": "2026-04-02T14:46:36.826088+00:00", "exception": false, "start_time": "2026-04-02T14:46:36.824462+00:00", "status": "completed"}
# ## Run drogue-depth point particle simulation

# %% papermill={"duration": 5.86748, "end_time": "2026-04-02T14:46:42.695140+00:00", "exception": false, "start_time": "2026-04-02T14:46:36.827660+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.001643, "end_time": "2026-04-02T14:46:42.698792+00:00", "exception": false, "start_time": "2026-04-02T14:46:42.697149+00:00", "status": "completed"}
# ## Summary

# %% papermill={"duration": 0.069446, "end_time": "2026-04-02T14:46:42.769959+00:00", "exception": false, "start_time": "2026-04-02T14:46:42.700513+00:00", "status": "completed"}
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
