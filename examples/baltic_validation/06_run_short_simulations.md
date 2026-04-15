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

# Short-horizon simulations: 12 h segments every 12 h

Like notebook 04, but re-initialized from observed positions every 12 hours.
Each segment runs for 12 hours. This gives many independent forecast segments
for computing skill scores as a function of lead time.

## Parameters

```python tags=["parameters"]
CSV_SCIENCE = "data/drifters_science.csv"
EFFECTIVE_CURRENTS_PATH = "data/cmems/effective_currents.nc"
OUTPUT_DIR = "output"
DROGUE_DEPTH = 3.0
DT = 300.0
SEGMENT_HOURS = 12
RESTART_HOURS = 12
OUTPUTDT = 3600.0
backend = "numba"
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

## Build release schedule

From the science-period observations, extract positions at every
`RESTART_HOURS` interval for each drifter. Each position becomes a
release point for a `SEGMENT_HOURS` simulation.

```python
df = pd.read_csv(CSV_SCIENCE, parse_dates=["date_UTC"])
drifter_ids = sorted(df["D_number"].unique())

# Resample to hourly for clean release times
releases = []
for did in drifter_ids:
    g = df[df["D_number"] == did].set_index("date_UTC").sort_index()
    g_h = g[["Latitude", "Longitude"]].resample("1h").nearest().dropna()
    # Pick every RESTART_HOURS
    g_r = g_h.iloc[::RESTART_HOURS]
    for t, row in g_r.iterrows():
        releases.append({
            "D_number": did, "time": t,
            "lat": row["Latitude"], "lon": row["Longitude"],
        })

release_df = pd.DataFrame(releases)

# Sort by (time, D_number) so row order matches zarr trajectory order:
# run_segments loops over sorted unique times; within each batch,
# drifters appear in the order they have in release_df, which after
# this sort is alphabetical D_number order.
release_df = release_df.sort_values(["time", "D_number"]).reset_index(drop=True)

print(f"{len(release_df)} release points across {len(drifter_ids)} drifters")
print(release_df.groupby("D_number").size())
```

## Build FieldSets

```python
ds_eff_raw = xr.open_dataset(EFFECTIVE_CURRENTS_PATH)[["U_eff", "V_eff"]].load()

# Prepend z=0 surface layer
ds_z0 = ds_eff_raw.isel(depth=0).assign_coords(depth=0.0)
ds_eff = xr.concat([ds_z0, ds_eff_raw], dim="depth").rename(
    {"U_eff": "U", "V_eff": "V"}
).fillna(0.0)

# sgrid topology
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

dd = DroguedDrifter(backend=backend)
dd_kernel = make_kernel(dd)

SURFACE_DEPTH = 0.0
DROGUE_DEPTH_LEVEL = float(ds_eff.depth.sel(depth=DROGUE_DEPTH, method="nearest"))
RUNTIME = SEGMENT_HOURS * 3600

print(f"Drogue depth level: {DROGUE_DEPTH_LEVEL:.2f} m")
print(f"Segment runtime: {SEGMENT_HOURS} h")
print(f"FieldSet built.")
```

## Helper kernel

```python
def DeleteOOB(particles, fieldset):
    state = np.asarray(particles.state)
    oob = (state == StatusCode.ErrorOutOfBounds) | (state == StatusCode.ErrorThroughSurface)
    if np.any(oob):
        particles.state = np.where(oob, StatusCode.Delete, state)


output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
```

## Run all segments

Loop over unique release times. At each release time, launch one
particle per drifter (up to 6) and run for `SEGMENT_HOURS`. Collect
all trajectories into a single xarray Dataset per simulation type.

```python
def run_segments(fieldset, release_df, depth, label, kernel):
    """Run segments grouped by release time, collect into one zarr."""
    all_results = []
    release_times = sorted(release_df["time"].unique())

    for t0 in release_times:
        batch = release_df[release_df["time"] == t0]
        pset = ParticleSet(
            fieldset=fieldset,
            pclass=Particle,
            lon=batch["lon"].values,
            lat=batch["lat"].values,
            z=[depth] * len(batch),
            time=[np.datetime64(t0)] * len(batch),
        )
        tmp_store = str(output_dir / f"_tmp_{label}.zarr")
        shutil.rmtree(tmp_store, ignore_errors=True)
        pset.execute(
            kernels=[kernel, DeleteOOB],
            dt=DT,
            runtime=RUNTIME,
            output_file=ParticleFile(store=tmp_store, outputdt=OUTPUTDT),
            verbose_progress=False,
        )
        ds_batch = xr.open_zarr(tmp_store).load()
        # Tag each trajectory with its release-table index
        all_results.append(ds_batch)
        shutil.rmtree(tmp_store, ignore_errors=True)

    # Concatenate along trajectory dimension
    ds_all = xr.concat(all_results, dim="trajectory")

    store = str(output_dir / f"short_{label}.zarr")
    shutil.rmtree(store, ignore_errors=True)
    ds_all.to_zarr(store)
    print(f"  {label}: {dict(ds_all.sizes)}, saved to {store}")
    return ds_all


print(f"Running {len(release_df)} segments x 3 sim types ({len(release_df['time'].unique())} unique release times)...")
ds_short_dd = run_segments(fieldset, release_df, SURFACE_DEPTH, "dd", dd_kernel)
ds_short_surface = run_segments(fieldset, release_df, SURFACE_DEPTH, "surface", AdvectionEE)
ds_short_3m = run_segments(fieldset, release_df, DROGUE_DEPTH_LEVEL, "3m", AdvectionEE)
print("Done.")
```

## Save release metadata

Store the release schedule so notebook 07 can match trajectories
back to drifter IDs and release times.

```python
release_df.to_csv(output_dir / "short_releases.csv", index=False)
print(f"Saved release metadata: {output_dir / 'short_releases.csv'}")
```
