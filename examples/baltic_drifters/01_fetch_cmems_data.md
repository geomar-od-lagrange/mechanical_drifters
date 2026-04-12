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

# Fetch CMEMS Data

Download and cache all CMEMS datasets needed by the Baltic drifter notebooks.
The time range is derived programmatically from the science CSV produced by notebook 00.

## Imports

```python
from pathlib import Path

import copernicusmarine
import pandas as pd
```

## Parameters

```python tags=["parameters"]
SCIENCE_CSV = "data/drifters_science.csv"
OUTPUT_DIR = "data/cmems"
LON_MIN = 9.0
LON_MAX = 13.0
LAT_MIN = 53.5
LAT_MAX = 56.0
```

## Derive time range from science data

```python
df = pd.read_csv(SCIENCE_CSV, parse_dates=["date_UTC"])
time_start = df["date_UTC"].min().floor("D").strftime("%Y-%m-%d")
time_end = df["date_UTC"].max().ceil("D").strftime("%Y-%m-%d")
print(f"Derived time range: {time_start} to {time_end}")
```

## Setup

```python
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
```

## Physics (currents)

```python
ds_phy = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_bal_phy_anfc_PT1H-i",
).sel(
    longitude=slice(LON_MIN, LON_MAX),
    latitude=slice(LAT_MIN, LAT_MAX),
    time=slice(time_start, time_end),
    depth=slice(0, 5),
)[["uo", "vo"]].load()

out_path = output_dir / "cmems_mod_bal_phy_anfc_PT1H-i.nc"
ds_phy.to_netcdf(out_path)
print(f"Physics: {dict(ds_phy.dims)}")
print(f"  {out_path.name}: {out_path.stat().st_size / 1e6:.1f} MB")
```

## Waves

```python
WAVE_VARS = [
    "VSDX", "VSDY",                        # surface Stokes drift components
    "VHM0", "VTPK",                        # total Hs and peak period
    "VHM0_WW", "VTM01_WW", "VMDR_WW",     # wind wave partition
    "VHM0_SW1", "VTM01_SW1", "VMDR_SW1",  # swell partition 1
    "VHM0_SW2", "VTM01_SW2", "VMDR_SW2",  # swell partition 2
]

ds_wav = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_bal_wav_anfc_PT1H-i",
).sel(
    longitude=slice(LON_MIN, LON_MAX),
    latitude=slice(LAT_MIN, LAT_MAX),
    time=slice(time_start, time_end),
)[WAVE_VARS].load()

out_path = output_dir / "cmems_mod_bal_wav_anfc_PT1H-i.nc"
ds_wav.to_netcdf(out_path)
print(f"Waves: {dict(ds_wav.dims)}")
print(f"  {out_path.name}: {out_path.stat().st_size / 1e6:.1f} MB")
```

## Static (land mask)

```python
ds_static = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_bal_phy_anfc_static",
    service="static-arco",
)[["mask"]].load()

out_path = output_dir / "cmems_mod_bal_phy_anfc_static.nc"
ds_static.to_netcdf(out_path)
print(f"Static: {dict(ds_static.dims)}")
print(f"  {out_path.name}: {out_path.stat().st_size / 1e6:.1f} MB")
```

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Mean current speed at shallowest depth
current_speed = np.sqrt(ds_phy["uo"] ** 2 + ds_phy["vo"] ** 2).isel(depth=0).mean("time")

# Mean Stokes drift speed
stokes_speed = np.sqrt(ds_wav["VSDX"] ** 2 + ds_wav["VSDY"] ** 2).mean("time")

# Land mask at shallowest depth (first time step is representative)
land_mask = ds_static["mask"].sel(
    longitude=slice(LON_MIN, LON_MAX),
    latitude=slice(LAT_MIN, LAT_MAX),
).isel(depth=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

current_speed.plot(ax=axes[0])
axes[0].set_title("Mean current speed (shallowest depth)")

stokes_speed.plot(ax=axes[1])
axes[1].set_title("Mean Stokes drift speed")

land_mask.plot(ax=axes[2])
axes[2].set_title("Land mask (shallowest depth)")

plt.tight_layout()
plt.show()
```

## Summary

```python
print("Saved files:")
for f in sorted(output_dir.glob("*.nc")):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name}: {size_mb:.1f} MB")
```
