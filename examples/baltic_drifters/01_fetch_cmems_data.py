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

# %% [markdown] papermill={"duration": 0.009407, "end_time": "2026-04-02T14:37:40.168720+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.159313+00:00", "status": "completed"}
# # Fetch CMEMS Data
#
# Download and cache all CMEMS datasets needed by the Baltic drifter notebooks.
# The time range is derived programmatically from the science CSV produced by notebook 00.

# %% [markdown] papermill={"duration": 0.002764, "end_time": "2026-04-02T14:37:40.176236+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.173472+00:00", "status": "completed"}
# ## Imports

# %% papermill={"duration": 0.602896, "end_time": "2026-04-02T14:37:40.781995+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.179099+00:00", "status": "completed"}
from pathlib import Path

import copernicusmarine
import pandas as pd

# %% [markdown] papermill={"duration": 0.000773, "end_time": "2026-04-02T14:37:40.783782+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.783009+00:00", "status": "completed"}
# ## Parameters

# %% papermill={"duration": 0.00356, "end_time": "2026-04-02T14:37:40.788070+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.784510+00:00", "status": "completed"} tags=["parameters"]
SCIENCE_CSV = "data/drifters_science.csv"
OUTPUT_DIR = "data/cmems"
LON_MIN = 9.0
LON_MAX = 13.0
LAT_MIN = 53.5
LAT_MAX = 56.0

# %% [markdown] papermill={"duration": 0.000752, "end_time": "2026-04-02T14:37:40.789681+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.788929+00:00", "status": "completed"}
# ## Derive time range from science data

# %% papermill={"duration": 0.033659, "end_time": "2026-04-02T14:37:40.824069+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.790410+00:00", "status": "completed"}
df = pd.read_csv(SCIENCE_CSV, parse_dates=["date_UTC"])
time_start = df["date_UTC"].min().floor("D").strftime("%Y-%m-%d")
time_end = df["date_UTC"].max().ceil("D").strftime("%Y-%m-%d")
print(f"Derived time range: {time_start} to {time_end}")

# %% [markdown] papermill={"duration": 0.000747, "end_time": "2026-04-02T14:37:40.825700+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.824953+00:00", "status": "completed"}
# ## Setup

# %% papermill={"duration": 0.003235, "end_time": "2026-04-02T14:37:40.829656+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.826421+00:00", "status": "completed"}
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.00081, "end_time": "2026-04-02T14:37:40.831313+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.830503+00:00", "status": "completed"}
# ## Physics (currents)

# %% papermill={"duration": 65.916545, "end_time": "2026-04-02T14:38:46.748586+00:00", "exception": false, "start_time": "2026-04-02T14:37:40.832041+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.000845, "end_time": "2026-04-02T14:38:46.750651+00:00", "exception": false, "start_time": "2026-04-02T14:38:46.749806+00:00", "status": "completed"}
# ## Waves

# %% papermill={"duration": 57.4593, "end_time": "2026-04-02T14:39:44.210803+00:00", "exception": false, "start_time": "2026-04-02T14:38:46.751503+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.000844, "end_time": "2026-04-02T14:39:44.212645+00:00", "exception": false, "start_time": "2026-04-02T14:39:44.211801+00:00", "status": "completed"}
# ## Static (land mask)

# %% papermill={"duration": 5.112923, "end_time": "2026-04-02T14:39:49.326419+00:00", "exception": false, "start_time": "2026-04-02T14:39:44.213496+00:00", "status": "completed"}
ds_static = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_bal_phy_anfc_static",
    service="static-arco",
)[["mask"]].load()

out_path = output_dir / "cmems_mod_bal_phy_anfc_static.nc"
ds_static.to_netcdf(out_path)
print(f"Static: {dict(ds_static.dims)}")
print(f"  {out_path.name}: {out_path.stat().st_size / 1e6:.1f} MB")

# %% papermill={"duration": 0.500798, "end_time": "2026-04-02T14:39:49.828442+00:00", "exception": false, "start_time": "2026-04-02T14:39:49.327644+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.001216, "end_time": "2026-04-02T14:39:49.831328+00:00", "exception": false, "start_time": "2026-04-02T14:39:49.830112+00:00", "status": "completed"}
# ## Summary

# %% papermill={"duration": 0.004652, "end_time": "2026-04-02T14:39:49.837152+00:00", "exception": false, "start_time": "2026-04-02T14:39:49.832500+00:00", "status": "completed"}
print("Saved files:")
for f in sorted(output_dir.glob("*.nc")):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name}: {size_mb:.1f} MB")
