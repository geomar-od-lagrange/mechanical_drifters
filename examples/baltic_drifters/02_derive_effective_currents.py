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

# %% [markdown] papermill={"duration": 0.010166, "end_time": "2026-04-02T16:27:20.006675+00:00", "exception": false, "start_time": "2026-04-02T16:27:19.996509+00:00", "status": "completed"}
# # Effective current fields for the Baltic drifter study
#
# Building velocity fields that combine Eulerian currents with wave-driven
# Stokes drift at each depth level. The CMEMS Baltic wave model provides
# partitioned wave parameters (wind waves, primary and secondary swell),
# from which we construct a Stokes drift profile using the deep-water
# monochromatic approximation per partition. Adding this to the Eulerian
# currents gives the effective current that a Lagrangian particle experiences.
#
# This notebook produces `data/cmems/effective_currents.nc` for use in
# subsequent Parcels runs.

# %% [markdown] papermill={"duration": 0.003008, "end_time": "2026-04-02T16:27:20.015472+00:00", "exception": false, "start_time": "2026-04-02T16:27:20.012464+00:00", "status": "completed"}
# ## Parameters

# %% papermill={"duration": 0.010547, "end_time": "2026-04-02T16:27:20.028831+00:00", "exception": false, "start_time": "2026-04-02T16:27:20.018284+00:00", "status": "completed"} tags=["parameters"]
CMEMS_DIR = "data/cmems"
OUTPUT_PATH = "data/cmems/effective_currents.nc"

# %% [markdown] papermill={"duration": 0.002494, "end_time": "2026-04-02T16:27:20.034283+00:00", "exception": false, "start_time": "2026-04-02T16:27:20.031789+00:00", "status": "completed"}
# ## Imports

# %% papermill={"duration": 0.817031, "end_time": "2026-04-02T16:27:20.853773+00:00", "exception": false, "start_time": "2026-04-02T16:27:20.036742+00:00", "status": "completed"}
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from drogued_drifters.stokes import compute_stokes_profile

# %% [markdown] papermill={"duration": 0.000891, "end_time": "2026-04-02T16:27:20.855847+00:00", "exception": false, "start_time": "2026-04-02T16:27:20.854956+00:00", "status": "completed"}
# ## Load Eulerian currents
#
# CMEMS Baltic physics analysis (hourly, native grid). Contains `uo` and `vo`
# at near-surface depth levels.

# %% papermill={"duration": 0.188903, "end_time": "2026-04-02T16:27:21.045592+00:00", "exception": false, "start_time": "2026-04-02T16:27:20.856689+00:00", "status": "completed"}
ds_phy = xr.open_dataset(Path(CMEMS_DIR) / "cmems_mod_bal_phy_anfc_PT1H-i.nc").load()

ds_phy

# %% [markdown] papermill={"duration": 0.001108, "end_time": "2026-04-02T16:27:21.048081+00:00", "exception": false, "start_time": "2026-04-02T16:27:21.046973+00:00", "status": "completed"}
# ## Load wave partition data
#
# CMEMS Baltic wave analysis (hourly). We load the partitioned wave
# parameters: significant wave height, mean period, and mean direction
# for wind waves (WW), primary swell (SW1), and secondary swell (SW2).

# %% papermill={"duration": 0.12238, "end_time": "2026-04-02T16:27:21.171572+00:00", "exception": false, "start_time": "2026-04-02T16:27:21.049192+00:00", "status": "completed"}
WAVE_VARS = [
    "VHM0_WW", "VTM01_WW", "VMDR_WW",
    "VHM0_SW1", "VTM01_SW1", "VMDR_SW1",
    "VHM0_SW2", "VTM01_SW2", "VMDR_SW2",
]

ds_wav = xr.open_dataset(Path(CMEMS_DIR) / "cmems_mod_bal_wav_anfc_PT1H-i.nc")[WAVE_VARS].load()

ds_wav

# %% [markdown] papermill={"duration": 0.001294, "end_time": "2026-04-02T16:27:21.174552+00:00", "exception": false, "start_time": "2026-04-02T16:27:21.173258+00:00", "status": "completed"}
# ## Build Stokes drift profiles
#
# Following the deep-water monochromatic approximation
# (e.g., Liu et al., 2020; https://doi.org/10.1029/2020MS002172),
# for each wave partition (wind waves, primary swell, secondary swell), the
# Stokes drift profile is:
#
# $$u_{\mathrm{St},i}(z) = A_i^2 \, \sigma_i \, k_i \, e^{2 k_i z} \, \hat{d}_i$$
#
# where $A_i = H_{s,i}/2$ is the amplitude, $\sigma_i = 2\pi / T_i$ the
# angular frequency, $k_i = \sigma_i^2 / g$ the deep-water wavenumber, and
# $\hat{d}_i$ is the unit direction vector. The CMEMS direction convention is
# meteorological ("coming from"), so we convert:
# $\theta = (270° - \mathrm{dir_{from}})$ in radians.
#
# We evaluate this at each depth level of the Eulerian grid (positive-down
# depth coordinate), sum over all wave components, and later interpolate onto
# the physics grid.

# %% papermill={"duration": 0.893707, "end_time": "2026-04-02T16:27:22.069539+00:00", "exception": false, "start_time": "2026-04-02T16:27:21.175832+00:00", "status": "completed"}
g = 9.81
depth_levels = ds_phy.depth.values  # positive-down depth coordinate
depth_levels_zup = -depth_levels[::-1]  # z-up (negative, ascending) for compute_stokes_profile

# Partitions: (Hs variable, period variable, direction variable)
PARTITIONS = [
    ("VHM0_WW", "VTM01_WW", "VMDR_WW"),
    ("VHM0_SW1", "VTM01_SW1", "VMDR_SW1"),
    ("VHM0_SW2", "VTM01_SW2", "VMDR_SW2"),
]

# Accumulate Stokes drift on the wave grid, with a depth dimension
u_stokes_total = np.zeros((len(depth_levels_zup), len(ds_wav.time), len(ds_wav.latitude), len(ds_wav.longitude)))
v_stokes_total = np.zeros_like(u_stokes_total)

for hs_var, t_var, dir_var in PARTITIONS:
    hs = ds_wav[hs_var]
    T = ds_wav[t_var].where(ds_wav[t_var] > 0)  # guard T==0 -> NaN before division
    dir_from = ds_wav[dir_var]

    # Filter negligible wave components
    valid = hs > 0.01

    A = (hs / 2).where(valid, 0.0).fillna(0.0)
    sigma = (2 * np.pi / T).fillna(0.0)
    k = sigma**2 / g
    theta = np.deg2rad(270.0 - dir_from).fillna(0.0)

    # Surface Stokes drift components
    stokes_surf = A**2 * sigma * k
    surface_u = (stokes_surf * np.cos(theta)).values
    surface_v = (stokes_surf * np.sin(theta)).values

    du, dv = compute_stokes_profile(surface_u, surface_v, T.fillna(1.0).values, depth_levels_zup, g=g)
    u_stokes_total += du
    v_stokes_total += dv

# Flip depth axis back to z-down order to match ds_phy.depth
u_stokes_total = u_stokes_total[::-1]
v_stokes_total = v_stokes_total[::-1]

u_stokes = xr.DataArray(
    u_stokes_total,
    dims=["depth", "time", "latitude", "longitude"],
    coords={
        "time": ds_wav.time,
        "depth": depth_levels,
        "latitude": ds_wav.latitude,
        "longitude": ds_wav.longitude,
    },
).transpose("time", "depth", "latitude", "longitude")

v_stokes = xr.DataArray(
    v_stokes_total,
    dims=["depth", "time", "latitude", "longitude"],
    coords={
        "time": ds_wav.time,
        "depth": depth_levels,
        "latitude": ds_wav.latitude,
        "longitude": ds_wav.longitude,
    },
).transpose("time", "depth", "latitude", "longitude")

print(f"Stokes drift computed at {len(depth_levels)} depth levels: {depth_levels}")

# %% [markdown] papermill={"duration": 0.001318, "end_time": "2026-04-02T16:27:22.072590+00:00", "exception": false, "start_time": "2026-04-02T16:27:22.071272+00:00", "status": "completed"}
# ## Interpolate Stokes onto the physics grid
#
# The wave and physics grids differ slightly. We interpolate the Stokes
# drift fields onto the physics grid coordinates so they can be added
# directly to the Eulerian currents. Ocean points where `uo` is NaN
# (land mask) are preserved.

# %% papermill={"duration": 1.637299, "end_time": "2026-04-02T16:27:23.711145+00:00", "exception": false, "start_time": "2026-04-02T16:27:22.073846+00:00", "status": "completed"}
ds_stokes = xr.Dataset({"u_stokes": u_stokes, "v_stokes": v_stokes})

# Interpolate onto physics grid
ds_stokes_phys = ds_stokes.interp(
    longitude=ds_phy.longitude,
    latitude=ds_phy.latitude,
    method="linear",
).fillna(0.0)

# Apply land mask from Eulerian data
land_mask = ds_phy["uo"].isnull()
ds_stokes_phys["u_stokes"] = ds_stokes_phys["u_stokes"].where(~land_mask)
ds_stokes_phys["v_stokes"] = ds_stokes_phys["v_stokes"].where(~land_mask)

ds_stokes_phys

# %% [markdown] papermill={"duration": 0.001446, "end_time": "2026-04-02T16:27:23.714431+00:00", "exception": false, "start_time": "2026-04-02T16:27:23.712985+00:00", "status": "completed"}
# ## Build effective current
#
# The effective current at each depth is the sum of the Eulerian current
# and the Stokes drift. We store all components so downstream notebooks
# can inspect the individual contributions.

# %% papermill={"duration": 0.083613, "end_time": "2026-04-02T16:27:23.799425+00:00", "exception": false, "start_time": "2026-04-02T16:27:23.715812+00:00", "status": "completed"}
ds_eff = xr.Dataset(
    {
        "U_eff": ds_phy["uo"] + ds_stokes_phys["u_stokes"],
        "V_eff": ds_phy["vo"] + ds_stokes_phys["v_stokes"],
        "uo": ds_phy["uo"],
        "vo": ds_phy["vo"],
        "u_stokes": ds_stokes_phys["u_stokes"],
        "v_stokes": ds_stokes_phys["v_stokes"],
    },
    attrs={"description": "Effective current: Eulerian + Stokes drift profile"},
)

ds_eff

# %% [markdown] papermill={"duration": 0.001918, "end_time": "2026-04-02T16:27:23.803818+00:00", "exception": false, "start_time": "2026-04-02T16:27:23.801900+00:00", "status": "completed"}
# ## Save effective current dataset

# %% papermill={"duration": 0.383334, "end_time": "2026-04-02T16:27:24.188897+00:00", "exception": false, "start_time": "2026-04-02T16:27:23.805563+00:00", "status": "completed"}
ds_eff.to_netcdf(OUTPUT_PATH)
print(f"Saved effective currents to {OUTPUT_PATH}")

# %% [markdown] papermill={"duration": 0.001678, "end_time": "2026-04-02T16:27:24.192557+00:00", "exception": false, "start_time": "2026-04-02T16:27:24.190879+00:00", "status": "completed"}
# ## Diagnostic: depth profile of mean Eulerian vs mean Stokes speed
#
# Time- and space-averaged speed at each depth level, showing the rapid
# exponential decay of Stokes drift and the Stokes-to-Eulerian ratio.

# %% papermill={"duration": 0.24565, "end_time": "2026-04-02T16:27:24.439937+00:00", "exception": false, "start_time": "2026-04-02T16:27:24.194287+00:00", "status": "completed"}
euler_mean = np.sqrt(ds_eff["uo"]**2 + ds_eff["vo"]**2).mean(["time", "longitude", "latitude"])
stokes_mean = np.sqrt(ds_eff["u_stokes"]**2 + ds_eff["v_stokes"]**2).mean(["time", "longitude", "latitude"])

for z in ds_eff.depth.values:
    e = float(euler_mean.sel(depth=z))
    s = float(stokes_mean.sel(depth=z))
    ratio = s / e if e > 0 else 0
    print(f"z = {z:.2f} m:  Eulerian {e:.4f} m/s,  Stokes {s:.4f} m/s,  ratio {ratio:.0%}")

# %% papermill={"duration": 0.047177, "end_time": "2026-04-02T16:27:24.489116+00:00", "exception": false, "start_time": "2026-04-02T16:27:24.441939+00:00", "status": "completed"}
fig, ax = plt.subplots()

ax.plot(stokes_mean.values, -stokes_mean.depth.values, "o-", label="Stokes")
ax.plot(euler_mean.values, -euler_mean.depth.values, "s-", label="Eulerian")

ax.set_xlabel("Mean speed [m/s]")
ax.set_ylabel("Depth [m]")
ax.legend()
plt.show()

# %% papermill={"duration": 0.64931, "end_time": "2026-04-02T16:27:25.140421+00:00", "exception": false, "start_time": "2026-04-02T16:27:24.491111+00:00", "status": "completed"}
# Mean effective current speed and Stokes drift speed at ~0 m and ~3 m
current_speed_0 = np.sqrt(ds_eff["U_eff"] ** 2 + ds_eff["V_eff"] ** 2).sel(depth=0, method="nearest").mean("time")
current_speed_3 = np.sqrt(ds_eff["U_eff"] ** 2 + ds_eff["V_eff"] ** 2).sel(depth=3, method="nearest").mean("time")
stokes_speed_0 = np.sqrt(ds_eff["u_stokes"] ** 2 + ds_eff["v_stokes"] ** 2).sel(depth=0, method="nearest").mean("time")
stokes_speed_3 = np.sqrt(ds_eff["u_stokes"] ** 2 + ds_eff["v_stokes"] ** 2).sel(depth=3, method="nearest").mean("time")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

current_speed_0.plot(ax=axes[0, 0])
axes[0, 0].set_title(f"Mean current speed ~{float(current_speed_0.depth):.1f} m")

current_speed_3.plot(ax=axes[0, 1])
axes[0, 1].set_title(f"Mean current speed ~{float(current_speed_3.depth):.1f} m")

stokes_speed_0.plot(ax=axes[1, 0])
axes[1, 0].set_title(f"Mean Stokes speed ~{float(stokes_speed_0.depth):.1f} m")

stokes_speed_3.plot(ax=axes[1, 1])
axes[1, 1].set_title(f"Mean Stokes speed ~{float(stokes_speed_3.depth):.1f} m")

plt.tight_layout()
plt.show()
