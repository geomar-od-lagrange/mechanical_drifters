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

# Effective current fields for the Baltic drifter study

Building velocity fields that combine Eulerian currents with wave-driven
Stokes drift at each depth level. The CMEMS Baltic wave model provides
partitioned wave parameters (wind waves, primary and secondary swell),
from which we construct a Stokes drift profile using the deep-water
monochromatic approximation per partition. Adding this to the Eulerian
currents gives the effective current that a Lagrangian particle experiences.

This notebook produces `data/cmems/effective_currents.nc` for use in
subsequent Parcels runs.

## Parameters

```python tags=["parameters"]
CMEMS_DIR = "data/cmems"
OUTPUT_PATH = "data/cmems/effective_currents.nc"
```

## Imports

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from drogued_drifters.stokes import compute_stokes_profile
```

## Load Eulerian currents

CMEMS Baltic physics analysis (hourly, native grid). Contains `uo` and `vo`
at near-surface depth levels.

```python
ds_phy = xr.open_dataset(Path(CMEMS_DIR) / "cmems_mod_bal_phy_anfc_PT1H-i.nc").load()

ds_phy
```

## Load wave partition data

CMEMS Baltic wave analysis (hourly). We load the partitioned wave
parameters: significant wave height, mean period, and mean direction
for wind waves (WW), primary swell (SW1), and secondary swell (SW2).

```python
WAVE_VARS = [
    "VHM0_WW", "VTM01_WW", "VMDR_WW",
    "VHM0_SW1", "VTM01_SW1", "VMDR_SW1",
    "VHM0_SW2", "VTM01_SW2", "VMDR_SW2",
]

ds_wav = xr.open_dataset(Path(CMEMS_DIR) / "cmems_mod_bal_wav_anfc_PT1H-i.nc")[WAVE_VARS].load()

ds_wav
```

## Build Stokes drift profiles

Following the deep-water monochromatic approximation
(e.g., Liu et al., 2020; https://doi.org/10.1029/2020MS002172),
for each wave partition (wind waves, primary swell, secondary swell), the
Stokes drift profile is:

$$u_{\mathrm{St},i}(z) = A_i^2 \, \sigma_i \, k_i \, e^{2 k_i z} \, \hat{d}_i$$

where $A_i = H_{s,i}/2$ is the amplitude, $\sigma_i = 2\pi / T_i$ the
angular frequency, $k_i = \sigma_i^2 / g$ the deep-water wavenumber, and
$\hat{d}_i$ is the unit direction vector. The CMEMS direction convention is
meteorological ("coming from"), so we convert:
$\theta = (270° - \mathrm{dir_{from}})$ in radians.

We evaluate this at each depth level of the Eulerian grid (positive-down
depth coordinate), sum over all wave components, and later interpolate onto
the physics grid.

```python
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
```

## Interpolate Stokes onto the physics grid

The wave and physics grids differ slightly. We interpolate the Stokes
drift fields onto the physics grid coordinates so they can be added
directly to the Eulerian currents. Ocean points where `uo` is NaN
(land mask) are preserved.

```python
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
```

## Build effective current

The effective current at each depth is the sum of the Eulerian current
and the Stokes drift. We store all components so downstream notebooks
can inspect the individual contributions.

```python
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
```

## Save effective current dataset

```python
ds_eff.to_netcdf(OUTPUT_PATH)
print(f"Saved effective currents to {OUTPUT_PATH}")
```

## Diagnostic: depth profile of mean Eulerian vs mean Stokes speed

Time- and space-averaged speed at each depth level, showing the rapid
exponential decay of Stokes drift and the Stokes-to-Eulerian ratio.

```python
euler_mean = np.sqrt(ds_eff["uo"]**2 + ds_eff["vo"]**2).mean(["time", "longitude", "latitude"])
stokes_mean = np.sqrt(ds_eff["u_stokes"]**2 + ds_eff["v_stokes"]**2).mean(["time", "longitude", "latitude"])

for z in ds_eff.depth.values:
    e = float(euler_mean.sel(depth=z))
    s = float(stokes_mean.sel(depth=z))
    ratio = s / e if e > 0 else 0
    print(f"z = {z:.2f} m:  Eulerian {e:.4f} m/s,  Stokes {s:.4f} m/s,  ratio {ratio:.0%}")
```

```python
fig, ax = plt.subplots()

ax.plot(stokes_mean.values, -stokes_mean.depth.values, "o-", label="Stokes")
ax.plot(euler_mean.values, -euler_mean.depth.values, "s-", label="Eulerian")

ax.set_xlabel("Mean speed [m/s]")
ax.set_ylabel("Depth [m]")
ax.legend()
plt.show()
```

```python
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
```
