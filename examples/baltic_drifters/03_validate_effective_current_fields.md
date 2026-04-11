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

<!-- #region papermill={"duration": 0.001578, "end_time": "2026-04-11T15:46:13.126448+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.124870+00:00", "status": "completed"} -->
# Validate effective current fields along drifter tracks

Sample CMEMS effective currents (Eulerian + Stokes drift) along observed drifter
trajectories and compare model-predicted velocity against observed drifter
velocity. No forward simulation -- just field interpolation at drifter positions.

Validation is restricted to science periods extracted in notebook 00.
<!-- #endregion -->

<!-- #region papermill={"duration": 0.000833, "end_time": "2026-04-11T15:46:13.128326+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.127493+00:00", "status": "completed"} -->
## Parameters
<!-- #endregion -->

```python papermill={"duration": 0.004206, "end_time": "2026-04-11T15:46:13.133351+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.129145+00:00", "status": "completed"} tags=["parameters"]
CSV_PATH = "data/drifters_science.csv"
EFFECTIVE_CURRENTS_PATH = "data/cmems/effective_currents.nc"
WAVE_PATH = "data/cmems/cmems_mod_bal_wav_anfc_PT1H-i.nc"
RESAMPLE_INTERVAL = "1h"
```

<!-- #region papermill={"duration": 0.000884, "end_time": "2026-04-11T15:46:13.135187+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.134303+00:00", "status": "completed"} -->
## Imports
<!-- #endregion -->

```python papermill={"duration": 0.471193, "end_time": "2026-04-11T15:46:13.607218+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.136025+00:00", "status": "completed"}
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
```

<!-- #region papermill={"duration": 0.000858, "end_time": "2026-04-11T15:46:13.609222+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.608364+00:00", "status": "completed"} -->
## Load and resample science observations
<!-- #endregion -->

```python papermill={"duration": 0.058871, "end_time": "2026-04-11T15:46:13.668924+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.610053+00:00", "status": "completed"}
# CSV_PATH points to drifters_science.csv, which contains only science-period
# positions (extracted in notebook 00). No additional filtering needed.
df_raw = pd.read_csv(CSV_PATH, parse_dates=["date_UTC"])
df_raw = df_raw.rename(columns={"Latitude": "lat", "Longitude": "lon"})

# Resample each drifter to hourly, keeping nearest observation
frames = []
for name, grp in df_raw.groupby("D_number"):
    grp = grp.set_index("date_UTC").sort_index()
    grp_hourly = grp[["lat", "lon"]].resample(RESAMPLE_INTERVAL).nearest()
    grp_hourly["D_number"] = name
    frames.append(grp_hourly)

df = pd.concat(frames).reset_index().rename(columns={"date_UTC": "time"})
df = df.dropna(subset=["lat", "lon"])
print(f"Resampled observations (science periods only): {len(df)} rows, {df['D_number'].nunique()} drifters")
df.head()
```

<!-- #region papermill={"duration": 0.000943, "end_time": "2026-04-11T15:46:13.670992+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.670049+00:00", "status": "completed"} -->
## Load effective currents
<!-- #endregion -->

```python papermill={"duration": 0.179278, "end_time": "2026-04-11T15:46:13.851178+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.671900+00:00", "status": "completed"}
ds = xr.open_dataset(EFFECTIVE_CURRENTS_PATH)
print(ds)
# Available depths
print("Depths:", ds.depth.values)
```

```python papermill={"duration": 25.91983, "end_time": "2026-04-11T15:46:39.773635+00:00", "exception": false, "start_time": "2026-04-11T15:46:13.853805+00:00", "status": "completed"}
# Extrapolate onto land: rolling mean fills NaN coastal cells
# Apply along latitude, then longitude (width=3, min_periods=1)
ds_vars = ["U_eff", "V_eff", "uo", "vo", "u_stokes", "v_stokes"]
for _ in range(3):  # iterate to propagate further into land
    for var in ds_vars:
        filled = (
            ds[var]
            .rolling(latitude=3, center=True, min_periods=1).mean()
            .rolling(longitude=3, center=True, min_periods=1).mean()
        )
        ds[var] = ds[var].fillna(filled)

print(f"Extrapolated {ds_vars} onto land (3 passes, rolling width=3)")
```

<!-- #region papermill={"duration": 0.000971, "end_time": "2026-04-11T15:46:39.775900+00:00", "exception": false, "start_time": "2026-04-11T15:46:39.774929+00:00", "status": "completed"} -->
## Interpolate fields to drifter positions

Interpolate U_eff, uo, vo at surface (~0.5 m) and drogue depth (3.0 m) along each drifter track.
<!-- #endregion -->

```python papermill={"duration": 7.636873, "end_time": "2026-04-11T15:46:47.413704+00:00", "exception": false, "start_time": "2026-04-11T15:46:39.776831+00:00", "status": "completed"}
DEPTH_SURFACE = float(ds.depth.isel(depth=0))   # ~0.5 m
DEPTH_DROGUE = 3.0                           # interpolated to exactly 3 m

sampled_records = []

for name, grp in df.groupby("D_number"):
    grp = grp.sort_values("time").reset_index(drop=True)

    times = xr.DataArray(grp["time"].values, dims="points")
    lats = xr.DataArray(grp["lat"].values, dims="points")
    lons = xr.DataArray(grp["lon"].values, dims="points")

    interp_surf = ds[["U_eff", "V_eff", "uo", "vo"]].interp(
        time=times, latitude=lats, longitude=lons, depth=DEPTH_SURFACE,
        method="linear",
    )
    interp_drog = ds[["U_eff", "V_eff", "uo", "vo"]].interp(
        time=times, latitude=lats, longitude=lons, depth=DEPTH_DROGUE,
        method="linear",
    )

    rec = grp.copy()
    rec["U_eff_surf"] = interp_surf["U_eff"].values
    rec["V_eff_surf"] = interp_surf["V_eff"].values
    rec["uo_surf"] = interp_surf["uo"].values
    rec["vo_surf"] = interp_surf["vo"].values
    rec["U_eff_drog"] = interp_drog["U_eff"].values
    rec["V_eff_drog"] = interp_drog["V_eff"].values
    rec["uo_drog"] = interp_drog["uo"].values
    rec["vo_drog"] = interp_drog["vo"].values
    sampled_records.append(rec)

df_sampled = pd.concat(sampled_records).reset_index(drop=True)
print(f"Sampled dataset: {len(df_sampled)} rows")
print(f"Surface depth: {DEPTH_SURFACE:.2f} m, drogue depth: {DEPTH_DROGUE:.1f} m")
df_sampled.head()
```

<!-- #region papermill={"duration": 0.001061, "end_time": "2026-04-11T15:46:47.416113+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.415052+00:00", "status": "completed"} -->
## Compute observed drift speed

Estimate observed speed from consecutive hourly positions using the haversine formula.
<!-- #endregion -->

```python papermill={"duration": 0.014505, "end_time": "2026-04-11T15:46:47.431659+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.417154+00:00", "status": "completed"}
def haversine_speed(lats, lons, times):
    """Compute speed in m/s from arrays of lat, lon (degrees) and times (datetime64)."""
    R = 6371e3
    lat = np.deg2rad(lats.values)
    lon = np.deg2rad(lons.values)
    dt_s = np.diff(times.values.astype("datetime64[s]").astype(float))

    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(a))

    speed = np.where(dt_s > 0, dist / dt_s, np.nan)
    return np.concatenate([[np.nan], speed])


m_per_deg = 111120.0

obs_frames = []
for name, grp in df_sampled.groupby("D_number"):
    grp = grp.sort_values("time").reset_index(drop=True)
    grp["obs_speed"] = haversine_speed(grp["lat"], grp["lon"], grp["time"])

    # Observed U (eastward) and V (northward) from finite differences
    dt_s = grp["time"].diff().dt.total_seconds()
    grp["obs_u"] = grp["lon"].diff() * m_per_deg * np.cos(np.deg2rad(grp["lat"])) / dt_s
    grp["obs_v"] = grp["lat"].diff() * m_per_deg / dt_s
    obs_frames.append(grp)

df_sampled = pd.concat(obs_frames).reset_index(drop=True)

# Model speed magnitudes
df_sampled["eff_speed_surf"] = np.sqrt(df_sampled["U_eff_surf"]**2 + df_sampled["V_eff_surf"]**2)
df_sampled["eff_speed_drog"] = np.sqrt(df_sampled["U_eff_drog"]**2 + df_sampled["V_eff_drog"]**2)
df_sampled["euler_speed_surf"] = np.sqrt(df_sampled["uo_surf"]**2 + df_sampled["vo_surf"]**2)
df_sampled["euler_speed_drog"] = np.sqrt(df_sampled["uo_drog"]**2 + df_sampled["vo_drog"]**2)

print(df_sampled[["obs_speed", "eff_speed_surf", "eff_speed_drog", "euler_speed_surf", "euler_speed_drog"]].describe())
```

<!-- #region papermill={"duration": 0.001056, "end_time": "2026-04-11T15:46:47.433920+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.432864+00:00", "status": "completed"} -->
## Along-track speed comparison

Time series of observed drift speed versus model effective current speed (surface and drogue depth) per drifter.
<!-- #endregion -->

```python papermill={"duration": 0.377347, "end_time": "2026-04-11T15:46:47.812271+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.434924+00:00", "status": "completed"}
drifter_ids = sorted(df_sampled["D_number"].unique())
n = len(drifter_ids)

fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
if n == 1:
    axes = [axes]

for ax, did in zip(axes, drifter_ids):
    sub = df_sampled[df_sampled["D_number"] == did].sort_values("time")
    ax.plot(sub["time"], sub["obs_speed"], color="black", label="Observed")
    ax.plot(sub["time"], sub["euler_speed_surf"], color="C1", label=f"Euler surface ({DEPTH_SURFACE:.1f} m)")
    ax.plot(sub["time"], sub["eff_speed_surf"], color="C1", ls="--", label=f"Eff surface ({DEPTH_SURFACE:.1f} m)")
    ax.plot(sub["time"], sub["euler_speed_drog"], color="C0", label=f"Euler {DEPTH_DROGUE:.0f} m")
    ax.plot(sub["time"], sub["eff_speed_drog"], color="C0", ls="--", label=f"Eff {DEPTH_DROGUE:.0f} m")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(did)
    ax.grid()
    ax.legend(loc=0)

fig.suptitle("Along-track speed: observed vs model fields", y=1.01)
plt.tight_layout()
plt.show()
```

<!-- #region papermill={"duration": 0.002625, "end_time": "2026-04-11T15:46:47.818072+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.815447+00:00", "status": "completed"} -->
## Along-track U/V component comparison

Eastward (U) and northward (V) velocity components per drifter: observed from finite differences vs model fields.
<!-- #endregion -->

```python papermill={"duration": 0.567333, "end_time": "2026-04-11T15:46:48.387759+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.820426+00:00", "status": "completed"}
fig, axes = plt.subplots(n, 2, figsize=(14, 2.5 * n), sharex=False)

for i, did in enumerate(drifter_ids):
    sub = df_sampled[df_sampled["D_number"] == did].sort_values("time")
    ax_u, ax_v = axes[i, 0], axes[i, 1]

    for ax, obs_col, model_pairs, component in [
        (ax_u, "obs_u",
         [("uo_surf", "U_eff_surf", f"surface ({DEPTH_SURFACE:.1f} m)", "C1"),
          ("uo_drog", "U_eff_drog", f"{DEPTH_DROGUE:.0f} m", "C0")],
         "U (m/s)"),
        (ax_v, "obs_v",
         [("vo_surf", "V_eff_surf", f"surface ({DEPTH_SURFACE:.1f} m)", "C1"),
          ("vo_drog", "V_eff_drog", f"{DEPTH_DROGUE:.0f} m", "C0")],
         "V (m/s)"),
    ]:
        ax.plot(sub["time"], sub[obs_col], color="black", label="Observed")
        for euler_col, eff_col, label, color in model_pairs:
            ax.plot(sub["time"], sub[euler_col], color=color, label=f"Euler {label}")
            ax.plot(sub["time"], sub[eff_col], color=color, ls="--", label=f"Eff {label}")
        ax.grid()
        ax.set_ylabel(component)
        if i == 0:
            ax.legend(loc=0)

    ax_u.set_title(f"{did} — U (eastward)")
    ax_v.set_title(f"{did} — V (northward)")

plt.tight_layout()
plt.show()
```
