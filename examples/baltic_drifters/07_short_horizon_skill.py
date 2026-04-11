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
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.006521, "end_time": "2026-04-02T15:35:41.521389+00:00", "exception": false, "start_time": "2026-04-02T15:35:41.514868+00:00", "status": "completed"}
# # Short-horizon skill scores
#
# Analyse the 12 h simulation segments from notebook 06.
# Compute separation distance as a function of lead time (0–12 h),
# averaged over all release times and drifters.

# %% [markdown] papermill={"duration": 0.00396, "end_time": "2026-04-02T15:35:41.531946+00:00", "exception": false, "start_time": "2026-04-02T15:35:41.527986+00:00", "status": "completed"}
# ## Parameters

# %% papermill={"duration": 0.011982, "end_time": "2026-04-02T15:35:41.546439+00:00", "exception": false, "start_time": "2026-04-02T15:35:41.534457+00:00", "status": "completed"} tags=["parameters"]
ZARR_DD = "output/short_dd.zarr"
ZARR_SURFACE = "output/short_surface.zarr"
ZARR_3M = "output/short_3m.zarr"
RELEASES_CSV = "output/short_releases.csv"
CSV_SCIENCE = "data/drifters_science.csv"
SEGMENT_HOURS = 12

# %% [markdown] papermill={"duration": 0.002657, "end_time": "2026-04-02T15:35:41.551900+00:00", "exception": false, "start_time": "2026-04-02T15:35:41.549243+00:00", "status": "completed"}
# ## Imports

# %% papermill={"duration": 0.478228, "end_time": "2026-04-02T15:35:42.032456+00:00", "exception": false, "start_time": "2026-04-02T15:35:41.554228+00:00", "status": "completed"}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# %% [markdown] papermill={"duration": 0.000771, "end_time": "2026-04-02T15:35:42.034281+00:00", "exception": false, "start_time": "2026-04-02T15:35:42.033510+00:00", "status": "completed"}
# ## Load data

# %% papermill={"duration": 0.762854, "end_time": "2026-04-02T15:35:42.797859+00:00", "exception": false, "start_time": "2026-04-02T15:35:42.035005+00:00", "status": "completed"}
release_df = pd.read_csv(RELEASES_CSV, parse_dates=["time"])
obs_df = pd.read_csv(CSV_SCIENCE, parse_dates=["date_UTC"])

ds_dd = xr.open_zarr(ZARR_DD).load()
ds_surface = xr.open_zarr(ZARR_SURFACE).load()
ds_3m = xr.open_zarr(ZARR_3M).load()

drifter_ids = sorted(release_df["D_number"].unique())
print(f"Releases: {len(release_df)}, drifters: {drifter_ids}")
print(f"DD dims: {dict(ds_dd.sizes)}")


# %% [markdown] papermill={"duration": 0.000809, "end_time": "2026-04-02T15:35:42.799745+00:00", "exception": false, "start_time": "2026-04-02T15:35:42.798936+00:00", "status": "completed"}
# ## Helpers

# %% papermill={"duration": 0.00396, "end_time": "2026-04-02T15:35:42.804449+00:00", "exception": false, "start_time": "2026-04-02T15:35:42.800489+00:00", "status": "completed"}
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# %% [markdown] papermill={"duration": 0.000776, "end_time": "2026-04-02T15:35:42.806243+00:00", "exception": false, "start_time": "2026-04-02T15:35:42.805467+00:00", "status": "completed"}
# ## Compute separation vs lead time
#
# For each segment, interpolate observed positions to simulated output
# times and compute haversine separation. Collect as (lead_time_h, sep_km)
# for each simulation type.

# %% papermill={"duration": 1.069693, "end_time": "2026-04-02T15:35:43.876672+00:00", "exception": false, "start_time": "2026-04-02T15:35:42.806979+00:00", "status": "completed"}
def compute_separations(ds_sim, release_df, obs_df):
    """Return DataFrame with columns: D_number, release_time, lead_h, sep_km."""
    m_per_deg = 111120.0
    rows = []
    for traj_idx in range(ds_sim.sizes["trajectory"]):
        sim_lon = ds_sim.lon.isel(trajectory=traj_idx).values
        sim_lat = ds_sim.lat.isel(trajectory=traj_idx).values
        sim_time = ds_sim.time.isel(trajectory=traj_idx).values

        valid = np.isfinite(sim_lon) & np.isfinite(sim_lat)
        if valid.sum() < 2:
            continue
        sim_lon, sim_lat, sim_time = sim_lon[valid], sim_lat[valid], sim_time[valid]

        rel = release_df.iloc[traj_idx]
        did = rel["D_number"]
        t0 = np.datetime64(rel["time"])

        # Get observed positions for this drifter in the segment window
        obs = obs_df[obs_df["D_number"] == did]
        obs_time = obs["date_UTC"].values.astype("datetime64[ns]")
        mask = (obs_time >= sim_time[0]) & (obs_time <= sim_time[-1])
        if mask.sum() < 2:
            continue

        ot = obs_time[mask]
        obs_lon = obs["Longitude"].values[mask]
        obs_lat = obs["Latitude"].values[mask]

        # Interpolate sim to obs times
        sim_t_f = sim_time.astype(np.float64)
        obs_t_f = ot.astype(np.float64)
        slon_i = np.interp(obs_t_f, sim_t_f, sim_lon)
        slat_i = np.interp(obs_t_f, sim_t_f, sim_lat)

        sep = haversine_km(slon_i, slat_i, obs_lon, obs_lat)
        lead_h = (ot - t0).astype(np.float64) / 3.6e12

        for lh, sk in zip(lead_h, sep):
            rows.append({"D_number": did, "release_time": t0, "lead_h": lh, "sep_km": sk})

    return pd.DataFrame(rows)


sep_dd = compute_separations(ds_dd, release_df, obs_df)
sep_dd["sim"] = "Drogued drifter"
sep_surface = compute_separations(ds_surface, release_df, obs_df)
sep_surface["sim"] = "Surface pp"
sep_3m = compute_separations(ds_3m, release_df, obs_df)
sep_3m["sim"] = "3 m pp"

sep_all = pd.concat([sep_dd, sep_surface, sep_3m], ignore_index=True)
print(f"Total separation records: {len(sep_all)}")
print(sep_all.groupby("sim").size())

# %% [markdown] papermill={"duration": 0.000812, "end_time": "2026-04-02T15:35:43.878653+00:00", "exception": false, "start_time": "2026-04-02T15:35:43.877841+00:00", "status": "completed"}
# ## Separation vs lead time
#
# Individual segment lines (thin) and per-drifter mean (thick), one panel per drifter.

# %% papermill={"duration": 0.459998, "end_time": "2026-04-02T15:35:44.339399+00:00", "exception": false, "start_time": "2026-04-02T15:35:43.879401+00:00", "status": "completed"}
sep_all["lead_h_bin"] = sep_all["lead_h"].round()

sim_names = ["Drogued drifter", "Surface pp", "3 m pp"]
sim_colors = {"Drogued drifter": "C0", "Surface pp": "C1", "3 m pp": "C2"}

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)

for ax, did in zip(axes.flat, drifter_ids):
    sub = sep_all[sep_all["D_number"] == did]

    # Individual lines
    for sim in sim_names:
        s = sub[sub["sim"] == sim]
        for _, seg in s.groupby("release_time"):
            ax.plot(seg["lead_h"].values, seg["sep_km"].values,
                    color=sim_colors[sim], lw=1, alpha=0.5)

    # Mean lines
    for sim in sim_names:
        g = sub[sub["sim"] == sim].groupby("lead_h_bin")["sep_km"].mean()
        ax.plot(g.index, g.values, color=sim_colors[sim], lw=3, label=sim, zorder=10)

    ax.set_title(did)
    ax.set_xlim(0, SEGMENT_HOURS)
    ax.grid()

axes.flat[0].legend(loc=0)
fig.supxlabel("Lead time (h)")
fig.supylabel("Separation (km)")
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.002465, "end_time": "2026-04-02T15:35:44.344637+00:00", "exception": false, "start_time": "2026-04-02T15:35:44.342172+00:00", "status": "completed"}
# ## Trajectory maps
#
# One panel per drifter showing the full observed track (black) with
# all short simulation segments overlaid.

# %% papermill={"duration": 3.305421, "end_time": "2026-04-02T15:35:47.652395+00:00", "exception": false, "start_time": "2026-04-02T15:35:44.346974+00:00", "status": "completed"}
PAD = 0.1
lon_min = obs_df["Longitude"].min() - PAD
lon_max = obs_df["Longitude"].max() + PAD
lat_min = obs_df["Latitude"].min() - PAD
lat_max = obs_df["Latitude"].max() + PAD

osm = cimgt.OSM()
geodetic = ccrs.Geodetic()

sim_configs = [
    ("Drogued drifter", ds_dd, "tab:blue"),
    ("Surface pp", ds_surface, "tab:orange"),
    ("3 m pp", ds_3m, "tab:green"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 7), subplot_kw={"projection": osm.crs}, dpi=200)

for ax, did in zip(axes.flat, drifter_ids):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=geodetic)
    ax.add_image(osm, 10)

    # Full observed track
    obs = obs_df[obs_df["D_number"] == did]
    ax.plot(obs["Longitude"].values, obs["Latitude"].values, lw=0.7,
            color="black", transform=geodetic, label="Observed")

    # Short simulation segments for this drifter
    did_mask = release_df["D_number"] == did
    did_indices = release_df.index[did_mask].values

    for sim_label, ds_sim, color in sim_configs:
        for traj_idx in did_indices:
            if traj_idx >= ds_sim.sizes["trajectory"]:
                continue
            slon = ds_sim.lon.isel(trajectory=traj_idx).values
            slat = ds_sim.lat.isel(trajectory=traj_idx).values
            valid = np.isfinite(slon) & np.isfinite(slat)
            if valid.sum() < 2:
                continue
            ax.plot(slon[valid], slat[valid], color=color, lw=0.7,
                    transform=geodetic)

    ax.set_title(did)

# Manual legend on first panel
from matplotlib.lines import Line2D
legend_handles = [Line2D([], [], color="black", label="Observed")]
for label, _, color in sim_configs:
    legend_handles.append(Line2D([], [], color=color, label=label))
axes.flat[0].legend(handles=legend_handles, loc=0)

fig.suptitle(f"Observed tracks with {SEGMENT_HOURS} h simulation segments")
plt.tight_layout()
plt.show()

# %% [markdown] papermill={"duration": 0.005364, "end_time": "2026-04-02T15:35:47.664390+00:00", "exception": false, "start_time": "2026-04-02T15:35:47.659026+00:00", "status": "completed"}
# ## Summary statistics

# %% papermill={"duration": 0.022938, "end_time": "2026-04-02T15:35:47.692853+00:00", "exception": false, "start_time": "2026-04-02T15:35:47.669915+00:00", "status": "completed"}
# Mean separation at selected lead times
for lead in [1, 3, 6]:
    sub = sep_all[sep_all["lead_h_bin"] == lead]
    if len(sub) == 0:
        continue
    print(f"\nLead time = {lead} h:")
    summary = sub.groupby("sim")["sep_km"].agg(["mean", "std", "count"])
    print(summary.to_string(float_format="{:.2f}".format))

# Overall stats for lead <= 6h
print("\n--- Overall mean separation (lead time <= 6 h) ---")
sub6 = sep_all[sep_all["lead_h"] <= 6]
summary6 = sub6.groupby("sim")["sep_km"].agg(["mean", "std", "count"])
print(summary6.to_string(float_format="{:.2f}".format))
