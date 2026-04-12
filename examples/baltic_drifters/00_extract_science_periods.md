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
    display_name: default
    language: python
    name: python3
---

# Extract science periods from raw drifter GPS data

Filter raw Baltic drifter trajectories to isolate the science (free-drifting) periods by removing pre-deployment, beaching, and post-beaching phases.

```python tags=["parameters"]
raw_csv = "data/drifters_raw.csv"
out_csv = "data/drifters_science.csv"

# Baltic bounding box
lat_min, lat_max = 53.5, 56.0
lon_min, lon_max = 9.0, 13.0
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from pathlib import Path

```

## Load and basic filter

```python
df = pd.read_csv(raw_csv, parse_dates=["date_UTC"], dayfirst=True)
df["D_number"] = "D" + df["D_number"].astype(str)
df = df.sort_values(["D_number", "date_UTC"]).reset_index(drop=True)

# Filter to Baltic bounding box
df = df[
    (df["Latitude"] >= lat_min) & (df["Latitude"] <= lat_max)
    & (df["Longitude"] >= lon_min) & (df["Longitude"] <= lon_max)
].copy()

# Deduplicate
df = df.drop_duplicates(subset=["D_number", "date_UTC"]).reset_index(drop=True)

# Drop D290: single record, not a real deployment
df = df[df["D_number"] != "D290"].reset_index(drop=True)

print(f"Drifters: {sorted(df['D_number'].unique())}")
print(f"Total records: {len(df)}")
print(df.groupby("D_number").size())
```

## Resample to 1-minute resolution

```python
m_per_deg = 111120.0

resampled = []
for did, g in df.groupby("D_number"):
    g = g.set_index("date_UTC")[["Latitude", "Longitude"]]
    g = g.resample("1min").mean()  # puts NaN where no data
    g = g.interpolate("time")      # linear in time
    g = g.dropna()
    g["D_number"] = did
    resampled.append(g)

df = pd.concat(resampled).reset_index()
df = df.sort_values(["D_number", "date_UTC"]).reset_index(drop=True)

# Derive speed and acceleration from interpolated positions
dt = df.groupby("D_number")["date_UTC"].diff().dt.total_seconds()
dlat = df.groupby("D_number")["Latitude"].diff() * m_per_deg
dlon = df.groupby("D_number")["Longitude"].diff() * m_per_deg * np.cos(np.radians(df["Latitude"]))
df["speed_mps"] = np.sqrt(dlat**2 + dlon**2) / dt

# Smooth speed before differentiating (avoids interpolation kink noise)
df["speed_smooth"] = df.groupby("D_number")["speed_mps"].transform(
    lambda s: s.rolling(15, center=True, min_periods=1).mean()
)
df["accel_mps2"] = df.groupby("D_number")["speed_smooth"].diff() / dt

# Position in metres for beaching detection
df["x_m"] = df["Longitude"] * m_per_deg * np.cos(np.radians(df["Latitude"]))
df["y_m"] = df["Latitude"] * m_per_deg

print(f"{len(df)} records at 1-min resolution")
for did in sorted(df["D_number"].unique()):
    g = df[df["D_number"] == did]
    print(f"  {did}: {len(g)} pts, {g['date_UTC'].min()} to {g['date_UTC'].max()}")
```

## Classify science periods

```python
accel_threshold = 0.002   # m/s², max absolute accel per bin
speed_threshold = 2.0     # m/s, max speed per bin
beaching_std = 5.0        # m, position spread per bin (below = beached)
bin_size = "1h"

# Assign each record to an hourly bin
df["bin"] = df.groupby("D_number")["date_UTC"].transform(
    lambda t: t.dt.floor(bin_size)
)

# Per-bin statistics
bin_stats = df.groupby(["D_number", "bin"]).agg(
    speed_max=("speed_mps", "max"),
    accel_max=("accel_mps2", lambda s: s.abs().max()),
    pos_std_x=("x_m", "std"),
    pos_std_y=("y_m", "std"),
).reset_index()
bin_stats["pos_std"] = np.sqrt(bin_stats["pos_std_x"]**2 + bin_stats["pos_std_y"]**2)

# Classify with priority: beached > too fast > too much accel > science
bin_stats["reason"] = "science"
bin_stats.loc[bin_stats["accel_max"] > accel_threshold, "reason"] = "accel"
bin_stats.loc[bin_stats["speed_max"] > speed_threshold, "reason"] = "fast"
bin_stats.loc[bin_stats["pos_std"] < beaching_std, "reason"] = "beached"

df = df.merge(bin_stats[["D_number", "bin", "reason"]], on=["D_number", "bin"])
df["is_science"] = df["reason"] == "science"

print(f"Science: {df['is_science'].sum()} / {len(df)} records")
print(f"\nBin counts by reason:")
print(bin_stats["reason"].value_counts())
print()
for did in sorted(df["D_number"].unique()):
    g = df[df["D_number"] == did]
    ns = g["is_science"].sum()
    t0 = g.loc[g["is_science"], "date_UTC"].min()
    t1 = g.loc[g["is_science"], "date_UTC"].max()
    print(f"  {did}: {ns} / {len(g)} science, {t0} to {t1}")
```

## All drifter trajectories

```python
tiles = cimgt.OSM()
geo = ccrs.Geodetic()

drifter_ids = sorted(df["D_number"].unique())
colors = {did: f"C{i}" for i, did in enumerate(drifter_ids)}

fig, ax = plt.subplots(subplot_kw=dict(projection=tiles.crs))
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=geo)
ax.add_image(tiles, 8)

for did in drifter_ids:
    g = df[df["D_number"] == did]
    ax.plot(g["Longitude"].values, g["Latitude"].values,
            transform=geo, label=did, color=colors[did])

ax.legend(loc=0)
ax.set_title("All drifter trajectories")
plt.show()
```

## Science periods only

```python
fig, ax = plt.subplots(subplot_kw=dict(projection=tiles.crs), dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=geo)
ax.add_image(tiles, 8)

for did in drifter_ids:
    g = df.loc[df["D_number"] == did].copy()
    lon = g["Longitude"].values.astype(float)
    lat = g["Latitude"].values.astype(float)
    lon[~g["is_science"].values] = np.nan
    lat[~g["is_science"].values] = np.nan
    ax.plot(lon, lat, transform=geo, label=did, color=colors[did])

ax.legend(loc=0)
ax.set_title("Science periods only")
plt.show()
```

## Science / non-science timeline

```python
reason_colors = {
    "science": "C2",
    "beached": "C1",
    "fast": "C3",
    "accel": "C0",
}
reason_labels = {
    "science": "science",
    "beached": f"beached (pos_std < {beaching_std} m)",
    "fast": f"too fast (speed > {speed_threshold} m/s)",
    "accel": f"too much accel (> {accel_threshold} m/s²)",
}

fig, ax = plt.subplots(figsize=(15, 5))

bar_width = pd.Timedelta(bin_size)
for i, did in enumerate(drifter_ids):
    bs = bin_stats[bin_stats["D_number"] == did].sort_values("bin")
    colors = [reason_colors[r] for r in bs["reason"]]
    ax.barh(i, [bar_width] * len(bs), left=bs["bin"].values, height=0.8,
            color=colors, edgecolor="none", linewidth=0)

t_min = bin_stats["bin"].min().floor("D") - pd.Timedelta("1D")
t_max = bin_stats["bin"].max().ceil("D") + pd.Timedelta("1D")
ax.set_xlim(t_min, t_max)

ax.set_yticks(range(len(drifter_ids)))
ax.set_yticklabels(drifter_ids)
ax.set_xlabel("Date")

from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(color=c, label=reason_labels[r]) for r, c in reason_colors.items()],
    loc=0,
)

fig.autofmt_xdate()
plt.show()
```

```python
# Filter out short science segments (< 2 days)
min_science_duration = pd.Timedelta("2D")
reclassified = 0

for did in df["D_number"].unique():
    mask = df["D_number"] == did
    sci = df.loc[mask, "is_science"].values

    # Label contiguous science segments
    segment_id = (sci != np.roll(sci, 1)).cumsum()
    segment_id = pd.Series(segment_id, index=df.loc[mask].index)

    for seg in segment_id[df.loc[mask, "is_science"]].unique():
        seg_mask = mask & (segment_id == seg)
        seg_duration = (
            df.loc[seg_mask, "date_UTC"].max() - df.loc[seg_mask, "date_UTC"].min()
        )
        if seg_duration < min_science_duration:
            n = seg_mask.sum()
            df.loc[seg_mask, "reason"] = "short"
            df.loc[seg_mask, "is_science"] = False
            reclassified += n

# Update bin_stats to match
short_bins = df.loc[df["reason"] == "short", ["D_number", "bin"]].drop_duplicates()
for _, row in short_bins.iterrows():
    bmask = (bin_stats["D_number"] == row["D_number"]) & (bin_stats["bin"] == row["bin"])
    bin_stats.loc[bmask, "reason"] = "short"

print(f"Reclassified {reclassified} records as 'short' (science segment < 2 days)")
print(f"Science after filtering: {df['is_science'].sum()} / {len(df)} records")
```

```python
reason_colors2 = {
    "science": "C2",
    "beached": "C1",
    "fast": "C3",
    "accel": "C0",
    "short": "C4",
}
reason_labels2 = {
    "science": "science",
    "beached": f"beached (pos_std < {beaching_std} m)",
    "fast": f"too fast (speed > {speed_threshold} m/s)",
    "accel": f"too much accel (> {accel_threshold} m/s²)",
    "short": "short segment (< 2 days)",
}

fig, ax = plt.subplots(figsize=(15, 5))

bar_width = pd.Timedelta(bin_size)
for i, did in enumerate(drifter_ids):
    bs = bin_stats[bin_stats["D_number"] == did].sort_values("bin")
    bar_colors = [reason_colors2[r] for r in bs["reason"]]
    ax.barh(i, [bar_width] * len(bs), left=bs["bin"].values, height=0.8,
            color=bar_colors, edgecolor="none", linewidth=0)

t_min = bin_stats["bin"].min().floor("D") - pd.Timedelta("1D")
t_max = bin_stats["bin"].max().ceil("D") + pd.Timedelta("1D")
ax.set_xlim(t_min, t_max)

ax.set_yticks(range(len(drifter_ids)))
ax.set_yticklabels(drifter_ids)
ax.set_xlabel("Date")
ax.set_title("After short-segment filtering")

from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(color=c, label=reason_labels2[r]) for r, c in reason_colors2.items()],
    loc=0,
)

fig.autofmt_xdate()
plt.show()
```

## Save output

```python
out_path = Path(out_csv)
out_path.parent.mkdir(parents=True, exist_ok=True)
df_science = df.loc[df["is_science"], ["D_number", "date_UTC", "Latitude", "Longitude"]]
df_science.to_csv(out_path, index=False)
print(f"Saved {len(df_science)} science records to {out_path}")

```
