# Outlook: BSH-HBM integration

Replace CMEMS Nemo-Nordic (~1.85 km, hourly, A-grid) with BSH-HBMnoku
(~900 m, 15 min, C-grid) for the Kiel Bight drifter simulations.

## Data overview

The BSH operational model (HBM, contact: opmod@bsh.de) provides four
file types on a common grid (630×387×25 layers):

| File | Variable | Dims | Size |
|------|----------|------|------|
| `c_file_fine` | uvel, vvel, wvel | (time, layer, lat, lon) | ~90 MB/6h |
| `h_file_fine` | cellthick | (time, layer, lat, lon) | ~31 MB/6h |
| `t_file_fine` | temp, salt | (time, layer, lat, lon) | ~30 MB/6h |
| `z_file_fine` | elev (SSH) | (time, lat, lon) | small |

Public access: `https://gdi.bsh.de/data/OpenData/OperationalModel/`
Naming: `{type}_file_fine_{YYYY}/{type}_file_fine_{YYYYMMDDHH}_000_006.nc`
User also has data on HPC disk: `h_file_fine_YYYY/h_file_fine_YYYYMMDD....nc`

## Grid properties

- **Horizontal**: ~900 m (0.0139° lon × 0.0083° lat), regular
- **Coverage**: 6.17–14.91°E, 53.23–56.45°N
- **Vertical**: 25 layers, general vertical coordinate (z*-type)
  - Layer thicknesses vary with (x, y, t) — depend on bathymetry and SSH
  - At Kiel Bight (~15 m depth): 7 wet layers, ~2 m each
  - Layer 1 center ~1 m, layer 2 center ~3 m — good resolution for drogue at 3 m
  - SSH-driven thickness variation is small (~1 cm/6h) but can be significant
    during storm surges (~0.5 m → ~15% depth shift)
- **Staggering**: C-grid (uvel at east face, vvel at south face, wvel at upper face)
- **Time**: 15-minute output, 4 runs/day (00, 06, 12, 18 UTC), 6h per file

## Data volume for drifter period

Apr 24 – May 10, 2023 (17 days):
- Full domain c+h: 136 files, ~8 GB
- Subsetted to Kiel Bight (9.8–11.5°E, 54.1–55.1°N): ~0.5 GB

## Key integration challenges

### 1. Time-varying vertical coordinate

`cellthick(time, layer, lat, lon)` must be cumulatively summed to get
`depth(time, layer, lat, lon)`. This is a 4D depth field.

Options:
- **A. Time-varying depth in Parcels**: Pass depth as a `Field` object.
  Parcels supports this but requires careful setup.
- **B. Time-mean approximation**: Average cellthick over the 17-day period,
  compute static depth levels. Simpler but introduces ~0.5 m error during
  storm surges (~15% at drogue depth).

Recommendation: Start with B, validate against A if results are sensitive.

### 2. C-grid staggering

Velocities are on cell faces, not centers. Parcels has C-grid support
(`FieldSet.from_c_grid_dataset()` or via SGRID conventions with proper
topology attributes). Need to verify that the BSH files have or can be
augmented with the right CF/SGRID metadata.

### 3. No Stokes drift in HBM

HBM provides only Eulerian currents (no wave coupling). Stokes drift
must still come from CMEMS waves (`cmems_mod_bal_wav_anfc_PT1H-i`),
interpolated to the BSH grid and added to the effective current.

The wave data is hourly while HBM is 15-min — need to handle the
temporal mismatch (interpolate waves to 15-min, or average HBM to 1h).

Crucially, the Stokes drift must be mapped onto the C-grid face points
(u-Stokes on east faces, v-Stokes on south faces) and respect the same
zero-normal-flow land boundary condition as the Eulerian velocities.
Otherwise the Stokes contribution would reintroduce artificial beaching
at the coast. In practice: interpolate CMEMS Stokes to the BSH grid,
then zero out face-normal components at land boundaries (same mask as
the Eulerian fields).

Note: Stokes drift is not horizontally non-divergent — its divergence is
balanced by a return flow in the Eulerian mean and by vertical Stokes
transport. Zeroing the normal component at land faces therefore
introduces a small artificial divergence at the coast. This is
acceptable because: (a) the CMEMS wave model doesn't resolve the surf
zone at 1.85 km anyway, (b) Stokes drift at 3 m drogue depth is small
(~6 mm/s mean vs ~10 cm/s Eulerian currents), and (c) the BSH Eulerian
field already implicitly contains wave-driven return flow via the wind
stress forcing. The divergence error is negligible compared to other
error sources.

### 4. Multiple fill values

The NetCDF files use two fill values per variable (`-31111` and `-32222`),
which triggers xarray warnings. Need `decode_cf=True` and possibly manual
NaN handling.

### 5. No artificial beaching

On a C-grid, velocities live on cell faces with zero normal flow at land
boundaries. Parcels particles follow the flow and cannot enter land
cells — the A-grid beaching artifact that dominated the CMEMS validation
disappears entirely. The only beaching is physical: the 3 m drogue
touches bottom in shallow water. The h_file provides the actual water
column depth at each point, so true grounding can be detected and
handled correctly.

## Implementation steps

1. **Data download/subset**: Script to download c+h files for the drifter
   period and subset to Kiel Bight bounding box.
2. **Vertical coordinate**: Compute depth from cumulative cellthick.
   Decide on time-varying vs time-mean approach.
3. **FieldSet construction**: Build Parcels FieldSet from BSH C-grid data
   with proper staggering. Add Stokes drift from CMEMS waves.
4. **Validation notebook**: Run the same 6-drifter simulation as notebook
   09a but with BSH currents. Compare separation/skill against CMEMS.
5. **Re-seeded validation**: Repeat notebook 11a/b with BSH currents.

## Expected improvements

- C-grid eliminates artificial beaching → full-length trajectories (only physical grounding at <3 m depth)
- Higher temporal resolution (15 min vs 1h) → better tidal current representation
- Vertical velocity available → could be used for more realistic drogue dynamics
- Finer horizontal resolution → better mesoscale/submesoscale features
