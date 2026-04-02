# Examples Rewrite: Two Distinct Example Sets

## Executive Summary

Restructure `examples/` into two orthogonal, clearly-scoped example sets:

1. **Idealized Flow** — pure drifter physics in synthetic flows and wave fields, no external data
2. **Baltic Drifters** — realistic validation using CMEMS + observation data

Each set is self-contained. The library (`src/drogued_drifters/`) provides the core components: `DroguedDrifter`, `compute_stokes_profile()`, and interpolator helpers.

---

## Current State (after initial implementation)

### Set 1: Idealized Flow — DONE

**Directory**: `examples/idealized_flow/`

Three notebooks covering pure drifter physics in controlled conditions:

| Notebook | Purpose | Status |
|----------|---------|--------|
| `01_synthetic_flow_profiles.ipynb` | DD in sheared profile; intermediate speed between surface and drogue depth | Done |
| `02_sheared_jet_parcels.ipynb` | Same physics coupled to Parcels v4 for Lagrangian particle tracking | Done |
| `03_drogued_drifter_in_wave_orbitals.ipynb` | Stokes drift reference levels: sympy Lagrangian derivation comparing 4 objects in progressive waves | Done |

**Key physics in notebook 03**:
- Lagrangian mechanics derivation (sympy): buoy constrained to free surface z=η(x,t), rigid pole, drogue at depth
- Four comparison objects:
  1. Surface passive tracer at z=η (Lagrangian on free surface)
  2. Deep passive tracer at z=−l (free Lagrangian parcel)
  3. Extended drogued drifter in full wave orbital velocities (2D Euler-Lagrange)
  4. Production `DroguedDrifter` from `src/` driven by steady Stokes profile
- Comparison table: mean drifts vs Stokes drift at various reference levels
- Orbital trajectory plot and x-z position time series
- Drogue drag is horizontal only (crossed plates are edge-on vertically)
- 3D→2D projection: `theta_xz = atan2(sin(theta)*cos(phi), cos(theta))` maps the src model's spherical pendulum to the notebook's planar angle
- Parameter cell tagged `"parameters"` for papermill; `DroguedDrifter` constructed with explicit params from notebook

**z-convention**: z positive upward throughout (matching src/ after migration)

**Dependencies**: numpy, scipy, sympy, matplotlib, xarray, Parcels v4 (notebook 02 only)

**Runtime**: ~1 min (01), ~5 min (02), ~30 s (03)

---

### Set 2: Baltic Drifters — DONE

**Directory**: `examples/baltic_drifters/`

**Purpose**: Validate the drogued drifter model against real observations in the southern Baltic Sea.

**Physics narrative**:
- CMEMS provides Eulerian velocity profiles (5 depth levels, 0.5–4.7 m)
- CMEMS provides wave partitions; Stokes drift profiles via deep-water dispersion
- Effective current = Eulerian + Stokes (varies with depth)
- DD model + effective currents + Parcels → simulated trajectories
- Compare against observed drifter tracks (6 buoys, April 2023)

**Dependencies**: Parcels v4, copernicusmarine, numpy, xarray, pandas, matplotlib, scipy

**Notebooks**:

| Notebook | Purpose | Status |
|----------|---------|--------|
| `00_extract_science_periods.ipynb` | Extract science periods from raw drifter GPS data; 1-min resampled positions, 1h-binned classification (speed/accel/beaching thresholds), 2-day minimum segment filter | Done |
| `01_fetch_cmems_data.ipynb` | Data acquisition and caching; time range from 00's science period coverage (full days); overview plots of mean currents, Stokes, landmask | Done |
| `02_derive_effective_currents.ipynb` | Eulerian + Stokes → effective currents; Liu et al. 2020 citation; maps of mean speed at 0m and 3m | Done |
| `03_validate_effective_current_fields.ipynb` | Sample effective currents along observed tracks (surface + 3m); speed and U/V component time series; land extrapolation via rolling fill | Done |
| `04_run_simulations.ipynb` | Full-duration forward integration (DD, surface PP, 3m PP); output to `output/*.zarr` | Done |
| `05_validation_plots.ipynb` | Trajectory maps (2×3), separation distance curves, summary statistics; science-period obs at 1-min resolution | Done |
| `06_run_short_simulations.ipynb` | 12h segments restarted every 12h from observed positions; loops over release times to handle Parcels runtime semantics; output to `output/short_*.zarr` | Done |
| `07_short_horizon_skill.ipynb` | Separation vs lead time (0–12h) with individual + mean lines per drifter; trajectory maps; summary stats at 1/3/6h | Done |

**Notebooks removed**: 03_clean_observations (superseded by 00), 08 (redundant with 10), old 01-05 (absorbed into idealized_flow set)

---

## Package API Used

### Core Classes & Functions
- **`DroguedDrifter`**: Main class; created with geometry parameters (default Callies et al. 2017)
- **`DroguedDrifter.get_final_drift()`**: Scalar drift velocity from currents
- **`DroguedDrifter.get_final_drift_batch()`**: Vectorized for N particles
- **`DroguedDrifter.get_full_solution()`**: Time-series solution (xarray Dataset)
- **`compute_stokes_profile()`**: Depth-dependent Stokes from surface + period
- **`make_profile_sampler()`**: Fast depth interpolator for Parcels
- **`make_dd_velocity_interpolator()`**: Custom interpolator for Parcels integration

---

## Directory Structure

```
examples/
├── idealized_flow/                              ← DONE
│   ├── README.md
│   ├── 01_synthetic_flow_profiles.ipynb
│   ├── 02_sheared_jet_parcels.ipynb
│   ├── 03_drogued_drifter_in_wave_orbitals.ipynb
│   └── output/
│
├── baltic_drifters/                             ← DONE
│   ├── 00_extract_science_periods.ipynb
│   ├── 01_fetch_cmems_data.ipynb
│   ├── 02_derive_effective_currents.ipynb
│   ├── 03_validate_effective_current_fields.ipynb
│   ├── 04_run_simulations.ipynb                  → output/*.zarr
│   ├── 05_validation_plots.ipynb                  ← reads zarr
│   ├── 06_run_short_simulations.ipynb             → output/short_*.zarr
│   ├── 07_short_horizon_skill.ipynb               ← reads short zarr
│   ├── data/
│   └── output/
```

---

## Key Design Principles

1. **Independence**: Each example set runs in isolation; no cross-dependencies
2. **z-up convention**: z positive upward throughout (`src/` and all notebooks)
3. **Clarity**: One physics concept per notebook (where practical)
4. **Vanilla plotting**: No custom colormaps, figsize; let xarray/matplotlib handle labels
5. **No fabricated summaries**: Every result is computed dynamically
6. **Parametric**: Parameter cells tagged for papermill; derived quantities in separate cells

---

## Remaining Work

### Optional Enhancements
- `examples_utils.py` module for shared Baltic notebook helpers
- Shallow-water wavenumber adjustment in `stokes.py`
- Automated notebook testing with papermill + pytest
- Parameter sensitivity notebook (k_d, m_tilde_d)
- README files for each example set
