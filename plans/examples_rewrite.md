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

### Set 2: Baltic Drifters — TODO

**Directory**: `examples/baltic_drifters/` (to be refined from current 13 notebooks)

**Purpose**: Validate the drogued drifter model against real observations in the southern Baltic Sea.

**Physics narrative**:
- CMEMS provides Eulerian velocity profiles (5 depth levels, 0.5–4.7 m)
- CMEMS provides wave partitions; Stokes drift profiles via deep-water dispersion
- Effective current = Eulerian + Stokes (varies with depth)
- DD model + effective currents + Parcels → simulated trajectories
- Compare against observed drifter tracks (6 buoys, April 2023)

**Dependencies**: Parcels v4, copernicusmarine, numpy, xarray, pandas, matplotlib, scipy

**Target notebooks** (refined from current):

| Notebook | Purpose | Based on |
|----------|---------|----------|
| `00_fetch_cmems_data.ipynb` | Data acquisition and caching | Existing 00 |
| `01_derive_effective_currents.ipynb` | Eulerian + Stokes → effective currents | Existing 06 |
| `02_clean_observations.ipynb` | GPS data QA, phase detection | Existing 07 |
| `03_validate_effective_current_fields.ipynb` | Diagnostic: sample fields along observed tracks | Existing 10 |
| `04_run_simulations.ipynb` | Forward integration (DD, surface PP, drogue-depth PP) | Existing 09a+09b |
| `05_validation_plots.ipynb` | Skill metrics, trajectory maps, separation curves | Existing 09b |
| `06_parameter_sensitivity.ipynb` (optional) | Sensitivity to k_d, m_tilde_d | Existing 12 |

**Notebooks to remove**: 08 (redundant with 10), old 01-05 (absorbed into idealized_flow set)

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
├── baltic_drifters/                             ← TODO (refactor from current)
│   ├── README.md
│   ├── 00_fetch_cmems_data.ipynb
│   ├── 01_derive_effective_currents.ipynb
│   ├── 02_clean_observations.ipynb
│   ├── 03_validate_effective_current_fields.ipynb
│   ├── 04_run_simulations.ipynb
│   ├── 05_validation_plots.ipynb
│   ├── 06_parameter_sensitivity.ipynb           (optional)
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

### Baltic Drifters Refactor
1. Consolidate current 13 notebooks into 6–7 focused notebooks
2. Ensure z-up convention throughout (see `plans/z_convention_upward.md`)
3. Use `compute_stokes_profile()` from `src/` for wave partition handling
4. Remove redundant notebooks (08, old idealized/wave notebooks)
5. Write README

### Optional Enhancements
- `examples_utils.py` module for shared Baltic notebook helpers
- Shallow-water wavenumber adjustment in `stokes.py`
- Automated notebook testing with papermill + pytest
