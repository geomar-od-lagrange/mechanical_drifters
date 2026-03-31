# Examples Rewrite: Three Distinct Example Sets

## Executive Summary

Restructure `examples/` into three orthogonal, clearly-scoped example sets that showcase different physics and use cases of the drogued drifter model:

1. **Idealized Flow** — pure drifter physics in a synthetic flow, showing intermediate trajectory behavior
2. **Wave Orbitals** — wave filtering and Stokes drift characterization, justifying the Stokes profile approach
3. **Baltic Drifters** — realistic validation using CMEMS + observation data, demonstrating skill and robustness

Each set is self-contained and can run independently without understanding the others. The library (`src/drogued_drifters/`) provides the core components: `DroguedDrifter`, `compute_stokes_profile()`, and interpolator helpers.

## Current State

The codebase currently has:
- **Package**: `src/drogued_drifters/` with `drifter.py` (DroguedDrifter class), `stokes.py` (Stokes profile computation), and `lagrange_model.py` (symbolic EOM via srepr + CSE)
- **Tests**: `tests/test_drogued_drifter.py` with comprehensive unit and integration tests
- **Examples**: `examples/baltic_drifters/` with 13 notebooks (00-12) that mix idealized flow, wave analysis, and Baltic validation

## Vision

Each example set demonstrates one aspect of the physics:
- **Idealized**: Does the model work correctly in controlled conditions?
- **Wave Orbitals**: Can we use Stokes profiles instead of explicit orbital velocities?
- **Baltic**: Does it predict real drifter trajectories with meaningful skill?

---

## Overview

The current `examples/baltic_drifters/` directory is a monolithic case study that mixes three distinct scientific narratives:

1. **Idealized flow physics** — pure drogued drifter model behavior in vertically sheared synthetic flow
2. **Wave orbital filtering** — how the drifter's inertia filters out high-frequency wave motion
3. **Baltic case study** — realistic validation against observations using Parcels + CMEMS + Stokes profiles

This plan reorganizes the examples into three independent, self-contained directories, each with a clear scientific purpose and minimal dependencies. Each set uses only the parts of `src/drogued_drifters/` that are essential to its narrative.

---

## What Each Example Set Will Teach

### Set 1: Idealized Flow (2 notebooks, ~200 lines)
**Who**: Users learning what a drogued drifter is and how the model works  
**What**: Synthetic sheared flow with no external data dependencies  
**Results**: Trajectories lie between surface and depth-fixed particles  
**Imports from package**: `DroguedDrifter`, `make_dd_velocity_interpolator()`, `make_profile_sampler()`

### Set 2: Wave Orbitals (2 notebooks, ~300 lines)
**Who**: Users wanting to understand wave effects and Stokes drift  
**What**: Pendulum filtering, monochromatic and multi-partition wave tests  
**Results**: Stokes profile ≈ orbital velocity; transfer function plot  
**Imports from package**: `DroguedDrifter`, `compute_stokes_profile()`

### Set 3: Baltic Drifters (6 notebooks: 4 core + 2 optional, ~2000 lines)
**Who**: Users validating against real data and assessing model skill  
**What**: Full pipeline: CMEMS download → effective currents → observation cleaning → simulation → validation  
**Results**: Skill metrics, trajectory maps, parameter sensitivity  
**Imports from package**: `DroguedDrifter`, `compute_stokes_profile()`, `make_dd_velocity_interpolator()`, `make_profile_sampler()`

---

## Key Design Principles

1. **Independence**: Each example set runs in isolation; no cross-dependencies
2. **Clarity**: One physics concept or stage per notebook (where practical)
3. **Vanilla plotting**: No custom colormaps, figsize defaults; let xarray/matplotlib handle labels
4. **No fabricated summaries**: Every result is computed; no placeholder cells
5. **Immediate rerun**: After fixing bugs, rerun without asking
6. **Narrative arc**: Simple → complex within each set

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

### Utility Functions
- **`drogue_drag_coeff()`, `buoy_drag_coeff()`**: Parameterization from geometry
- **`drogue_added_mass()`, `buoy_added_mass()`**: Hydrodynamic parameters

### Low-Level (used internally)
- **`M_func()`, `F_func()`**: Mass matrix and force vector (from lagrange_model)

---

## Example Set 1: Idealized Flow

**Directory**: `examples/idealized_flow/`

**Purpose**: Demonstrate that a drogued drifter in vertically sheared flow follows a path intermediate between surface and depth-anchored point particles. Shows the model works correctly without needing external data.

**Physics narrative**:
- Two opposing, meandering jets with exponential depth decay and Ekman-like rotation
- Drogue anchors the buoy to depth z = 3 m
- Under sufficient shear, the pole tilts and the buoy+drogue system drifts at a compromise velocity
- Drifter travels slower than surface particles, faster than drogue-depth particles

**Dependencies**:
- `src/drogued_drifters/drifter.py`: `DroguedDrifter` class, `make_dd_velocity_interpolator`
- `src/drogued_drifters/lagrange_model.py`: mass matrix and force evaluations (internal to `DroguedDrifter`)
- Parcels v4 (recent git sha from main branch, for the new Sgrid conventions)
- NumPy, SciPy, xarray, matplotlib

**Notebooks**:

### `01_synthetic_flow_field.ipynb`
**Cells**:
1. Markdown: Title and narrative (drogued drifter concept, why shear matters)
2. Markdown: Physical parameters (flow, drifter geometry)
3. Code: Import packages
4. Code: Define flow parameters (U_0, H, L_Y, JET_SEP, etc.)
5. Code: Build synthetic velocity field using streamfunction (meandering jets, Ekman rotation)
6. Markdown: Visualization of the flow field
7. Code: Contour plot of streamfunction at surface and depth levels
8. Markdown: Section on drogued drifter physics
9. Code: Instantiate `DroguedDrifter`, set up Parcels FieldSet with custom velocity interpolator
10. Markdown: Particle release strategy
11. Code: Define release grid (5 x-positions × 10 y-positions spanning both jets)
12. Code: Run three simulations (DD, surface PP, drogue-depth PP) with separate FieldSets
13. Markdown: Results and interpretation
14. Code: Plot trajectories overlaid on streamfunction contours (3 overlays: DD, surface, drogue-depth)
15. Code: Compute and display mean drift speeds for each particle type
16. Code: Optional: scatter plot of final positions colored by particle type

**Expected output**:
- Zarr files (3 simulations × 50 particles = 150 traces)
- Trajectory plots showing DD traveling at intermediate speed
- Mean speed comparison table

**Key code patterns**:
- Use `FieldSet.from_sgrid_conventions(ds, mesh="flat")` (Parcels v4)
- `make_dd_velocity_interpolator()` wraps the drifter ODE solver for Parcels
- Warm-state caching: `dd_warm_state = {}` passed to interpolator to reuse ODE solver state across particles

---

## Example Set 2: Wave Orbital Filtering

**Directory**: `examples/wave_orbitals/`

**Purpose**: Show that a drogued drifter's finite inertia filters out wave-frequency oscillations, and that a Stokes profile (depth-dependent) approximation is equivalent to explicit wave orbital velocities for the drifter's timescale.

**Physics narrative**:
- Pole-pendulum mode has a period of ~27 seconds (heavily damped)
- A transfer function H(ω) = 1 / (1 + (ω/ω_p)²) characterizes the response
- For Baltic wind-wave periods (T = 2–4 s), ω/ω_p ≫ 1, so H ≈ 0 (waves are filtered out)
- Conclusion: Stokes profile is a valid model for drifter forcing

**Dependencies**:
- `src/drogued_drifters/drifter.py`: `DroguedDrifter` class
- `src/drogued_drifters/stokes.py`: `compute_stokes_profile` function (if used; can also use hard-coded deep-water exponential)
- NumPy, SciPy, matplotlib

**Notebooks**:

### `01_pendulum_transfer_function.ipynb`
**Cells**:
1. Markdown: Title and motivation (why filter wave forcing?)
2. Markdown: Pole-pendulum theory (natural period, damping, transfer function)
3. Code: Import packages, set up matplotlib
4. Code: Define pendulum natural frequency ω_p from drifter parameters
5. Code: Compute transfer function H(ω) for a frequency range
6. Code: Plot H(ω) with marked wave periods (wind waves T=2–4 s, swell T=8–15 s)
7. Markdown: Interpretation (waves are filtered; Stokes profile is OK)

**Output**: Transfer function plot with wave periods marked

### `02_monochromatic_wave_test.ipynb`
**Cells**:
1. Markdown: Test setup (monochromatic wave, steady-state drift comparison)
2. Code: Define a single wave with period T = 3 s, surface amplitude A = 0.5 m
3. Code: Derive orbital velocity amplitude u_orb = A * ω = A * (2π/T)
4. Code: Create two velocity fields:
   - Field A: explicit wave orbital velocity u(t) = u_orb * cos(ωt)
   - Field B: equivalent Stokes profile using `compute_stokes_profile` (or exponential decay)
5. Code: Initialize `DroguedDrifter` twice, once for each field
6. Code: Run each to steady state (t_span = (0, 100) seconds, well after transient)
7. Code: Extract final drift velocities and compare (should converge within ~1 mm/s)
8. Markdown: Results
9. Code: Plot time series of position and velocity for both cases (two subplots)
10. Code: Print RMSE or max absolute difference between drift velocities

**Expected output**:
- Time series plot showing trajectories from both methods converging after ~30 s
- Difference table (should be < 1 mm/s at steady state)

### `03_multipartition_wave_test.ipynb`
**Cells**:
1. Markdown: Test with 3-component wave spectrum (WW, SW1, SW2 from CMEMS partitions)
2. Code: Load sample CMEMS wave partition data (or use synthetic 3-partition spectrum)
3. Code: For each partition, compute Stokes profile and orbital velocity field separately
4. Code: Construct composite velocity field (time-varying, multi-frequency)
5. Code: Run DD model with both orbital and Stokes representations
6. Code: Repeat 10 times with random phase realizations
7. Code: Compute mean drift and std across phase realizations
8. Markdown: Results
9. Code: Difference histogram (orbital vs Stokes) for 10 realizations
10. Code: Summary table: mean difference, std, max difference

**Expected output**:
- Phase-ensemble statistics showing negligible difference (< 2 mm/s std)
- Justification for using Stokes profile in CMEMS-driven simulations

---

## Example Set 3: Baltic Case Study

**Directory**: `examples/baltic_drifters/` (refined from current)

**Purpose**: Validate the drogued drifter model against real observations in the southern Baltic Sea, showing that:
- DD outperforms surface point particles
- DD captures the effective physics when driven by Eulerian currents + Stokes drift
- The model is mechanistically sound

**Physics narrative**:
- CMEMS provides Eulerian velocity profiles (5 depth levels, 0.5–4.7 m)
- CMEMS also provides wave partitions; we derive Stokes drift profiles using deep-water dispersion
- Effective current = Eulerian + Stokes (varies with depth)
- DD model + effective currents + Parcels → simulated trajectories
- Compare simulated trajectories against observed drifter tracks (6 buoys, April 2023)

**Dependencies**:
- `src/drogued_drifters/drifter.py`: `DroguedDrifter`, `make_dd_velocity_interpolator`, `make_profile_sampler`
- `src/drogued_drifters/stokes.py`: `compute_stokes_profile`
- Parcels v4
- `copernicusmarine.open_dataset` (Copernicus Marine SDK)
- NumPy, xarray, pandas, matplotlib, scipy

**Data**:
- CMEMS Baltic physics (hourly, multi-depth)
- CMEMS Baltic waves (hourly, wave partitions)
- Raw GPS drifter trajectories (CSV, 6 buoys)

**Notebooks**:

### `00_fetch_cmems_data.ipynb`
**Cells**:
1. Markdown: Data acquisition strategy
2. Code: Set up Copernicus Marine credentials (if needed)
3. Code: Define spatial and temporal bounds (Kiel Bight, 2023-04-24 to 2023-05-10)
4. Code: Download CMEMS physics and waves using `copernicusmarine.open_dataset`
5. Code: Subset to near-surface physics (0–5 m depth) to minimize data
6. Code: Cache to NetCDF files locally
7. Markdown: Outputs summary

**Outputs**: 
- `cmems_physics.nc` (hourly, 5 depth levels)
- `cmems_waves.nc` (hourly, 3 partitions + surface Stokes)
- `cmems_mask.nc` (land mask)

### `01_derive_effective_currents.ipynb`
**Cells**:
1. Markdown: Effective current concept (Eulerian + Stokes)
2. Code: Load CMEMS physics and wave partitions
3. Code: For each partition, compute Stokes drift at depth levels using `compute_stokes_profile`
4. Code: Accumulate u_stokes, v_stokes across partitions
5. Code: Add to Eulerian: U_eff = uo + u_stokes, V_eff = vo + v_stokes
6. Code: Visualize depth-dependent effective currents (panels: surface, 3m depth)
7. Code: Plot mean/RMS speed maps for Eulerian, Stokes, and effective at two depths
8. Code: Profile plot: time-mean speed vs depth for the three fields
9. Markdown: Results (Stokes contribution ~50% at surface, ~1% at 4.7 m)

**Outputs**: Effective current dataset, visualization plots

### `02_clean_observations.ipynb`
**Cells**:
1. Markdown: Data cleaning workflow (phase detection)
2. Code: Load raw GPS drifter CSV files (6 buoys, IDs 298–303)
3. Code: Detect phases:
   - Pre-deployment: distance < 2 km from dock (54.33°N, 10.15°E)
   - Deployment jump: first record leaving dock
   - Science: rolling mean speed 0.5–50 mm/s, isolated jumps filtered
   - Beached: rolling mean speed < 5 mm/s (shallow water, stalled)
   - Post-beaching: impossible speeds (navy pickups, trash)
4. Code: Extract science-phase records per drifter
5. Code: Visualize timeline (phase bars per drifter) and map (colors by phase)
6. Code: Output clean CSV with UTC timestamp, lon, lat, science phase only

**Outputs**: 
- Clean drifter CSV (9017 science-phase records across 6 drifters)
- Phase timeline plot
- Map of all phases

### `03_validate_effective_current_fields.ipynb`
**Cells**:
1. Markdown: Diagnostic check before simulating (are the fields reasonable?)
2. Code: Load clean observed tracks and effective current fields
3. Code: For each hourly observation, sample CMEMS fields at observed position and extract velocities
4. Code: Compute observed velocity from finite differences (hourly spacing)
5. Code: Compute velocity RMSE, correlation, direction error per drifter
6. Markdown: Results
7. Code: RMSE and correlation bar charts
8. Code: u/v component time series per drifter (observed vs CMEMS surface vs 3 m)
9. Code: Direction error histogram and time series
10. Markdown: Conclusion (RMSE ~0.10 m/s; fields are usable but imperfect; direction errors ~40°)

**Outputs**: Diagnostic plots, RMSE/correlation table

### `04_run_simulations.ipynb`
**Cells**:
1. Markdown: Simulation setup (deployment positions, effective currents, 12-day runtime)
2. Code: Extract deployment position and time for each drifter
3. Code: Set up Parcels FieldSet from effective current dataset
4. Code: Define custom DroguedDrifter kernel for Parcels
5. Code: Create ParticleSets for three simulation types (DD, surface PP, 3m PP)
6. Code: Release particles at observed deployment positions
7. Code: Run 288-hour (12-day) simulations with AdvectionEE/RK4
8. Code: Handle out-of-bounds particles (beaching, land mask)
9. Code: Save zarr files and convert to CSV for downstream analysis
10. Markdown: Simulation summary (particle counts, runtime, output locations)

**Outputs**: 
- Three CSV files (DD, surface PP, 3m PP trajectories)
- Summary of simulation completion

### `05_validation_plots.ipynb`
**Cells**:
1. Markdown: Post-processing and validation metrics
2. Code: Load simulated and observed trajectories
3. Code: Detect beaching (speed < 2 cm/s rolling mean, or land mask)
4. Code: Truncate simulations to earliest beaching time
5. Code: Compute Haversine distance between simulated (interpolated) and observed
6. Code: Compute area-based separation metric (Liu & Weisberg skill score)
7. Markdown: Results
8. Code: Separation vs time curves (mean with error bars, pooled across drifters)
9. Code: Per-drifter trajectory maps (observed + 3 simulations, with coastline)
10. Code: Skill score table per drifter and sim type
11. Code: Summary panel: bar chart of area separation by sim type
12. Markdown: Interpretation
    - DD skill ~0.4–0.6 (moderate)
    - Surface PP skill ~0.1–0.2 (poor)
    - 3m PP skill ~0.9 (excellent short-term, but also beaches quickly)
    - Conclusion: DD mechanistically captures intermediate physics

**Outputs**: 
- Trajectory maps per drifter
- Separation vs time curves
- Skill score summary table

### `06_parameter_sensitivity.ipynb` (optional, advanced)
**Cells**:
1. Markdown: Sensitivity to drogue drag (k_d) and added mass (m_tilde_d)
2. Code: Define 2D parameter grid (k_d: 8 values, m_tilde_d: 8 values)
3. Code: For each grid point, instantiate a DroguedDrifter with modified parameters
4. Code: For all observed positions, solve ODE to steady state using `get_final_drift_batch`
5. Code: Compute RMSE between model-predicted velocity and observed
6. Code: Plot RMSE heatmap with default parameters marked
7. Markdown: Results (RMSE landscape is broad; default values are near optimum but not sharp minimum)

**Outputs**: RMSE heatmap, parameter grid analysis

---

## Source Code Additions / Changes

### 1. Utility Module: `src/drogued_drifters/examples_utils.py`

To avoid repeating code across example notebooks, create a shared utility module with:

```python
def load_effective_current_fields(physics_file, waves_file):
    """Load and combine Eulerian + Stokes fields."""
    # Implementation

def clean_drifter_data(gps_csv_list):
    """Detect phases and extract science segment."""
    # Implementation

def compute_beaching_time(trajectory, land_mask=None, speed_threshold=0.02):
    """Detect when a particle stops (beached)."""
    # Implementation

def compute_separation_metric(sim_trajectory, obs_trajectory):
    """Compute Liu & Weisberg skill score."""
    # Implementation
```

**Rationale**: Baltic notebooks will import these functions to reduce boilerplate and ensure consistency across notebooks.

### 2. Stokes Profile Improvements (Optional)

The current `compute_stokes_profile` uses deep-water dispersion (k = ω²/g), which is crude in shallow water (Baltic ~10–20 m). For the wave orbital notebooks, this is fine (test case is synthetic). For the Baltic case, add a comment noting the limitation and consider adding a shallow-water mode in the future.

### 3. No Changes to `drifter.py` or `lagrange_model.py`

The core physics is solid and used as-is. All three example sets rely on the existing public API:
- `DroguedDrifter.__init__` and `.get_final_drift`
- `DroguedDrifter.get_full_solution` (for time-series output)
- `make_dd_velocity_interpolator` (for Parcels integration)
- `make_profile_sampler` (for fast depth interpolation)

---

## Directory Structure After Reorganization

```
examples/
├── idealized_flow/
│   ├── README.md                    (brief overview)
│   ├── 01_synthetic_flow_field.ipynb
│   └── output/                      (generated zarr/plots)
│
├── wave_orbitals/
│   ├── README.md
│   ├── 01_pendulum_transfer_function.ipynb
│   ├── 02_monochromatic_wave_test.ipynb
│   ├── 03_multipartition_wave_test.ipynb
│   └── output/
│
├── baltic_drifters/
│   ├── README.md
│   ├── 00_fetch_cmems_data.ipynb
│   ├── 01_derive_effective_currents.ipynb
│   ├── 02_clean_observations.ipynb
│   ├── 03_validate_effective_current_fields.ipynb
│   ├── 04_run_simulations.ipynb
│   ├── 05_validation_plots.ipynb
│   ├── 06_parameter_sensitivity.ipynb     (optional)
│   ├── data/                        (cached CMEMS + raw drifter CSVs)
│   └── output/                      (simulations + validation plots)
```

---

## Example-Specific READMEs

Each example directory should have a concise README:

### `idealized_flow/README.md`
```markdown
# Drogued Drifter in Idealized Flow

Demonstrates the core drogued drifter physics in a synthetic velocity field 
with strong vertical shear. Shows that a drogued drifter travels at an 
intermediate speed between surface and depth-anchored point particles.

**Requirements**: Parcels v4, numpy, scipy, matplotlib, xarray
**Runtime**: ~2–5 minutes
**Output**: Trajectory comparisons showing intermediate behavior
```

### `wave_orbitals/README.md`
```markdown
# Wave Orbital Filtering by Drogued Drifter Inertia

Shows that a drogued drifter's finite inertia (pole-pendulum damping) 
filters out wave-frequency oscillations. Demonstrates equivalence between 
Stokes drift profiles and explicit wave orbital velocities for drifter 
timescales. Justifies the use of depth-dependent Stokes profiles in 
Eulerian current fields.

**Requirements**: numpy, scipy, matplotlib
**Runtime**: ~1 minute
**Output**: Transfer function plot, drift velocity comparisons
```

### `baltic_drifters/README.md`
```markdown
# Drogued Drifter Validation in the Baltic Sea

Complete validation study: drogued drifter model driven by realistic 
CMEMS currents (Eulerian + Stokes drift) compared against 6 observed 
buoys in the southern Baltic Sea, April 2023.

**Prerequisite**: Copernicus Marine account (or cached data)
**Requirements**: parcels, drogued_drifters, xarray, pandas, copernicusmarine, matplotlib
**Runtime**: ~30 minutes (if data cached; ~1 hour to fetch CMEMS)
**Output**: Trajectory maps, skill metrics, parameter sensitivity analysis

**Notebooks**:
- 00: Fetch CMEMS data (skip if using cached)
- 01: Combine Eulerian + Stokes into effective currents
- 02: Clean raw drifter GPS data
- 03: Validate fields by sampling along observations
- 04: Run simulations from observed initial conditions
- 05: Compute skill metrics and visualize
- 06 (optional): Parameter sensitivity study
```

---

## Implementation Plan

### Phase 1: Create Example Directories and Notebooks

1. Create `examples/idealized_flow/` with `01_synthetic_flow_field.ipynb`
   - Copy and adapt current notebook 01 from `baltic_drifters`
   - Self-contained; no external data needed
   - Focus on intermediate trajectory behavior

2. Create `examples/wave_orbitals/` with three notebooks
   - `01_pendulum_transfer_function.ipynb`: purely analytical
   - `02_monochromatic_wave_test.ipynb`: synthetic 1-frequency wave
   - `03_multipartition_wave_test.ipynb`: synthetic 3-partition wave (or sample CMEMS data)
   - No Parcels needed; use `DroguedDrifter.get_full_solution` directly

3. Refactor `examples/baltic_drifters/`
   - Rename old notebooks (00, 01, etc.) → new names (00 through 06)
   - Keep the same physics but consolidate where possible
   - Merge notebooks 09a + 09b (simulation + validation plots)
   - Keep 04/05 merged into one wave analysis (if combining wave notebooks; otherwise drop)
   - Remove notebooks 08 and 10 (redundant velocity validation)

### Phase 2: Extract Utility Functions

1. Create `src/drogued_drifters/examples_utils.py` with:
   - CMEMS loading
   - Drifter data cleaning
   - Beaching detection
   - Skill metric computation

2. Update all three example sets to import from `examples_utils` (reduces duplication)

### Phase 3: Create READMEs and Documentation

1. Write brief READMEs for each example directory
2. Update top-level `README.md` to mention the three example sets
3. Add cell-level markdown commentary to justify design choices

---

## Notebook Metadata and Execution Notes

- All notebooks should be runnable in sequence within their directory
- Use relative paths for data (e.g., `./data/cmems_physics.nc`)
- Document expected runtime for each notebook
- Include "Outputs" cells that summarize what was generated

---

## Testing Strategy

After reorganization:

1. **Idealized flow**: Run end-to-end. Should complete in < 5 minutes. Verify trajectory shapes.
2. **Wave orbitals**: All three notebooks run independently. Verify that monochromatic and 3-partition tests show < 2 mm/s difference.
3. **Baltic case**: Run with cached CMEMS data. Verify beaching detection, skill scores match expected ranges.

---

## Future Extensions

- **Stokes profile improvements**: Add shallow-water depth scaling or directional wave spectrum
- **Sensitivity studies**: Extend parameter sensitivity to other drifter geometries
- **Wave spectrum**: Replace 3-partition approximation with full directional spectrum (if justified by results)
- **Forecasting**: Add a "reseeded forecasts" example (6-hourly re-initialization, lead-time skill)

---

## Summary Table: Notebook Purposes and Dependencies

| Example Set | Notebook | Purpose | Core Imports | Runtime |
|---|---|---|---|---|
| Idealized | 01_synthetic_flow_profiles | Direct API test (no Parcels) | DroguedDrifter, numpy | ~1 min |
| Idealized | 02_sheared_jet_parcels | Parcels + DD integration | DroguedDrifter, make_dd_velocity_interpolator, Parcels v4 | ~3 min |
| Wave orbitals | 01_stokes_depth_profile | Profile computation demo | compute_stokes_profile, numpy, matplotlib | ~1 min |
| Wave orbitals | 02_wave_filtering | Orbital vs Stokes comparison | DroguedDrifter, numpy | ~2 min |
| Baltic | 00_download_and_cache_data | Data acquisition | copernicusmarine, xarray | ~20 min (network) |
| Baltic | 01_effective_currents | Field construction | compute_stokes_profile, xarray, matplotlib | ~5 min |
| Baltic | 02_clean_observations | Data QA | pandas, numpy, matplotlib | ~2 min |
| Baltic | 03_velocity_field_validation | Diagnostic check | xarray, pandas, matplotlib | ~5 min |
| Baltic | 04_forward_simulation | Forward integration | DroguedDrifter, make_dd_velocity_interpolator, Parcels v4 | ~15 min |
| Baltic | 05_parameter_sensitivity | Robustness study | DroguedDrifter, numpy, matplotlib | ~10 min (optional) |
| Baltic | 06_reseeded_forecasts | Lead-time skill | DroguedDrifter, Parcels v4, matplotlib | ~20 min (optional) |

---

## Implementation Order & Effort

### Phase 1: Setup & Structure (1 day)
- Create directories: `examples/idealized_flow/`, `examples/wave_orbitals/`
- Keep `examples/baltic_drifters/` but plan notebook consolidation
- Write stub notebooks with markdown structure
- Create README files for each set

### Phase 2: Idealized Flow (2 days)
- Adapt existing `01_idealized_flow.ipynb` → split into two notebooks
- **01_synthetic_flow_profiles**: standalone DroguedDrifter tests (no Parcels)
- **02_sheared_jet_parcels**: Parcels integration (based on existing 01_idealized_flow.ipynb)
- Test: trajectories should lie between surface and drogue-depth cases

### Phase 3: Wave Orbitals (1.5 days)
- **01_stokes_depth_profile**: demo `compute_stokes_profile()`, show decay
- **02_wave_filtering**: monochromatic + multi-partition tests (based on existing 05_wave_orbital_effects.ipynb, condensed)
- Test: orbital vs Stokes drift difference <1 mm/s

### Phase 4: Baltic Drifters Core (4 days)
- **00_download_and_cache_data**: new (based on existing 00_get_cmems_data.ipynb)
- **01_effective_currents**: keep mostly as-is (existing 06_effective_current_fields.ipynb)
- **02_clean_observations**: keep mostly as-is (existing 07_clean_drifter_data.ipynb)
- **03_velocity_field_validation**: based on existing 10_along_track_validation.ipynb
- **04_forward_simulation**: merge existing 09a_simulation.ipynb + 09b_validation_plots.ipynb
- Test: skill metrics in expected ranges (~0.4-0.6 for DD)

### Phase 5: Baltic Drifters Optional (2 days)
- **05_parameter_sensitivity**: based on existing 12_parameter_sensitivity.ipynb
- **06_reseeded_forecasts**: merge existing 11a_reseeded_simulation.ipynb + 11b_reseeded_plots.ipynb
- Test: forecasts skill decay with lead time

### Phase 6: Cleanup & Documentation (1 day)
- Archive old `examples/baltic_drifters/{00-12}` notebooks
- Update top-level examples README
- Add per-notebook docstrings and narrative markdown

**Total estimated effort**: ~10-12 days for planning agent + implementation agent(s)

---

## Library Changes Needed

### No changes to package core required.

All three example sets use the existing public API:
- `DroguedDrifter` instantiation and methods
- `compute_stokes_profile()` for Stokes drift
- `make_dd_velocity_interpolator()` and `make_profile_sampler()` for Parcels v4 integration

### Optional enhancements (not blocking):
1. **`examples_utils.py`** module with helper functions:
   - `load_effective_current_fields()` — combine Eulerian + Stokes
   - `detect_beaching_time()` — from trajectory
   - `compute_separation_metric()` — Liu & Weisberg skill
   - `clean_drifter_gps()` — phase detection
   
   These reduce duplication across Baltic notebooks but are not required for examples to work.

2. **Stokes profile improvements** (future):
   - Shallow-water wavenumber adjustment (currently assumes deep-water k = ω²/g)
   - Directional spectrum support (currently 3-partition approach)

---

## Old Structure → New Structure Mapping

| Old Notebooks | New Location | Consolidation |
|---|---|---|
| 00 | baltic_drifters/00 | Kept, renamed slightly |
| 01 | idealized_flow/02 | Moved, split from 01_idealized_flow.ipynb |
| 02, 03 | idealized_flow/01-02 | Merged and refocused (02 → 01_synthetic_flow_profiles, 03 → 02_sheared_jet_parcels) |
| 04, 05 | wave_orbitals/01-02 | Consolidated and condensed |
| 06 | baltic_drifters/01 | Kept mostly as-is |
| 07 | baltic_drifters/02 | Kept as-is |
| 08 | Removed | Redundant with 10 |
| 09a, 09b | baltic_drifters/04 | Merged (simulation + validation) |
| 10 | baltic_drifters/03 | Renamed for clarity |
| 11a, 11b | baltic_drifters/06 | Merged, optional |
| 12 | baltic_drifters/05 | Reordered, optional |

**Action**: After new notebooks are validated, archive old `examples/baltic_drifters/{00-12}` to `examples/.archive/` or remove entirely.

---

## Validation Checklist

Each example set must:
- [ ] Run without errors (pixi run jupyter)
- [ ] Produce expected physics results:
  - Idealized: trajectories between surface and drogue-depth
  - Wave: orbital vs Stokes drift < 1 mm/s
  - Baltic: DD skill 0.3–0.6, surface skill < 0.3
- [ ] Have clear takeaways (one sentence per notebook)
- [ ] Compute all results dynamically (no hardcoded values)
- [ ] Use vanilla plotting (no custom figsize/colormaps except where xarray defaults don't work)
- [ ] Have no placeholder cells or "TODO: fill in" text
- [ ] Complete in reasonable time (idealized/wave < 5 min, baltic core < 30 min total)

---

## Future Enhancements

Out of scope for this rewrite but worth documenting:

1. **Utility module** (`examples/utils.py`): shared beaching detection, skill metrics, CMEMS loading
2. **Automated testing**: pytest + papermill for example notebooks
3. **Interactive components**: Voila widgets for parameter exploration in idealized flow
4. **Publication-ready styling**: high-DPI maps, consistent color schemes (if needed for paper)
5. **Extended forecasting**: add lead-time dependence of skill; compare to other model classes

---

## Summary

**Before**: 13 monolithic baltic_drifters notebooks mixing three physics concepts, ~400 cells, ~8000 lines with outputs

**After**: Three focused example sets with 9 core notebooks + 2 optional
- **Idealized**: 2 notebooks, ~200 lines, ~15 min runtime, 0 external dependencies
- **Wave Orbitals**: 2 notebooks, ~300 lines, ~3 min runtime, 0 external dependencies
- **Baltic**: 4 core + 2 optional notebooks, ~2500 lines, ~1 hour runtime (with CMEMS fetch), requires Copernicus Marine account

**Benefits**:
- Clear narrative: each set teaches one aspect of physics
- Modular: users pick what they need
- Maintainable: less duplication, clearer dependencies
- Educational: easy to understand progression from simple to complex
