# Repository Structure

## Directory Tree

```
2025_drogued_drifters/
├── src/drogued_drifters/          # Main package
│   ├── __init__.py                # Package entry point, exports DroguedDrifter
│   ├── drifter.py                 # Core DroguedDrifter class for ODE integration
│   ├── lagrange_model.py           # Lagrangian model formulation & coordinate transforms
│   ├── stokes.py                   # Stokes drift utilities
│   ├── cli.py                      # CLI for code generation
│   └── _generated_eom.py           # AUTO-GENERATED equations of motion (SymPy codegen)
│
├── examples/
│   └── baltic_drifters/            # Jupyter notebooks for Baltic Sea drifter study
│       ├── 00_get_cmems_data.ipynb
│       ├── 01_idealized_flow.ipynb
│       ├── 02_cmems_point_particles.ipynb
│       ├── 03_cmems_drogued_drifter.ipynb
│       ├── 04_stokes_analysis.ipynb
│       ├── 05_wave_orbital_effects.ipynb
│       ├── 06_effective_current_fields.ipynb
│       ├── 07_clean_drifter_data.ipynb
│       ├── 08_drifter_in_effective_currents.ipynb
│       ├── 09a_simulation.ipynb
│       ├── 09b_validation_plots.ipynb
│       ├── 10_along_track_validation.ipynb
│       ├── 11a_reseeded_simulation.ipynb
│       ├── 11b_reseeded_plots.ipynb
│       ├── 12_parameter_sensitivity.ipynb
│       └── data/                   # Data files for Baltic example workflows
│
├── notebooks/                      # General analysis & testing notebooks
│   ├── 01_download_baltic_data.ipynb
│   ├── 02_test_with_sheared_flow.ipynb
│   ├── 03_test_with_cmems_data.ipynb
│   ├── 04_steady_state_derivation.ipynb
│   ├── 05_baltic_stokes_drift.ipynb
│   ├── 06_baltic_surface_currents.ipynb
│   ├── 07_stokes_vs_currents.ipynb
│   ├── 08_drifter_in_waves.ipynb
│   ├── 09_collect_drifter_data.ipynb
│   └── data/                       # Supporting data files
│
├── tests/
│   └── test_drogued_drifter.py     # Unit tests for DroguedDrifter
│
├── plans/                          # Project planning & documentation
│   ├── done/                       # Archive of completed work items
│   ├── structure.md                # This file: repository layout
│   ├── summary.md                  # Project summary
│   ├── roadmap.md                  # High-level next steps
│   ├── api_review.md               # Public API documentation & status
│   ├── analytical_steady_state.md  # Steady-state velocity derivation notes
│   ├── fsolve_steady_state.md      # Steady-state computation via root-finding
│   ├── outlook_bsh_hbm_integration.md # Plan for Baltic Sea HBM integration
│   ├── parcels_v4_integration.md   # OceanParcels v4 integration notes
│   ├── d1_sympy_codegen.md         # SymPy code generation documentation
│   ├── d2_phi_regularization.md    # Pole angle regularization in stereographic coords
│   ├── d3_notebook_cleanup.md      # Notebook refactoring notes
│   └── d4_alpha_derivation.md      # Drogue angle parameterization
│
├── data/                           # Sample data outputs
│   ├── drifter_303_map.png
│   └── drifter_303_timeseries.png
│
├── pyproject.toml                  # Python project metadata & pixi config
├── pixi.lock                        # Locked dependency versions
├── README.md                        # Quick-start guide
├── AGENTS.md                        # Agent/Claude Code guidelines
└── .gitignore                       # Git exclusions
```

## Component Descriptions

### Core Package: `src/drogued_drifters/`

**drifter.py** — Main API for simulating drogued drifters. Exports the `DroguedDrifter` class with methods:
- `get_full_solution()` — Integrate ODE with full state trajectory
- `get_final_drift()` — Compute steady-state drift velocity
- Accepts custom velocity field callbacks for arbitrary ocean models

**lagrange_model.py** — Lagrangian mechanics foundation:
- Symbolic derivation of equations of motion using SymPy
- Coordinate transformations between stereographic (internal) and spherical (API)
- Conversion functions: `_spherical_to_uv()`, `_uv_to_spherical()`, `_uv_to_theta()`

**_generated_eom.py** — Auto-generated compiled equations of motion. Functions `compute_F` and `compute_M` produced by SymPy codegen, not hand-written.

**cli.py** — Command-line interface for code generation:
- `generate-eom` command regenerates `_generated_eom.py` from symbolic model
- Includes hash-based freshness checking

**stokes.py** — Utilities for wave-induced Stokes drift calculations

**\_\_init\_\_.py** — Package entry point; imports `DroguedDrifter` and handles fallback if generated EOM is missing

### Examples & Workflows: `examples/` and `notebooks/`

**examples/baltic_drifters/** — Complete end-to-end workflow for Baltic Sea drifter study:
- Data acquisition (CMEMS currents, wave data)
- Synthesis: point particles → drogued drifters
- Validation: simulation vs. observed drifter data
- Sensitivity analysis

**notebooks/** — Foundational analysis:
- Basic model testing (idealized shear flows, synthetic data)
- Steady-state equation derivation & numerical verification
- Stokes drift & wave orbital motion analysis
- Real ocean data (CMEMS) integration

### Testing: `tests/`

**test_drogued_drifter.py** — Unit tests for core model functionality

### Planning: `plans/`

Active planning documents:
- **api_review.md** — Tracks public API contract and TODOs
- **analytical_steady_state.md**, **fsolve_steady_state.md** — Steady-state theory & implementation
- **outlook_bsh_hbm_integration.md** — Next-phase integration with high-resolution Baltic HBM model
- **d1_sympy_codegen.md** through **d4_alpha_derivation.md** — Technical deep-dives on recent work

**plans/done/** — Archive of completed planning items

### Configuration

**pyproject.toml** — Project metadata, dependencies (NumPy, SciPy, SymPy, xarray, Parcels), pixi workspace config

**pixi.lock** — Reproducible environment lock file (pixi package manager)

## Key Concepts

- **State vector** (internal): `[x, y, u, v, xd, yd, ud, vd]` in stereographic coordinates
  - `(x, y)` — buoy position
  - `(u, v)` — stereographic pole direction (equilibrium at origin)
  - `(xd, yd, ud, vd)` — time derivatives

- **Output API** (spherical): `x, y, theta, phi, xd, yd, thetad, phid` 
  - `theta, phi` — spherical pole angles (user-facing)

- **Drogue geometry**: Rigid pole connecting buoy to subsurface drogue; tunable pole length/angle

- **Ocean model input**: Custom velocity callbacks (U_b, V_b at buoy depth; U_d, V_d at drogue depth)

## Workflow

1. **Setup**: `pixi install` creates locked environment
2. **Code generation**: `pixi run generate-eom` updates `_generated_eom.py` from symbolic model
3. **Testing**: `pixi run pytest` runs test suite
4. **Notebooks**: Execute example workflows in `examples/baltic_drifters/` or `notebooks/`
5. **API usage**: See README.md for quick start
