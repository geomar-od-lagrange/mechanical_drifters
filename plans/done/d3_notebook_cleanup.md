# D3: Final notebook cleanup

Per-notebook audit of `examples/baltic_drifters/` against the
conventions in `AGENTS.md`.

---

## Principles (from user feedback)

- **Output files named by notebook number**: e.g., `01_surface_pp.zarr`,
  `09a_sim_drogued_drifter.csv`. Makes it obvious which notebook
  produced which artifact.
- **Summary cells are science, not decoration**: Hardcoded prose
  summaries with fabricated numbers are forbidden. Either remove them
  or replace with code that computes and prints findings. OK to leave
  a placeholder for the human to fill in.
- **RNG seed always a parameter** (where RNG is used). Use
  `np.random.default_rng(RANDOM_SEED)`, not legacy `np.random.seed()`.
- **Unify parameter names across notebooks** where easy: same bbox
  names, same time range conventions, same variable names for shared
  concepts (DT, DROGUE_DEPTH, etc.).
- **DPI and FIG_WIDTH are not parameters** — use matplotlib defaults.
  Inline override only if truly needed (e.g., cartopy map aspect).
- **Keep kernel boilerplate inline** — duplicated kernel code across
  notebooks is acceptable for readability. Do not refactor into a
  shared module.

---

## Common patterns to fix

1. **"tracer" -> "point particle"** + rename output files by notebook
   number
   - 01: `surface_tracer.zarr` -> `01_surface_pp.zarr`, etc.
   - 06: "Lagrangian tracer" -> "Lagrangian particle" in intro

2. **Custom `figsize` calls** — remove, use defaults
   - 04: `figsize=(14, 10)` x2, `figsize=(9, 5)`, `figsize=(12, 4)`
   - 06: `figsize=(14, 8)`
   - 09b: remove `DPI` and `FIG_WIDTH` params, use defaults or inline
   - 10: `figsize=(12, 4)`

3. **Hardcoded summary cells** — remove or replace with code
   - 01: final "## Summary" — prose bullets, no code
   - 02: final "## Summary" — prose bullets
   - 04: final "## Summary" — hardcoded "80-130%", "20-30%"

4. **Legacy RNG API** — replace with `default_rng` + parameter
   - 05: `np.random.seed(7)`, `np.random.RandomState(seed)`
   - 02, 03, 08: inline `default_rng(42)` — move seed to params cell

5. **Scattered parameters** — consolidate into the first code cell
   - 01: `NX, NY, NZ`, `DT`, `RUNTIME`, `OUTPUTDT` scattered
   - 02, 03, 08: `N_PARTICLES`, bounds, `DT`, `RUNTIME` scattered
   - 05: `n_seeds = 10` inline

6. **Non-primitive parameters** — store bounds as floats/strings
   - 04, 06, 08, 10, 11a, 11b, 12: `slice(...)` in params cell
   - 01: `K_MEANDER` computed inline — store wavelength as primitive

7. **Unused/misplaced imports**
   - 10: cartopy imported mid-notebook — move to imports cell
   - 11a: matplotlib imported but unused — remove

---

## Per-notebook checklist

### 01_idealized_flow.ipynb
- [ ] Rename `surface_tracer.zarr` -> `surface_point_particle.zarr`
- [ ] Rename `drogue_depth_tracer.zarr` -> `drogue_depth_point_particle.zarr`
- [ ] Move `NX, NY, NZ` to the parameters cell or document as grid
      internals
- [ ] Move `DT`, `RUNTIME`, `OUTPUTDT` to the parameters cell
- [ ] Store meander wavelength as primitive, compute `K_MEANDER` from it
- [ ] Convert final Summary markdown cell to code-only or remove
- [ ] Narrative markdown is good throughout

### 02_cmems_point_particles.ipynb
- [ ] Create a proper parameters cell with `N_PARTICLES`,
      `LON_BOUNDS`, `LAT_BOUNDS`, `DT`, `RUNTIME`, `OUTPUTDT`,
      `RANDOM_SEED`
- [ ] Move `rng = np.random.default_rng(42)` seed to parameters
- [ ] Convert final Summary markdown cell to code-only or remove
- [ ] Narrative markdown is good

### 03_cmems_drogued_drifter.ipynb
- [ ] Create a proper parameters cell; consolidate scattered params
- [ ] Move random seed 42 to parameters cell
- [ ] No summary cell -- OK
- [ ] Narrative markdown is good

### 04_stokes_analysis.ipynb
- [ ] Parameters cell exists (`LON`, `LAT`, `TIME`) but uses slices;
      store as primitive strings/floats
- [ ] Remove custom `figsize` from all 4 plot cells
- [ ] Convert final Summary markdown cell to code-only or remove
      (contains hardcoded "80-130%", "20-30%")
- [ ] "Summary statistics" section (code cell) is fine -- computes
      and prints
- [ ] Narrative markdown is good

### 05_wave_orbital_effects.ipynb
- [ ] Replace `np.random.seed(7)` with `RANDOM_SEED = 7` parameter
      and `np.random.default_rng(RANDOM_SEED)`
- [ ] Replace `np.random.RandomState(seed)` with `default_rng`
- [ ] Move `n_seeds = 10` to parameters cell
- [ ] No summary cell -- OK
- [ ] Narrative markdown is excellent

### 06_effective_current_fields.ipynb
- [ ] "Lagrangian tracer" -> "Lagrangian particle" in intro
- [ ] Parameters cell uses tuples `LON_BDS`, `LAT_BDS` -- OK as
      primitives, but slices are constructed later
- [ ] Remove `figsize=(14, 8)` from visualization cell
- [ ] No summary cell -- OK
- [ ] Narrative markdown is good

### 07_clean_drifter_data.ipynb
- [ ] Parameters cell is well-structured with all primitives -- good
- [ ] No summary cell -- OK
- [ ] No random seeds -- OK
- [ ] No custom figsize -- OK
- [ ] Narrative markdown is good
- [ ] **This notebook is already clean.**

### 08_drifter_in_effective_currents.ipynb
- [ ] Move `LON_BOUNDS`, `LAT_BOUNDS` to parameters cell
- [ ] Move random seed 42 to parameters cell as `RANDOM_SEED`
- [ ] No summary cell -- OK
- [ ] No custom figsize -- OK
- [ ] Narrative markdown is good

### 09a_simulation.ipynb
- [ ] Parameters cell is well-structured -- good
- [ ] Summary section uses a code cell that prints dynamically -- good
- [ ] But the "## Summary" markdown header before it is bare -- OK,
      it introduces the code cell
- [ ] No random seeds needed (uses observed positions)
- [ ] No custom figsize -- OK (no plots)

### 09b_validation_plots.ipynb
- [ ] `DPI = 200` and `FIG_WIDTH = 5` in parameters -- these are
      custom figsize/DPI values. Consider removing `DPI` and using
      defaults. `FIG_WIDTH` drives `figsize` computation.
- [ ] Remove custom figsize/DPI from plot cells where possible
- [ ] No summary cell -- OK
- [ ] No random seeds -- OK

### 10_along_track_validation.ipynb
- [ ] Parameters cell is good, all primitives
- [ ] `import cartopy` appears mid-notebook (cell 8, "Science period
      tracks") -- move to imports cell
- [ ] Remove `figsize=(12, 4)` from scatter plot
- [ ] No summary cell -- OK
- [ ] Narrative markdown is good

### 11a_reseeded_simulation.ipynb
- [ ] Parameters cell is good, all primitives
- [ ] `import matplotlib.pyplot as plt` is unused (simulation-only)
      -- remove
- [ ] No summary cell -- OK
- [ ] No custom figsize -- OK

### 11b_reseeded_plots.ipynb
- [ ] `DPI = 300` in parameters -- custom DPI, consider removing
- [ ] No custom figsize otherwise -- OK
- [ ] No summary cell -- OK
- [ ] Narrative markdown is adequate but sparse

### 12_parameter_sensitivity.ipynb
- [ ] Parameters cell is good, all primitives
- [ ] No custom figsize -- OK (uses xarray `.plot()` and default
      `plt.subplots()`)
- [ ] No summary cell -- OK
- [ ] Narrative markdown is good

---

## Priority ordering

### High priority (most visible issues, roadmap D3 items)

1. **01_idealized_flow** -- "tracer" in filenames, scattered params,
   hardcoded summary. This is the intro notebook and sets the tone.
2. **04_stokes_analysis** -- hardcoded summary, custom figsize x4,
   non-primitive params.
3. **02_cmems_point_particles** -- hardcoded summary, scattered params,
   random seed not in params.
4. **05_wave_orbital_effects** -- legacy `np.random.seed()` and
   `RandomState`.

### Medium priority (parameter consolidation, minor fixes)

5. **06_effective_current_fields** -- "tracer" wording, figsize.
6. **03_cmems_drogued_drifter** -- scattered params, random seed.
7. **08_drifter_in_effective_currents** -- scattered params, random
   seed.
8. **10_along_track_validation** -- mid-notebook import, figsize.
9. **11a_reseeded_simulation** -- unused import.

### Low priority (already mostly clean)

10. **09b_validation_plots** -- custom DPI/figsize, but these may be
    justified for publication-quality figures.
11. **11b_reseeded_plots** -- custom DPI only.
12. **09a_simulation** -- already clean.
13. **07_clean_drifter_data** -- already clean.
14. **12_parameter_sensitivity** -- already clean.

---

## Candidates for deletion or merging

None of the notebooks should be deleted. Each serves a distinct purpose
in the analysis pipeline.

Kernel boilerplate is duplicated across 01, 03, 08, 09a, 11a — this is
intentional for readability. Do not refactor into a shared module.

---

## Non-notebook cleanup (also part of D3)

These items arose during D1/D2 implementation and should be addressed
in the same cleanup pass:

1. **Y_final public API conversion**: `get_final_drift_batch` should
   return Y_final in spherical coords (x, y, theta, phi, xd, yd,
   thetad, phid), not internal stereographic (u, v, ud, vd). Same for
   warm-start y0 input. Add `_uv_to_spherical` and `_spherical_to_uv`
   helpers to `drifter.py`. Stereographic coords must never leak into
   the public API.

2. **Test section headers**: Remove roadmap labels from test code (e.g.,
   `# D1: Verification of generated code` → descriptive comment like
   `# Generated code vs lambdified sympy`). Tests describe what they
   test, not which task created them.

3. **Remove `scripts/` directory**: Already deleted. Verify it's not
   referenced anywhere (AGENTS.md, plans, etc.).

4. **Create `00_get_cmems_data.ipynb`**: Download and cache all CMEMS
   data (physics + waves) for the drifter deployment period. Downstream
   notebooks load from local files instead of lazy `arco-geo-series`
   downloads. This eliminates the ~1 min download wait in every
   notebook run.

5. **Add click to dependencies**: `scripts/generate_eom.py` used click
   but it may not be in `pyproject.toml` or pixi deps. Check and add
   if missing.