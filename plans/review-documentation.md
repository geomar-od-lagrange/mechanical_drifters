# Documentation review

Review of human-facing documentation: README, docs/, and example notebooks.

## Current state

**What works well:**
- The README is concise and covers the right topics: purpose, quick-start, Parcels integration, setup, tests, and an index of all example notebooks.
- [docs/parcels-v4-coupling.md](../docs/parcels-v4-coupling.md) is excellent -- thorough, well-structured, explains design decisions by comparing alternatives and trade-offs. Follows the "describe what *is*" convention.
- Example notebooks have a consistent structure: imports, parameters, computation, visualization. Parameters are in tagged cells. Markdown narrative is clear and informative without being verbose.
- The idealized-flow notebooks (01, 02, 03) are strong standalone tutorials. Notebook 03 (wave orbitals) is particularly well-done: clear physics setup, symbolic derivation, four objects with distinct purposes, and a clean comparison table.
- Baltic notebooks form a coherent pipeline (00-07) where each notebook's purpose and dependencies are clear from the title and intro cell.

**What's lacking:**
- Only one doc file exists. Key concepts (the physics model, the `DroguedDrifter` API, Stokes drift utilities) have no standalone documentation.
- The README has minor issues: import path inconsistency, missing license, no mention of Python version or key dependencies.
- Notebook narrative is good individually but has some inconsistencies across the set.

---

## Actionable items

### README

**R1. Fix import path inconsistency in quick-start snippet**
The README quick-start uses `from drogued_drifters import DroguedDrifter` (which works via `__init__.py`), but the Parcels section uses `from drogued_drifters.drifter import DroguedDrifter`. Pick one and be consistent -- the short form via `__init__.py` is the public API.
- Priority: **high** -- newcomers will be confused by two different import paths
- Scope: one-liner fix in README Parcels section

**R2. Add a one-sentence description of what "drogued drifter" means physically**
The current opening ("Lagrangian model for drogued ocean drifters") assumes the reader knows what a drogued drifter is. The idealized notebooks have good one-sentence descriptions -- lift something similar to the README opening.
- Priority: **medium** -- helps non-oceanographers
- Scope: add 1-2 sentences after the title

**R3. Drop the internal state-vector paragraph**
Lines 8-13 describe the internal stereographic coordinate representation. This is implementation detail, not user-facing. It will confuse newcomers and will need updating if internals change.
- Priority: **medium** -- actively misleading for users who just want the public API
- Scope: delete one paragraph

**R4. Add license**
No LICENSE file exists at the project root. The README should either state the license or link to one.
- Priority: **high** -- anyone considering using or contributing to this code needs to know the terms
- Scope: add a LICENSE file and a one-liner in README

**R5. Mention Python version and key non-Python dependencies**
`pyproject.toml` requires Python >= 3.11 and pins Parcels to a specific commit. The README should mention this (especially the Parcels pin, since users can't just `pip install parcels`).
- Priority: **medium** -- saves users from a failed install
- Scope: 2-3 lines in the Setup section

**R6. Note that `pixi` is required (not optional)**
The setup section says `pixi install` but doesn't explain what pixi is or link to it. A newcomer who doesn't have pixi installed will be stuck.
- Priority: **medium**
- Scope: one sentence + link to pixi docs

### docs/

**D1. Add `docs/drifter-model.md` -- the physics and API**
There is no documentation of the core physics model. A reader who wants to understand what `DroguedDrifter` does, what parameters it takes, and what `get_full_solution` / `get_final_drift` / `get_final_drift_batch` return has to read the source. This is the single biggest documentation gap.

Should cover:
- Physical model: buoy + pole + drogue, Lagrangian mechanics, steady-state assumption
- `DroguedDrifter` constructor parameters (masses, drag coefficients, pole length, get_uv callback)
- Public methods and their return types (xarray Dataset, batch arrays)
- The stereographic-to-spherical coordinate conversion (so users understand the output variables)
- When to use `get_full_solution` vs `get_final_drift` vs `get_final_drift_batch`

- Priority: **high** -- the most important missing doc
- Scope: significant writing effort, but the content exists scattered across notebook markdown and source docstrings

**D2. Add `docs/stokes-drift.md` -- Stokes drift utilities**
The `stokes.py` module and notebook 03 contain important physics (deep-water monochromatic Stokes drift, partitioned wave fields). A short doc explaining `compute_stokes_profile` and how it's used in the Baltic pipeline would help users who want to build their own effective-current fields.
- Priority: **low** -- only needed by users working with real wave data
- Scope: short doc, mostly pointing to notebook 03 and the function signature

### Notebooks

**N1. Notebook 01 (synthetic flow profiles): separate imports from parameters cell**
The imports cell is labeled "Parameters" in the markdown header above it, but it contains `import numpy as np`, `import matplotlib.pyplot as plt`, etc. The actual parameters (`U_0`, `H`, etc.) are in the next cell. The markdown heading "Parameters" should be above the parameters cell, and the imports cell should have its own "Imports" heading (as other notebooks do).
- Priority: **low** -- cosmetic inconsistency
- Scope: move one markdown heading

**N2. Notebook 01: add a brief intro to the `get_final_drift_batch` API**
The notebook jumps from "Create interpolation functions" to calling `dd.get_final_drift_batch(sample_uv=...)` without explaining what this function expects or returns. A one-sentence markdown cell ("The `get_final_drift_batch` method takes a `sample_uv(z)` callback that returns `(u, v)` arrays at given depths, and returns the steady-state buoy drift velocity") would help.
- Priority: **medium** -- the notebook is a tutorial, readers need to understand the API
- Scope: add 1-2 sentences in a markdown cell

**N3. Notebook 02 (sheared jet): color naming inconsistency in plot legend vs prose**
The markdown says "Surface point particles (red) travel furthest, drogue-depth (green), drogued drifters (blue, dashed)". But the actual plot uses `tab:blue` for surface, `tab:orange` for drogue-depth, and `tab:green` for drogued drifter. The prose colors don't match the code.
- Priority: **medium** -- misleading narrative in a tutorial
- Scope: fix the markdown cell text to match the actual plot colors

**N4. Notebook 05 (validation plots): orphaned first markdown line**
The first markdown cell starts with "Observations are filtered to science periods from notebook 00." as a standalone sentence before the title "# Validation plots: ...". This looks like a leftover from editing. Move it after the title or into the intro paragraph.
- Priority: **low** -- cosmetic
- Scope: reorder one sentence

**N5. Baltic notebooks 00-07: no top-level overview explaining the pipeline**
Each notebook has a clear intro, but there is no high-level narrative explaining the full pipeline: "we start with raw GPS data, extract science periods, fetch forcing, build effective currents, run simulations, and validate." The README lists the notebooks but doesn't explain the flow. A 2-3 sentence paragraph in the README's Baltic section (or a short `examples/baltic_drifters/README.md`) would help a reader decide which notebooks to look at.
- Priority: **medium** -- helps navigation
- Scope: 3-5 sentences in the README Baltic section

**N6. Baltic notebook 04: hardcoded date filter**
Line `df_window = df[df["date_UTC"] >= "2023-04-24"]` uses a hardcoded date that is not in the parameters cell and is not explained. Add a parameter or a comment explaining why this cutoff exists (presumably: effective currents start at this date).
- Priority: **low** -- not a documentation issue per se, but a reader will wonder
- Scope: add one comment or move to parameters

**N7. Notebook 02 uses `AdvectionRK4` but README Parcels example uses `AdvectionEE`**
The README Parcels snippet shows `AdvectionEE` in the kernel name (`DDAdvectEE`), while notebook 02 uses `AdvectionRK4` for the comparison particles. This isn't a bug (the DD kernel is always Euler-forward internally, and the point-particle comparison intentionally uses RK4), but the README could note that the DD kernel does its own Euler-forward step so the `dt` choice matters.
- Priority: **low**
- Scope: one clarifying sentence in README

### Summary of priorities

| Priority | Items |
|----------|-------|
| High     | R1 (import path), R4 (license), D1 (drifter model doc) |
| Medium   | R2 (what is a drogued drifter), R3 (drop internal state vector), R5 (Python/Parcels version), R6 (pixi link), N2 (explain get_final_drift_batch), N3 (color mismatch), N5 (Baltic pipeline overview) |
| Low      | D2 (Stokes doc), N1 (imports heading), N4 (orphaned line), N6 (hardcoded date), N7 (EE vs RK4 note) |
