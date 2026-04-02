# Agent guidelines for this project

## Multi-agent workflow

**Model choice:** Use a lighter model for mechanical and verification tasks. Reserve more capable models for architecture, design decisions, and judgment calls.

**Planning before code:** Write plans to `plans/*.md` before touching source. Don't skip planning for complex changes; don't let implementation agents make architectural decisions unguided.

**Red-Green-Blue TDD:** One agent writes failing tests, another implements minimally to make them pass, a third reviews quality. Use this pattern for feature work and refactors.

**Always review after implementation:** A separate review agent should examine the result. This catches both conceptual mistakes and quality issues.

**Experimental validation:** Use `tmp_*/` directories to prove ideas before committing to architecture changes. Once validated, clean up or move to permanent locations.

## Environment and build

Use **pixi** for environment management. Run all commands with `pixi run`. Prefer conda packages; use pypi only when no suitable conda package exists.

## Code and derivations

**Sympy is a derivation tool, not a display tool.** Let sympy derive results from the physics — don't hand-derive and then use sympy to typeset. Keep the full chain from Lagrangian to executable functions intact and reproducible.

**Be careful with generated or derived artifacts.** Some files in the repository may be cached outputs of expensive computations rather than hand-written source. Before editing them, check how they were produced and whether changing the source is the right fix instead.

Be ruthless about dropping dead code. Patch sparingly; rewrite when the abstraction is wrong.

## Notebooks

- Markdown cells for narrative; clean code cells for execution.
- Well-scoped cells — don't mix imports, parameters, and calculations.
- Use `display()` for sympy output.
- **Never write summary cells with prose that assumes results.** Summary cells must compute and print dynamically.
- Use xarray, pandas etc. _public_ API. Example: `ds.lon.isel(traj=0)` instead of `ds.lon.values[0, :]` etc.
- After fixing bugs, rerun immediately without asking.
- **Execute notebooks with papermill**: `cd <notebook-dir> && pixi run papermill <nb.ipynb> <nb.ipynb>` (in-place, cwd = notebook directory). Use `--execution-timeout 600` for notebooks with symbolic derivations. Do not use `jupyter nbconvert --execute`.
- Ensure all parameters are in a parameters cell close to the beginning of the notebook. The parameters cell neesd to be tagged "parameters" and only declare and assign primitives. All calculations, transformations, etc. of these parameters have to happen outside of the parameters cell.

## Plotting

- Keep plots vanilla. No custom colormaps, figsize, or axis labels when xarray handles them.
- Use builtin plotting instead of raw matplotlib if possible.
- Focus on the data. Use cartopy with Natural Earth or OSM tiles for maps.

## Data access

- Use `copernicusmarine.open_dataset` with minimal arguments.
- Subset with `.sel()`.
