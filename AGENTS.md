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

**This is pre-alpha research code. Internal API changes are free.** Changing signatures of private functions (`_rhs_batch`, `_make_qdd_func`, etc.) is not only acceptable but expected — do not work around internal interfaces with monkey-patches, global state swaps, or closure tricks when passing a parameter through the call chain is cleaner. Even after release, anything prefixed with `_` is internal and can change between minor versions. Prefer clean parameter plumbing over clever hacks.

## Notebooks

- Markdown cells for narrative; clean code cells for execution.
- Well-scoped cells — don't mix imports, parameters, and calculations.
- Use `display()` for sympy output.
- **Never write summary cells with prose that assumes results.** Summary cells must compute and print dynamically.
- Use xarray, pandas etc. _public_ API. Example: `ds.lon.isel(traj=0)` instead of `ds.lon.values[0, :]` etc.
- After fixing bugs, rerun immediately without asking.
- Notebooks are paired `.md` + `.ipynb` via jupytext. The `.md` is the source of truth — always edit the `.md`, never the `.ipynb` directly.
- **Execute notebooks with jupytext**: `cd <notebook-dir> && pixi run jupytext --sync --execute <nb>.md`. This syncs .md → .ipynb, executes, and saves outputs. This is the default for all notebooks.
- **Papermill is only needed when injecting parameter overrides** (e.g. running the same experiment for a range of different parameter sets). Don't reach for papermill by default.
- Do not use `jupyter nbconvert --execute`.
- Every notebook must have one early parameters cell tagged `"parameters"` containing only primitive assignments. All calculations and transformations of those parameters happen in subsequent cells. This keeps notebooks papermill-compatible if parameter sweeps are needed later.

## Plotting

- Keep plots vanilla. No custom colormaps, figsize, or axis labels when xarray handles them.
- Use builtin plotting instead of raw matplotlib if possible.
- Focus on the data. Use cartopy with Natural Earth or OSM tiles for maps.

## Documentation

`docs/*.md` contains standalone documentation for the current state of the code. Each doc should make sense on its own without referencing previous implementations, changelogs, or development history. Explain design choices by comparing alternatives and their trade-offs, not by narrating what changed. Git history is the changelog; docs describe what *is*, not what *was*.

`plans/*.md` describe intent before implementation. When a plan is implemented: write a corresponding `docs/` file, move the plan to `plans/done/`, and add a one-liner at the top pointing to the doc. Plans have no frontmatter or structured metadata — roadmap files in `plans/` provide the index. Agents get context by reading `docs/*.md` (what is) + open `plans/*.md` (what's next).

Use markdown relative links when referencing other files in `plans/` and `docs/`. Example: `[parcels-v4-coupling.md](../docs/parcels-v4-coupling.md)` from a plan, `[backlog.md](backlog.md)` within `plans/`.

## Data access

- Use `copernicusmarine.open_dataset` with minimal arguments.
- Subset with `.sel()`.
