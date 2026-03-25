# Agent guidelines for this project

## Environment

- Use **pixi** for environment management. Run all commands with `pixi run`.
- Prefer conda packages. Use pypi only when no good conda package is available.
- Never commit. Never push.

## Code and derivations

- **Sympy is a derivation tool, not a display tool.** Let sympy *derive*
  the result from the physics. Don't hand-derive and then use sympy to
  typeset. Keep the full chain: Lagrangian → sympy EOM → generated code.
- When changing the physics or the drifter model, the sympy derivation
  must remain the source of truth.
- Plans go in `plans/*.md` before significant implementation work.
- Be ruthless about dropping code. Don't be attached to implementations.
  If something should be reimplemented differently, drop it and start
  fresh rather than patching.

## Notebooks

- Use **papermill** for notebook execution.
- Markdown cells for narrative.
- Well-scoped, human-facing code cells. Don't mix imports, parameters,
  and calculations in one cell.
- Use `display()` for sympy output.

## Plotting

- Don't customize plots unnecessarily. No custom colormaps, figsize, or
  axis labels when xarray's built-in `.plot()` handles them.
- Go with vanilla defaults. Focus on the data, not the presentation.
- For maps: cartopy with 10m Natural Earth features, or OSM tiles.
  Use `ccrs.Geodetic()` for transform with tile-based projections.

## Data access

- Use `copernicusmarine.open_dataset` with minimal arguments. Don't
  pin dataset versions. Subset with `.sel()`.
- Linspace-regularize coordinates when matplotlib requires evenly
  spaced grids.

## Review

- Use opus agents for critical review of derivations and notebooks.
  Don't brief the reviewer on the expected result — let them evaluate
  purely from what's written.
