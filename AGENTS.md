# Agent guidelines for this project

## Principles

**This is pre-alpha research code.** No installed base, no backwards compat, no users to migrate. Internal API changes are free — changing signatures of private functions (`_rhs_batch`, `_make_qdd_func`, etc.) is expected, not exceptional. Even after release, anything prefixed with `_` is internal and can change between minor versions.

**Greenfield mindset.** If the current shape is in the way of the right shape, reshape it. Don't add workaround constraints when restructuring eliminates the problem. Deletions, renames, and rewrites are the normal mode.

**Be ruthless about dropping dead code.** Patch sparingly; rewrite when the abstraction is wrong. Prefer clean parameter plumbing over clever hacks — no monkey-patches, global state swaps, or closure tricks when passing a parameter through the call chain is cleaner.

**Be diligent about follow-through.** When you touch a name, signature, or path, grep for every reference and update them in the same pass. Don't leave stale imports, dead references, or half-updated docs for a later cleanup step.

**Sympy is a derivation tool, not a display tool.** Let sympy derive results from the physics — don't hand-derive and then use sympy to typeset. Keep the full chain from Lagrangian to executable functions intact and reproducible.

## Agent workflow

**Planning before code:** Write plans to `plans/*.md` before touching source. Don't skip planning for complex changes; don't let implementation agents make architectural decisions unguided.

**Model choice:** Use a lighter model for mechanical and verification tasks. Reserve more capable models for architecture, design decisions, and judgment calls.

**Red-Green-Blue TDD:** One agent writes failing tests, another implements minimally to make them pass, a third reviews quality. Two rounds: interface first, then behavior.

**Always review after implementation:** A separate review agent should examine the result. This catches both conceptual mistakes and quality issues.

**Experimental validation:** Use `tmp_*/` directories to prove ideas before committing to architecture changes. Once validated, clean up or move to permanent locations.

## Tooling

Use **pixi** for environment management. Run all commands with `pixi run`. Prefer conda packages; use pypi only when no suitable conda package exists.

## Conventions

### Code

**Be careful with generated or derived artifacts.** Some files in the repository may be cached outputs of expensive computations rather than hand-written source. Before editing them, check how they were produced and whether changing the source is the right fix instead.

### Notebooks

See the **jupytext skill** (`.agents/skills/jupytext/SKILL.md`) for the full workflow: creating, syncing, executing, and fixing notebooks.

- The `.md` is the source of truth. Execute with `pixi run jupytext --sync --execute <nb>.md`. Papermill only for parameter injection. Do not use `jupyter nbconvert --execute`.
- Markdown cells for narrative; clean code cells for execution.
- Well-scoped cells — don't mix imports, parameters, and calculations.
- Every notebook must have one early parameters cell tagged `"parameters"` containing only primitive assignments (`int`, `float`, `str`, `bool`, `None`). All calculations, transformations, and derived values belong in subsequent cells. This keeps notebooks papermill-compatible for parameter sweeps.
- Use `display()` for sympy output.
- **Never write summary cells with prose that assumes results.** Summary cells must compute and print dynamically.
- Use xarray, pandas etc. _public_ API. Example: `ds.lon.isel(traj=0)` instead of `ds.lon.values[0, :]` etc.
- After fixing bugs, rerun immediately without asking.

### Plotting

- Keep plots vanilla. No custom colormaps, figsize, or axis labels when xarray handles them.
- Use builtin plotting instead of raw matplotlib if possible.
- Focus on the data. Use cartopy with Natural Earth or OSM tiles for maps.

### Documentation

`docs/*.md` contains standalone documentation for the current state of the code. Each doc should make sense on its own without referencing previous implementations, changelogs, or development history. Explain design choices by comparing alternatives and their trade-offs, not by narrating what changed. Git history is the changelog; docs describe what *is*, not what *was*.

`plans/*.md` describe intent before implementation. When a plan is implemented: write a corresponding `docs/` file, move the plan to `plans/done/`, and add a one-liner at the top pointing to the doc. Plans have no frontmatter or structured metadata — `ROADMAP.md` and `BACKLOG.md` in `plans/` provide the index. Agents get context by reading `docs/*.md` (what is) + open `plans/*.md` (what's next).

Use markdown relative links when referencing other files in `plans/` and `docs/`. Example: `[parcels-v4-coupling.md](../docs/parcels-v4-coupling.md)` from a plan, `[BACKLOG.md](BACKLOG.md)` within `plans/`.

### Data access

- Use `copernicusmarine.open_dataset` with minimal arguments.
- Subset with `.sel()`.
