# Spar buoy as first-class citizen

Elevate the SparBuoy model to equal standing with DroguedDrifter. Rename
the package from `drogued_drifters` to `mechanical_drifters` so the name
reflects a multi-model toolkit. Add a simple example notebook.

## 1. Package rename: `drogued_drifters` → `mechanical_drifters`

This is pre-alpha research code with no downstream users. Internal API
changes are free (CLAUDE.md L27). A mechanical rename now is cheaper
than carrying a misleading name forever.

### What changes

| What | From | To |
|------|------|----|
| Package dir | `src/drogued_drifters/` | `src/mechanical_drifters/` |
| pyproject.toml name | `drogued_drifters` | `mechanical_drifters` |
| pyproject.toml description | "...drogued ocean drifters" | "...mechanical ocean drifters" |
| All `from drogued_drifters import` | everywhere | `from mechanical_drifters import` |
| All `import drogued_drifters` | everywhere | `import mechanical_drifters` |
| README title | "2025 Drogued Drifters" | "Mechanical Ocean Drifters" |

Files affected: `pyproject.toml`, `__init__.py`, every test file, every
notebook `.md`, every doc, `AGENTS.md`, `base.py` docstring.

### What does NOT change

- Class names: `DroguedDrifter`, `SparBuoy` stay as-is (they name
  specific models, not the package).
- `DrifterPhysics`, `EOMState` stay (they're DroguedDrifter-specific
  types, correctly named).
- `make_dd_kernel` stays as a backward-compat alias.
- The git repo directory name `2025_drogued_drifters/` — that's an
  external concern (GitHub rename), not part of this PR.
- `eom_cache_drogued_drifter.pkl` — that's auto-derived from the class
  name, not the package name.

### Approach

One `git mv` + project-wide find-and-replace. Sync all notebooks after.

## 2. Example notebook

`examples/idealized_flow/04_spar_buoy_in_sheared_flow.md`

Simple demonstration: depth-averaged drifter in a vertically sheared
flow. The notebook should:

- Define an exponentially decaying velocity profile (same style as
  `01_synthetic_flow_profiles`).
- Create a `SparBuoy` with a few different lengths (e.g. 5 m, 15 m,
  30 m).
- Show that a longer buoy averages over more of the water column and
  therefore drifts slower in a surface-intensified flow.
- Compare SparBuoy drift to DroguedDrifter drift in the same profile,
  showing the qualitative difference (depth-averaged vs drag-weighted
  equilibrium).
- Keep it short — one flow, one sweep, one comparison plot.

No Parcels, no Stokes drift, no wave orbitals. Pure standalone API.

## 3. README and docs

- README title: "Mechanical Ocean Drifters"
- README subtitle: mention both models and the base class.
- README quick-start: show both models briefly.
- README examples: add the spar buoy notebook.
- `docs/drifter-model.md`: keep as DroguedDrifter-specific (it is).
  Add a note at the top that the package also includes SparBuoy.
- `docs/parcels-v4-coupling.md`: note that `make_kernel(model)` works
  for any model including SparBuoy.
- `base.py` docstring: mention both models as examples.

## 4. Checklist

- [ ] `git mv src/drogued_drifters src/mechanical_drifters`
- [ ] Update `pyproject.toml` (name, description, test command)
- [ ] Find-and-replace `drogued_drifters` → `mechanical_drifters` in
      all `.py`, `.md` files (imports, docs, notebooks)
- [ ] Update README title, subtitle, quick-start, examples
- [ ] Update `base.py` docstring
- [ ] Update `docs/parcels-v4-coupling.md`
- [ ] Write `04_spar_buoy_in_sheared_flow.md` notebook
- [ ] Sync and execute all notebooks with jupytext
- [ ] Run full test suite
- [ ] Update PR #15 checklist
