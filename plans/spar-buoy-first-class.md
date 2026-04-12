# Spar buoy as first-class citizen

Elevate the SparBuoy model from "also implemented" to equal standing with
DroguedDrifter. Add a simple example notebook. Review naming to ensure
the package presents itself as a multi-model toolkit, not a
drogued-drifter library with a spar buoy bolted on.

## 1. Example notebook

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

## 2. Naming review

### Package name: keep `drogued_drifters`

Renaming the package (`drogued_drifters` → `lagrangian_drifters` or
`mechanical_drifters`) would break every import in every notebook,
test, doc, and downstream user. The package started as a drogued
drifter model and the name is established. Python packages routinely
outgrow their names (e.g. `requests` does more than HTTP requests,
`pandas` isn't about pandas). The cost of renaming exceeds the benefit.

**Decision: keep the package name. Add a one-liner to the README
explaining it now supports multiple drifter types.**

### README title

Current: `# 2025 Drogued Drifters`

Change to something that signals multi-model scope without renaming:
`# Mechanical Ocean Drifters` or similar. The "2025" is a project year,
not a version — drop it. The subtitle should mention both models.

### Docs

- `docs/drifter-model.md` — this doc is specifically about the
  DroguedDrifter physics. Keep it as-is (it's already correctly scoped
  to one model). Add a note at the top that the SparBuoy is a separate,
  simpler model.
- `docs/parcels-v4-coupling.md` — already generic (`make_kernel` works
  for any model). Add a sentence noting SparBuoy works with
  `make_kernel(sb)` too.
- Consider a short `docs/spar-buoy.md` if there's anything worth
  explaining beyond "it averages velocity over its length". Probably
  not — the docstring in `spar_buoy.py` is sufficient.

### Source code

- `base.py`: replace the "See DroguedDrifter for a complete example"
  with "See DroguedDrifter and SparBuoy for examples."
- `__init__.py`: SparBuoy is already exported. Good.
- `parcels.py`: `make_dd_kernel` backward-compat alias stays. The
  generic entry point is `make_kernel`. Add a `make_sb_kernel` alias
  for symmetry? Probably not — `make_kernel(sb)` is clear enough.
  `make_dd_kernel` only exists because the old API used it by name.

### Example notebooks

Current idealized flow notebooks are all DroguedDrifter-titled:
- `01_synthetic_flow_profiles` — "Drogued Drifter in Synthetic Flow"
- `02_sheared_jet_parcels` — "Drogued Drifter in an Idealized Sheared Flow"
- `03_drogued_drifter_in_wave_orbitals` — drogued drifter specific

These are correctly DroguedDrifter-specific. No rename needed — they
demonstrate that specific model. The new `04_spar_buoy_in_sheared_flow`
demonstrates the other.

### README examples section

Add the spar buoy notebook to the idealized flow list. Add a brief
introductory sentence explaining the two model types.

## 3. Checklist

- [ ] Write `04_spar_buoy_in_sheared_flow.md` notebook
- [ ] Sync and execute with jupytext
- [ ] Update README title and subtitle
- [ ] Update README examples section (add spar buoy notebook)
- [ ] Update `base.py` docstring (mention both models)
- [ ] Update `docs/parcels-v4-coupling.md` (note SparBuoy compatibility)
- [ ] Run tests to confirm nothing broke
- [ ] Update PR #15 checklist
