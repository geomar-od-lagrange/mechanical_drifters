# Roadmap: Drogued drifter simulations in the Baltic

## Track A: Parcels integration ✓

### A1. Polished idealized example ✓
- `examples/idealized_flow/02_sheared_jet_parcels.ipynb`

### A2. Point-particle runs in CMEMS data ✓
- `examples/baltic_drifters/02_cmems_point_particles.ipynb`

### A3. Drogued drifter kernel in CMEMS data ✓
- `examples/baltic_drifters/03_cmems_drogued_drifter.ipynb`
- Parcels v4 coupling via `DDAdvectEE` kernel — [docs/parcels-v4-coupling.md](../docs/parcels-v4-coupling.md)

## Track B: Building the right sheared current ✓

### B1. Wave and current analysis ✓
### B2. Wave orbital effects ✓
### B3. Stokes drift profile builder ✓

## Track C: Bringing it together

### C1. Clean drifter dataset ✓
### C2. Drifter simulations in effective currents ✓
### C3. Validation: deployment simulations ✓
### C6. Along-track velocity validation ✓

C4 (re-seeded validation) and C5 (parameter sensitivity) moved to
[BACKLOG.md](BACKLOG.md).

## Track D: Code quality ✓

All items complete. Plans in [done/](done/).

## Track E: Release wrap-up ✓

### E1. Optional numba backend ✓
### E2. README update ✓
### E3. Repo cleanup ✓
### E4. Rerun all example notebooks ✓
### E5. Full repo review ✓
### E6. Finalize README ✓
### E7. Switch jupytext pairing to md ✓

## Track F: Architecture refactor A ✓

Conservative rewire for v0.1.0 release. Plan:
[done/implement-A.md](done/implement-A.md).

1. Extract `coords.py` (coordinate transforms) ✓
2. Extract `velocity.py` (`make_profile_sampler`) ✓
3. Rename `lagrange_model.py` → `eom.py` ✓
4. Cache `_make_qdd_func`, delete `_qdd_func` ✓
5. Add `backend=` to public `qdd_func` ✓
6. Kill `_adapt_get_uv`, unify on `sample_uv` protocol ✓
7. Extract `_rhs`, `_rhs_batch`, `_z_eff` to module-level functions ✓
8. Slim `_extract_profiles` to take `drogue_depth` ✓
9. Export `compute_stokes_profile` from `__init__` ✓
10. New EOM exploration example notebook ✓

## Future: multi-object generalization

Architecture proposals for supporting additional Lagrangian mechanics
models (e.g. SparBuoy) alongside DroguedDrifter:

- [architecture-v2.md](architecture-v2.md) — B: function-first radical restructure
- [architecture-v3-multi-object.md](architecture-v3-multi-object.md) — C: ModelSpec dataclass
- [architecture-v4-class-based.md](architecture-v4-class-based.md) — D: LagrangianMechanicsModel base class
- [architecture-v5-simplified.md](architecture-v5-simplified.md) — D variant

Decision: D (class-based) is the preferred direction. Implement after
v0.1.0 release.
