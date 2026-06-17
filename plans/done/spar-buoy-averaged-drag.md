# Spar-buoy: consolidate to one averaged-drag model

**Implemented — see [docs/spar-buoy.md](../docs/spar-buoy.md).** The final design
followed the "MF structure + tilt-ready Q" route: `spar_buoy_simple.py` is edited in
place (class `SparBuoySimple`), keeping MF's explicit `SparBuoyState` and the
`n_air`/`n_water` `Physics` fields, with the drag assembled as
$Q = \sum_i (\partial r_i/\partial q)\cdot F_i$. The dynamic-`State`, midpoint
sampling, and `_cache_path`-encoding ideas sketched below were **dropped** as
over-engineering.

Intent doc. Findings being addressed: [review-spar-buoy-pr.md](review-spar-buoy-pr.md).
Scope decision (Willi): ship the **depth-averaged-drag** spar buoy now; **defer tilt**
(azimuth/zenith) to later. Keep both air and water drag.

## Why

PR #21 shipped two same-named `SparBuoySimple` classes (an in-sympy-drag version A
and a numeric-drag version B) plus a vacuous A-vs-B notebook. The duplication causes a
silent EOM-cache collision (both keyed by class name), the equivalence test never ran B,
and the water column is sampled at the wrong z-sign. We collapse to **one** model and
fix the physics.

## Design decisions

**One model, drag derived in sympy (not in the rhs).** Per Willi: the per-level drag
and the generalized-force assembly must live in `_derive_symbolic`, so adding tilt later
is just extending `q` and the per-level positions `r_i` — exactly the `DroguedDrifter`
pattern (`models/drogued_drifter.py`, which sums two bodies' drag via
$Q = \sum_i (\partial r_i/\partial q)\cdot F_i$). If the rhs did the aggregation, none of it would carry over to
the tilting buoy. So:
- `_derive_symbolic` builds, in sympy, a sum over `n_air + n_water` drag points along the
  pole. Each point `i` has position $r_i = r_b + z_i\,\hat{e}$ ($\hat{e} = (0,0,1)$ vertical pole
  for now; later $\hat{e} = \hat{e}(\text{tilt})$), relative velocity $v_i - u_i$, and quadratic drag
  $F_i = -(k/n)\,|v_i - u_i|\,(v_i - u_i)$. Generalized force $Q = \sum_i (\partial r_i/\partial q)\cdot F_i$.
  For the vertical pole, $\partial r_i/\partial q$ is the identity on $(x,y)$ so `z_i` cancels and `Q`
  reduces to the mean horizontal drag — but the *structure* is the tilt-ready one.
- Mean (not sum) drag: per-level coefficient `k_air/n_air`, `k_water/n_water`.

**Sampling sign (verified, not just documented).** `sample_uv(z)` is **z-positive-up**
on both the Parcels-derived path (`parcels.py:_make_profile_sampler`/`_extract_profiles`,
which convert grid depths via `depth_up = -depth_levels[::-1]`) and the idealized
notebooks (`np.where(z <= 0, water, air)`). So in `_rhs_batch`:
- air levels at **positive** z (already correct),
- water levels at **negative** z — `np.linspace(-draft, 0, n_water)` (the fix; current
  code uses `+draft`). Confirmed empirically: a 10 m buoy in 1.0 m/s water + calm air
  drifts 0.27 as-is vs the correct 0.50 (= analytic steady state) once water is `−z`.
- Use midpoint placement to avoid double-counting the surface and zero-z edge cases:
  water `-draft*(i+0.5)/n_water`, air `height_air*(i+0.5)/n_air`.

**Level counts are structural, not runtime knobs.** A fixed `lambdify` signature can't
take a variable number of currents, so `n_air`/`n_water` are **class attributes**
(default e.g. 3 / 4), not `Physics` fields — this removes version A's inert-`n_air`
symbols and the `IndexError`/silent-drop bugs. The `State` NamedTuple's per-level current
fields (`U_air_i`, `V_air_i`, `U_water_i`, `V_water_i`) are generated from those counts
via the functional NamedTuple API, so there is no hand-written `U_air_0..2` to drift.

**Cache safety.** `caching._cache_key` hashes `_derive_symbolic` source only, so changing
`n_air`/`n_water` would otherwise load a stale pkl. Override `_cache_path` on the model to
encode the counts: `data/eom_cache_spar_buoy_simple_a{n_air}_w{n_water}.pkl` (the base
class documents this override point). Commit the regenerated pkl — sibling caches
(`eom_cache_drogued_drifter.pkl`, `..._point_surface_drifter.pkl`) **are** tracked, so we
match that convention rather than gitignoring (corrects review finding D3).

**`Physics`** = `m, m_tilde, k_air, k_water, draft, height_air`. `draft`/`height_air` are
sampling/geometry params (used in `_rhs_batch` and `_max_depth`); they appear in the
lambdify `args` because the generic `pack_eom_args` passes every `Physics` field — that
is the base-class contract, not dead code.

**`_max_depth = draft`** (property, mirrors `DroguedDrifter`) so the Parcels coupling
samples the water column. The **air** side has no super-surface coupling yet: under the
current `parcels.py`, `_extract_profiles` only samples grid depths (water). Real wind
forcing needs a signed-z / wind fieldset (Willi: "Parcels v4 … z-levels with a sign … a
fieldset of water velocities for $z \in [\ldots,0]$ and air velocities in $z \in (0,\ldots]$"). That is a
coupling+fieldset change → **out of scope here**, recorded in BACKLOG. The idealized
notebooks exercise full air+water with analytic `sample_uv`, so the model is correct by
contract today.

## Files

Model (keystone — authored/closely reviewed by me):
- **add** `src/mechanical_drifters/models/spar_buoy_simple.py` — `SparBuoySimple`,
  `SparBuoyPhysics`, dynamically-built `SparBuoyState`. Mirror the skeleton of
  `point_surface_drifter.py` + `drogued_drifter.py` (same `_derive_symbolic` →
  `linear_eq_to_matrix`, static-symbol subs, `symbol_map`/`args`; same `_rhs_batch`
  NaN-guard and `dY` packing; `drift_velocity`; `_max_depth`).
- **delete** `src/mechanical_drifters/models/spar_buoy_simple.py` and
  `…/spar_buoy_simple_reference.py`.
- **remove** stale `src/mechanical_drifters/data/eom_cache_spar_buoy_simple.pkl`;
  **regenerate + commit** `…_a{n_air}_w{n_water}.pkl`.

Tests (agent, mirrors `tests/test_point_surface_drifter.py`):
- **add** `tests/test_spar_buoy.py` — class/subclass, physics fields, state-field count
  = `2 + 2(n_air+n_water)`, `n_q==2`, `drift_velocity`, `state_size`, `_derive_symbolic`
  shapes, qdd drag direction, zero-relative-velocity → zero accel, `_max_depth==draft`,
  and a **signed-z** behaviour test: water-only current ⇒ submerged drift tracks water,
  and the 1.0-water/calm-air case ⇒ drift ≈ 0.5 (guards the sign regression). Uniform-flow
  steady state. No cache-collision (single class).

Notebook (agent + jupytext skill):
- **adopt + clean** `examples/spar_buoy/01_spar_buoy_idealized.ipynb` (currently untracked,
  already air+water with signed z, tagged `parameters`, markdown): repoint import to
  `mechanical_drifters.models.spar_buoy`, verify it runs, pair `.md` via jupytext, add to git.
- **delete** tracked `examples/spar_buoy/02_test_new_implementation.ipynb` (A-vs-B scaffold).

Docs (agent):
- `README.md` — add a `SparBuoySimple` bullet (after PointSurfaceDrifter) and an
  `examples/spar_buoy` entry under Examples.
- `docs/architecture.md` — "Two models" → three; add bullet + module-layout entry for
  `spar_buoy.py`.
- `docs/class-diagram.md` — add `SparBuoySimple` node + `LagrangianMechanicsModel <|--`
  edge.
- **add** `docs/spar-buoy.md` — standalone "what is": geometry (pole, draft, air column),
  depth-averaged quadratic drag for air+water, sympy $Q = \sum (\partial r_i/\partial q)\cdot F_i$ assembly and
  why it is structured for tilt, the z-positive-up sampling convention, `_max_depth`.
- `plans/ROADMAP.md:68` and `plans/BACKLOG.md:90-95` — reconcile: the averaged-drag
  `SparBuoySimple` is now intentionally included; keep **tilt dynamics** and a
  **signed-z / wind fieldset Parcels coupling** as distinct open items.

## Verification

- `pixi run pytest -v tests/test_spar_buoy.py` (and full suite green).
- `pixi run jupytext --sync --execute examples/spar_buoy/01_spar_buoy_idealized.md` runs clean.
- Sign regression: 1.0 m/s water + calm air ⇒ drift ≈ 0.5 (not 0.27).
- One process imports `SparBuoySimple` + a sibling; each `integrate()` runs, distinct
  `_qdd_func` / `_cache_path` (no collision).
- `grep -rin spar README.md docs/ plans/ROADMAP.md plans/BACKLOG.md` reflects the single model.
