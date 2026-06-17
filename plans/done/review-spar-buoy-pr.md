# Review: PR #21 "Add Spar-Buoy Model" (`mf/add-spar-buoy`)

**Addressed — see [docs/spar-buoy.md](../../docs/spar-buoy.md).** The review's
findings were implemented in PR #22 (consolidation to one `SparBuoySimple`
model, z-sign fix). Pole-tilt and signed-z wind forcing remain open in
[BACKLOG.md](../BACKLOG.md).

## Context

PR #21 (draft, branch `mf/add-spar-buoy`, tip `ac81a0d`) adds a spar-buoy drifter:
a partly-submerged float dragged by wind on its above-water column and by current
on its submerged hull. It ships **two** implementations of the same model plus a
notebook meant to prove they agree:

- `src/mechanical_drifters/models/spar_buoy_simple.py` — **version A**: per-level
  current samples are `State` fields (`U_air_0..2`, `U_water_0..3`, …) and the
  quadratic drag is summed *inside* sympy.
- `src/mechanical_drifters/models/spar_buoy_simple_reference.py` — **version B**:
  drag is computed numerically in `_rhs_batch`; only the aggregate `Fx_drag`,
  `Fy_drag` enter the symbolic EOM.
- `examples/spar_buoy/02_test_new_implementation.ipynb` — drives A and B to steady
  state and asserts equal trajectories.

This review was produced by a multi-agent workflow: four independent dimension
reviewers (physics, Parcels coupling, usability/API, documentation), an agent that
ran the code, and adversarial verification of every finding against the source and
against the **pinned Parcels SHA `17241585384f9cbb04a796cb3581cb49559df9df`**
(v3.1.3.dev2018, installed under `.pixi/`). 31 findings were raised and all 31
confirmed; several documentation findings were downgraded to LOW because
documenting a model that is currently broken is premature. Findings below are
deduplicated from the per-dimension reports.

**Key empirical results (reproduced):**
- Instantiating both classes in one process: the second one built **silently reuses
  the first's lambdified EOM** and then **crashes** (`TypeError: missing 12 required
  positional arguments`). The two `SparBuoyDrifter` classes cannot coexist.
- The notebook's A-vs-B assertion **passes for the wrong reason** — it compares A to
  A, and even a corrected comparison only passes because the test current is uniform.
  With a sheared current $U = 0.5 + 0.02\,z$, A and B diverge by **7.44 m** of along-track drift.
- Both implementations sample the *water* column at **positive z** (above the surface).

> Note on intent (read this first): `plans/ROADMAP.md:68` records that a previous
> SparBuoy was **dropped** — "a depth-average hack, not Lagrangian mechanics" — and
> `plans/BACKLOG.md:90-95` lists the *real* goal as "SparBuoy with real Lagrangian
> mechanics (tilt dynamics)". This PR reintroduces precisely the depth-averaged hack.
> The PR's own checklist confirms the real target is still open: "Later… Add tilt
> (azimuth, zenith) to `q`." **Before fixing line-level bugs, decide whether this
> simplified model is meant to ship at all, or is a scaffold toward the tilt model.**
> That decision changes which of the fixes below are worth doing.

---

## Blocking bugs (the PR does not work as written)

### B1 — Duplicate class name `SparBuoyDrifter` → EOM cache collision  ·  CRITICAL
`spar_buoy_simple.py:55` and `spar_buoy_simple_reference.py:42` both declare
`class SparBuoyDrifter`. The EOM machinery keys its caches solely on the class name:
- in-memory `_CALLABLE_CACHE` / `_QDD_CACHE` keyed by `type(model).__name__`
  (`eom.py:58,68,73`),
- on-disk cache path snake-cased from the class name → both map to
  `data/eom_cache_spar_buoy_drifter.pkl` (`base.py:80-82`).

So in one process the second model built reuses the first's lambdified callable
(`dB._qdd_func is dA._qdd_func` → `True`) and `integrate()` raises `TypeError`
(A expects 24 State args, B supplies 4). On disk the two `_derive_symbolic` source
hashes differ, so the shared pkl thrashes (re-derives) across runs.

**Fix:** Per AGENTS.md (greenfield, no backwards-compat), the clean fix is to keep a
single model and delete the other file — see D1. If two variants must coexist
temporarily, give them distinct class names so both cache keys diverge. Optionally
harden `eom.py` to key on the fully-qualified class (module + name) so future name
clashes fail loudly instead of silently aliasing. Do **not** patch around it with a
manual `_cache_path` override — that leaves the in-memory collision live.

### B2 — Notebook A-vs-B equivalence test is vacuous  ·  HIGH
`examples/spar_buoy/02_test_new_implementation.ipynb` (code cell In[7]) reads
`drifterB = SparBuoyDrifterA(physics_fast_B)` — it instantiates **A** for "B", so the
`assert_almost_equal` compares A to A and trivially passes. Fixing it to
`SparBuoyDrifterB(...)` immediately triggers the B1 crash; and once B1 is resolved,
the assertion still **fails on any non-uniform current** (see P4). The test currently
verifies nothing.

**Fix:** Use the real `B` class; resolve B1 first; and make the test exercise a
**sheared** current so equivalence is actually meaningful (a uniform field makes A and
B agree regardless of their differences). Also fix the notebook conventions (Doc5).

---

## Physics correctness

### P1 — Water drag sampled at POSITIVE z (above the surface)  ·  HIGH
`sample_uv(z)` takes z **positive-upward**, 0 = surface, **negative below MSL**
(`parcels.py:23-25,70-71`). Both files build water levels with positive z:
`np.linspace(0.0, self.physics.draft, n_water)` (`spar_buoy_simple.py:225`,
`spar_buoy_simple_reference.py:165`) → `[0 … +10] m`, i.e. 10 m *into the air*.
Reproduced against a sheared profile: the submerged hull samples linearly
*extrapolated super-surface* currents (increasing) instead of the real subsurface
shear (decreasing). Masked by the uniform-current notebook; only manifests under shear.

**Fix:** Sample water below the waterline, e.g.
`z_water_levels = np.linspace(-self.physics.draft, 0.0, int(self.physics.n_water))`.
Keep air levels positive. (Necessary but not sufficient under Parcels — see P3.)

### P2 — Air drag samples a wind field that does not exist  ·  CRITICAL (conceptual)
The Parcels coupling carries **ocean currents only** — `_extract_profiles` uses a
single `fieldset.UV` VectorField (`parcels.py:78,82,94`); there is no wind/air field
anywhere. Both models nonetheless sample "air" drag from that same sampler at positive
z above the surface (`spar_buoy_simple.py:212-218`, `_reference.py:141-147`), where
`_make_profile_sampler` linearly **extrapolates** past the top level (`parcels.py:42-49`).
So `k_air`-scaled "air drag" is really ocean current invented above the sea surface.

**Fix:** Decide the intended physics. If wind forcing is wanted, a separate wind
`VectorField` and air sampler must be plumbed through `_extract_profiles`/`make_kernel`
(the current single-UV coupling cannot supply it). Independently, harden
`_make_profile_sampler` to clamp z to `[depth_levels[0], depth_levels[-1]]` (or return
NaN) so out-of-range sampling fails loudly instead of silently extrapolating.

### P3 — No `_max_depth` → Parcels never samples below ~the 2nd grid level  ·  HIGH
`make_kernel` reads `max_depth = getattr(model, '_max_depth', 0.0)` (`parcels.py:142`);
only `DroguedDrifter` defines `_max_depth` (`drogued_drifter.py:416-419`). Neither
spar-buoy model does, so `_extract_profiles` keeps only the top two grid levels
(`parcels.py:82-86`). Under real Parcels forcing the 10 m draft is invisible — every
"water" depth interpolates from the two shallowest cells. (Inert in the notebook,
which uses an analytic `sample_uv`.)

**Fix:** Add `@property def _max_depth(self): return self.physics.draft` to the
surviving model, mirroring `DroguedDrifter`.

### P4 — A and B are not the same model  ·  HIGH
Even with B1/B2 fixed, A ≠ B: air levels differ (A `linspace(height_air/n_air,
height_air, n_air)` excludes the surface; B `linspace(0, height_air, n_air)` includes
it), level **counts** differ (defaults A = 3 air / 4 water, B = 10/10), and the
normalisation differs (A divides by literal `/3`, `/4`; B uses `np.mean`). For a
quadratic drag the mean-over-levels depends on count and placement, so the two diverge
for any depth-varying current (reproduced: 7.44 m drift difference; ~5% on a milder
shear).

**Fix:** One `SparBuoyPhysics` as the single source of truth; identical level
placement, counts, and normalisation in both formulations; pin them explicitly in the
equivalence test against a sheared current. (If D1 collapses to one model, this
dissolves.)

### P5 — Version A hardcodes 3 air / 4 water slots but exposes `n_air`/`n_water`  ·  MEDIUM
`spar_buoy_simple.py` declares fixed `State` fields `U_air_0..2` / `U_water_0..3`,
hardcoded symbol lists, and literal `/3`, `/4` divisors (`:132-133,144,152-153,164`),
while `_rhs_batch` samples `int(n_air)`/`int(n_water)` levels (`:212,225`) and packs the
first 3/4 (`:236`). Result: `n_air`/`n_water` are inert (the sympy symbols never appear
in any expression), larger values are silently dropped, and `n_air=2` raises
`IndexError`. The knobs lie about what they do.

**Fix:** Either drop `n_air`/`n_water` from version A and document the fixed 3/4
quadrature as intrinsic, or make A fully dynamic like B (variable-length State, symbol
list and divisor built from `n_air`/`n_water`). The latter is what version B already
does cleanly — another argument for D1.

---

## Design & usability

### D1 — Collapse the two near-duplicate models into one  ·  (root cause of B1, P4, P5)
Two files implementing the same physics under the same class name is the source of the
cache collision (B1), the impossible equivalence (P4), and the hardcoding (P5). AGENTS.md
is explicit: pre-alpha, no installed base, "Deletions, renames, and rewrites are the
normal mode." Pick one formulation — version B's aggregate-force approach handles
arbitrary level counts cleanly — delete the other file and the A-vs-B notebook, and
update every reference in the same pass. If a transient comparison is genuinely needed,
do it in one throwaway notebook with two distinctly-named classes, then delete it.

### D2 — Dead/unused sympy symbols and duplicated assignments  ·  MEDIUM
In B, `draft`, `height_air`, `n_air`, `n_water` are declared as sympy symbols
(`_reference.py:75-79`) but never used (B's EOM uses only `m`, `m_tilde`, `Fx_drag`,
`Fy_drag`); they are still threaded into `symbol_map`/`args` via `_fields`. Same in A
(`:88-92`), plus `q`/`r` are assigned twice (`:119-120`). These are sampling/integration
parameters consumed in `_rhs_batch`, not lambdify args.

**Fix:** Stop declaring/lambdifying the sampling parameters; build `args` only from the
load-bearing fields. Remove the duplicate `q`/`r` assignment.

### D3 — Committed EOM pickle is a build artifact  ·  MEDIUM
`src/mechanical_drifters/data/eom_cache_spar_buoy_drifter.pkl` is a generated cache
(see `caching.py`), currently untracked in the working tree. It should not be committed.

**Fix:** Add `data/*.pkl` to `.gitignore` (the `data/.gitkeep` already preserves the
dir) and remove the artifact. Check whether the sibling `eom_cache_*.pkl` files were
committed by mistake too.

---

## Documentation

The reviewers were unanimous that the *source-level* doc fixes (Doc1, Doc2) are worth
doing now, but the *repo-wide* doc additions (Doc3) should wait until the model is
consolidated and correct — documenting a broken, soon-to-be-renamed model just creates
more stale text to fix. Doc4 is a genuine conceptual conflict to resolve regardless.

### Doc1 — Stale class/field docstrings  ·  LOW
Both files: `SparBuoyPhysics` docstring says "draft=7, n_z=7" and the class docstring
says "7m draft" (`spar_buoy_simple.py:15,56`, `_reference.py:15,43`), but the default is
`draft=10.0` and there is **no `n_z` field** (the fields are `n_air`/`n_water`, which
differ per file: 3/4 vs 10/10). Fix to match reality after D1.

### Doc2 — Typos and mislabeled comments  ·  LOW
`_reference.py:187` `# totsal drag` → `# total drag`. The `n_air`/`n_water` comments say
"sampling rate" but the value is a level **count** passed as `np.linspace(..., num)`
(`:24-25` both files). The `k_air`/`k_water` "drag coefficient" comment is a lumped
quadratic factor $\approx \tfrac{1}{2}\rho\,C_d\,A$ [kg/m], not a dimensionless Cd. (The earlier-claimed
typo at `spar_buoy_simple.py:120` is **not** real — that line is fine.)

### Doc3 — Repo-wide docs omit the model — DEFER until after D1  ·  LOW
`README.md:5-13` ("The package includes:") and `docs/architecture.md:12` ("Two models
are included") list only Drogued/PointSurface; `docs/class-diagram.md` has no
`SparBuoyDrifter` node; there is no `docs/spar-buoy.md` though both siblings have a
per-model doc. Per AGENTS.md these *should* be updated when a model lands — but only
once there is a single, correct model to describe. Add the README bullet,
`architecture.md` entry, class-diagram node, and `docs/spar-buoy.md` in the same pass
that consolidates the model.

### Doc4 — ROADMAP/BACKLOG contradict the PR  ·  resolve explicitly
`plans/ROADMAP.md:68` marks SparBuoy **dropped** ("a depth-average hack, not Lagrangian
mechanics"); `plans/BACKLOG.md:90-95` keeps "SparBuoy with real Lagrangian mechanics
(tilt dynamics)" as open work. This PR is the depth-average hack. Either (a) update both
files to state that a simplified depth-averaged `SparBuoyDrifter` is now intentionally
included and explicitly distinguish it from the still-open tilt model, or (b) treat this
PR as a scaffold and keep it on a branch until the tilt dynamics (the PR's own open
checklist item) are in. This is the author's call and should be made before merge.

### Doc5 — Notebook violates project conventions  ·  MEDIUM
`02_test_new_implementation.ipynb` has **zero markdown cells** (AGENTS.md: "Markdown
cells for narrative") and **no `parameters`-tagged cell** — `t_end = 600.0` sits in an
untagged cell, breaking papermill compatibility (AGENTS.md notebook rules). Add a title
+ narrative, tag the parameters cell, fix the B2 instantiation bug, and rename to
something descriptive (e.g. `02_compare_drag_in_sympy_vs_numpy`). Source of truth is the
`.md`; sync/execute via the jupytext skill.

---

## Recommended remediation order

1. **Decide intent (Doc4):** ship the simplified model, or hold the PR for the tilt
   model. This gates everything else.
2. If shipping: **collapse to one model (D1)** — this alone removes B1, P4, P5 and the
   dual-file maintenance hazard.
3. Fix the physics on the surviving model: **water sign (P1)**, **`_max_depth` (P3)**,
   and resolve the **air-drag/wind question (P2)** (drop the air term or plumb a wind
   field; clamp the sampler against extrapolation).
4. Rewrite the notebook (B2 + Doc5) to a real, sheared-current check; clean dead
   symbols (D2); gitignore the cache artifact (D3); fix docstrings/comments (Doc1, Doc2).
5. Only then add repo-wide docs (Doc3) and update ROADMAP/BACKLOG (Doc4).

## Verification

- **Cache collision gone:** in one process, build both surviving + any sibling model,
  assert `model._qdd_func` objects are distinct and each `integrate()` runs without
  `TypeError`.
- **Water sign / `_max_depth`:** with a sheared analytic `sample_uv`, assert the
  submerged levels sample *decreasing-with-depth* (below-surface) values, not
  extrapolated super-surface ones; under a real Parcels `FieldSet`, assert
  `_extract_profiles` returns levels down to `draft`.
- **Equivalence test (if two formulations are kept transiently):** assert agreement
  under a **non-uniform** current, not just uniform.
- **Notebook:** `pixi run jupytext --sync --execute examples/spar_buoy/02_*.md` runs
  clean with a `parameters`-tagged cell.
- **Docs:** `grep -rin spar README.md docs/` reflects the final single model; ROADMAP/
  BACKLOG no longer contradict the shipped state.
