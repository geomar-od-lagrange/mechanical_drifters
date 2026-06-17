# Consolidate the spar-buoy PRs (#21 + #22) and land on main

**Done.** Both PRs are consolidated onto `mf/add-spar-buoy` (#22 merged
into #21), reviewed, and the cleanup/restoration landed. The only
remaining act is the mechanical **squash-merge of #21 → main**. Spar-buoy
behaviour and the Parcels coupling are documented in
[docs/spar-buoy.md](../../docs/spar-buoy.md) and
[docs/parcels-v4-coupling.md](../../docs/parcels-v4-coupling.md).

## Situation

Two open PRs, cleanly stacked:

- **#21 "Add Spar-Buoy Model"** — `mf/add-spar-buoy` → `main`, draft.
  7 commits, all by **Merle Friederichsen**. Original model + the
  checklist fixes (syntax, state, derive Q from forces, drag in sympy).
  Two files, **both defining a class named `SparBuoyDrifter`**:
  `spar_buoy_simple.py` and the *separate* second implementation
  `spar_buoy_simple_reference.py`. The duplicate class name is what
  caused the silent EOM cache collision (cache keyed by class name).
- **#22 "Consolidate spar buoy into one averaged-drag model"** —
  `wr/spar-buoy-fixes` → `mf/add-spar-buoy`, ready.
  Stacked directly on #21 (which is a clean ancestor). Adds one more
  Merle commit ("Add spar buoy test") + the WR "Consolidate…" commit,
  which does **two distinct things**:
  1. **Deletes `spar_buoy_simple_reference.py`** — the separate
     spar-buoy-simple implementation. It is *born in #21 and killed in
     #22*, so it should never reach `main`.
  2. **Rewrites `spar_buoy_simple.py` in place** (124 ins / 139 del),
     renaming its class `SparBuoyDrifter` → `SparBuoySimple`; plus
     z-sign fix, docs/tests/example.

`#22` already contains all of `#21`. `wr/spar-buoy-fixes` merges to
`main` with **zero conflicts** and carries the whole stack (8 MF commits
+ 1 WR commit).

## Decisions (confirmed with WR)

- **Consolidate by merging #22 into #21**, then **squash-merge #21 to
  main** as the single final landing commit. Preserve the full commit
  history on the branch through consolidation; collapse to one commit
  only at the main boundary.
- **Attribution trade-off (considered and accepted):** because #22
  deletes Merle's reference file and rewrites the model in place, a
  squash collapses all 9 commits into one WR-authored commit — `git
  blame` on `spar_buoy_simple.py` will credit WR, not Merle. We accept
  this for a clean single-commit `main`; Merle is credited via a
  `Co-Authored-By` trailer and the closed-PR records keep her commits.
  Bonus of squashing: the `spar_buoy_simple_reference.py` create+delete
  cancels out, so it never appears on `main`.
- **`02_test_uv_profile.ipynb`: assess, then keep-and-jupytext or drop.**
  *Resolved → KEEP + jupytext* (after a wrong detour). I first dropped it
  on a flawed reading of the glue, then verified empirically that the
  notebook is correct: it places air at negative `depth`, and
  `parcels._extract_profiles` (`depth_up = -depth_levels[::-1]`) maps that
  onto the model's z-positive-up air sampling — so air→wind, water→current
  both resolve correctly. Restored, cleaned, paired, executed, and renamed
  to `02_parcels_wind_and_current`. This is the only example exercising the
  Parcels coupling for `SparBuoySimple`.
- **Sequential**, not single-PR: merge the two PRs into each other first,
  land #21 last.

## Steps

### 1. Cleanup pass on `wr/spar-buoy-fixes` (PR #22)

Do this on the WR branch so the work flows through #22 → #21 → main.

1a. **`examples/spar_buoy/02_test_uv_profile.ipynb` — KEPT, jupytexted,
    renamed `02_parcels_wind_and_current`.**
    Correction: an earlier pass dropped this notebook on the false premise
    that `parcels._extract_profiles` "only samples water and extrapolates
    into the air." Empirically (a 6-line check against the real glue) that
    is wrong: the notebook puts air at negative `depth` and water at
    positive `depth`, and `_extract_profiles` does `depth_up =
    -depth_levels[::-1]`, which maps the negative-depth air onto the
    model's positive z-up air levels. Sampling resolves correctly —
    air→wind (V=10), water→decaying current — and the run gives a sensible
    blended drift (u≈0.50 east, v≈4.8 north over 6 h).
    Action taken: restore from git, rewrite as a clean `.md` source
    (English, `parameters`-tagged primitives cell, dead debug cell
    removed), pair `md,ipynb`, execute clean, rename to
    `02_parcels_wind_and_current`. It is the only example exercising the
    Parcels coupling for `SparBuoySimple`.

1b. **Retire the implemented plans.** Move to `plans/done/` with a
    one-line pointer at the top to the doc that replaced them:
    - `plans/review-spar-buoy-pr.md` → `plans/done/`
    - `plans/spar-buoy-averaged-drag.md` → `plans/done/`
      (both point to `docs/spar-buoy.md`).
    ROADMAP/BACKLOG already reflect the shipped state — no index edits
    needed beyond confirming the moved files aren't referenced by a live
    relative link elsewhere in `plans/`.

1c. **Leave `eom_cache_spar_buoy_simple.pkl` as-is** — checked-in EOM
    cache is the established pattern (`..._drogued_drifter.pkl`,
    `..._point_surface_drifter.pkl` are already committed).

1d. **Verify.** `pixi run` the full test suite (expect 139 passed,
    5 deselected) and execute the `01_` and (if kept) `02_` example
    notebooks clean. Commit the cleanup on `wr/spar-buoy-fixes`.

### 2. Merge #22 into #21

Advance `mf/add-spar-buoy` to include the (now-cleaned) WR branch. Since
`#21` is an ancestor of `#22`, this is a fast-forward — all commits
preserved, no merge commit. PR #22 closes as merged. (On GitHub: set
#22's base = `mf/add-spar-buoy` and merge; or fast-forward the branch
locally and push.)

### 3. Ready #21 for main

- Update PR #21's base to `main` (already is) and **mark it ready for
  review** (currently draft).
- Refresh #21's description to describe the *consolidated* scope
  (model + z-sign fix + rename + docs/tests/example), superseding the
  original checklist body. Note that #22 was folded in.

### 4. Squash-merge #21 → main

Single squash commit on `main`. The net diff drops
`spar_buoy_simple_reference.py` entirely (created in #21, deleted in #22
→ absent from the squash) and lands only the final `spar_buoy_simple.py`
with `SparBuoySimple`. Author of the squash commit is WR; Merle is
credited via a trailer:

    Co-Authored-By: Merle Friederichsen <merlefriederichsen@users.noreply.github.com>

Per-commit MF authorship lives on in the #21/#22 closed-PR records, not
on `main` (accepted — see Decisions).

### 5. Post-merge

- Delete `wr/spar-buoy-fixes` and `mf/add-spar-buoy` (local + remote).
- `git checkout main && git pull`; sanity-run the test suite on main.
- Write/confirm `docs/spar-buoy.md` is the standing doc (it ships in the
  PR already).

## Open / watch

- The `02_parcels_wind_and_current` notebook is the working idealized
  demonstration of air+water Parcels forcing; the BACKLOG item is now
  scoped down to building such a fieldset from *real* ocean/atmosphere
  data.
- Pole-tilt dynamics remain deferred (already tracked in BACKLOG).
