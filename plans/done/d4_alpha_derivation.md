# D4: Drop the α formula and clean up notebooks 10 and 12

## Decision

Drop the α = √k_b/(√k_b+√k_d) shortcut from the codebase and
notebooks. The full ODE (via Parcels + DroguedDrifter kernel) is the
proper prediction method. The α-weighted velocity was a useful
exploration tool but is not rigorous enough for the paper — it assumes
a single velocity at the buoy and a single velocity at the drogue,
which breaks down for atypical velocity profiles (sharp shear layers,
Ekman spiral, near-surface jets).

## What to change

### Notebook 10: along-track velocity validation

Currently compares three prediction types:
1. Surface effective current
2. 3m effective current
3. Alpha-weighted: `(1-α) * U_eff(3m) + α * U_eff(surface)`

**Change:** Remove the alpha-weighted prediction. Keep surface and 3m
as honest baselines — "what does the CMEMS current look like at these
depths vs what the drifter actually did?"

Specific edits:
- Remove `K_B` and `K_D` from the parameters cell
- Remove the alpha computation and `u_pred_alpha`, `v_pred_alpha`,
  `speed_alpha` columns
- Update `pred_configs` to only have Surface and 3m
- Update all plots and tables (velocity time series, scatter, RMSE,
  direction validation) to show 2 types instead of 3
- Update markdown narrative to remove alpha references

The notebook becomes a pure field-vs-observation comparison, which is
its proper role.

### Notebook 12: parameter sensitivity

Currently has two parts:
1. **1D α-sweep** (cheap, numpy-only): Sweeps k_d, computes α, evaluates
   α-weighted RMSE. This is entirely based on the α formula.
2. **2D ODE grid search** (k_d × m̃_d): Runs the actual DroguedDrifter ODE
   at each parameter combination. This is the legitimate sensitivity
   analysis.

**Change:** Remove the 1D α-sweep and α-space sections entirely. Keep
and expand the 2D ODE grid search as the sole sensitivity analysis.

Specific edits:
- Remove the 1D k_d sweep cells (sweep, RMSE vs k_d plot, per-drifter
  RMSE vs k_d)
- Remove the α-space sweep cells
- Remove the correlation sensitivity cell
- Remove the "physical interpretation" cell mapping α back to C_D,d
- Remove the "optimal k_d per drifter" table (based on α)
- Keep the 2D grid search (k_d × m̃_d via ODE integration)
- Keep the RMSE heatmap
- Optionally add a 1D k_d slice from the 2D grid (at default m̃_d) to
  show the k_d sensitivity without α

### Plans

- Delete or archive `plans/analytical_steady_state.md` and
  `plans/fsolve_steady_state.md` if they are purely about the α
  approach. If they contain useful ideas about precomputed drift
  fields, keep those parts.

### Source code

- The α formula is not in `drifter.py` — it was only used in notebooks.
  No source code changes needed.
- If α appears in any docstrings or comments in `drifter.py`, remove
  those references.

### Manuscript

- Do not derive α in the paper. The steady-state analysis (if included)
  should present the full force balance, not the shortcut formula.
- The α formula can be mentioned as "in the limit of uniform currents
  within buoy and drogue layers, the force balance simplifies to..."
  but should not be the primary result.

## What we lose

- The cheap 1D sensitivity sweep (no ODE runs needed). This was
  convenient but misleading — it only captured the steady-state drag
  balance, not the transient dynamics or the actual velocity profile
  sampling.
- A simple mental model ("the drifter drifts at α-weighted velocity").
  This is replaced by the more accurate statement: "the drifter model
  solves for the steady-state tilt and drift given the full velocity
  profile."

## What we gain

- No shortcut results that could be wrong for atypical profiles
- Cleaner notebooks focused on what we actually validate (field
  quality at specific depths, full ODE sensitivity)
- The 2D ODE grid search is the honest sensitivity analysis
