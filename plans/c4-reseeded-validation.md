# C4: Re-seeded validation

Lagged re-initialization: every N hours, re-deploy virtual drifters
at the observed positions. Run each re-seeded ensemble forward for a
fixed window (24h, 48h). Compute separation distance as a function of
lead time, averaged over all re-seedings.

Reference: Liu & Weisberg (2011, doi:10.1029/2010JC006837).

Notebook 06 (`06_run_short_simulations.ipynb`) already implements
12h segments re-initialized every 12h. This plan extends it to
produce the skill-score analysis.

## Steps

1. Run 06 with current DD kernel (already done, output in
   `output/short_*.zarr`).
2. Match simulated trajectories back to observed positions using
   `output/short_releases.csv`.
3. Compute separation distance vs lead time for each re-seeding.
4. Plot mean separation as a function of lead time, comparing DD vs
   surface PP vs 3m PP.
5. Notebook 07 or 08 for the analysis.

## Depends on

- D-I (Parcels isolation) — done.
- Clean drifter dataset (C1) — done.
- Effective currents (B3) — done.
