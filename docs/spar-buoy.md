# Spar buoy

`SparBuoySimple` models a vertical spar buoy that pierces the ocean surface: a
rigid pole with a submerged hull of length `draft` below the waterline and an air
column of height `height_air` above. Like `PointSurfaceDrifter` it has two
generalized coordinates `(x, y)` and a purely kinetic Lagrangian
$L = \tfrac{1}{2}(m + \tilde{m})(\dot{x}^2 + \dot{y}^2)$, but it is driven by drag from *two* media:
wind on the air column and current on the submerged hull.

## Drag

Drag is evaluated at a fixed set of points along the pole,
$r_i = r_b + z_i\,\hat{p}$ (with $r_b = (x, y, 0)$ at the surface and the pole
direction $\hat{p} = (0, 0, 1)$ vertical, i.e. `pole_hat`): `n_air` points spread
through the air column and `n_water` through the hull. Each point feels quadratic
drag $F_i = -\tfrac{k}{n}\,\lvert v_i - u_i\rvert\,(v_i - u_i)$, where $u_i$ is
the ambient flow there and $k$ is `k_air` or `k_water`. Dividing by the level
count `n` makes the total a *mean* over levels, i.e. the depth-averaged drag of
each medium. With equal `k_air`/`k_water` in a 1.0 m/s current and calm air the
buoy drifts at 0.5 m/s — the mean of the two media.

## Why the drag is assembled symbolically

The averaged drag is built inside `_derive_symbolic` as a sum of per-level
generalized forces — the same construction `DroguedDrifter` uses to sum its buoy
and drogue drag (see [drifter-model.md](drifter-model.md)):

$$
Q = \sum_i \frac{\partial r_i}{\partial q} \cdot F_i
$$

For the vertical pole $\partial r_i/\partial q$ is the identity on `(x, y)`, so the level height
`z_i` cancels and `Q` collapses to the mean horizontal drag. The height is carried
through the derivation anyway (sympy is given `z_i` even though it cancels) so the
*structure* is tilt-ready: once tilt coordinates enter `q` and
`pole_hat = pole_hat(tilt)`, `z_i` re-enters through `r_i` and `Q` automatically
gains the drag torques about the tilt axes — only `r_i` and `pole_hat` change.

Averaging the levels numerically in the right-hand side instead, feeding only an
aggregate `(Fx, Fy)` into the EOM, gives identical numbers for the vertical pole
but cannot carry over to tilt: a pre-averaged horizontal force carries no
information about where along the pole it acts, so it can never produce a torque.
Pole-tilt (azimuth/zenith) is deferred; keeping the averaging symbolic is what
makes adding it a local change rather than a rewrite.

## Sampling convention

`sample_uv(z)` takes `z` **positive upward**: $z = 0$ is the surface, $z > 0$ is in
the air, $z < 0$ is below the waterline. The air levels sit at positive `z` spread
through the air column, the water levels at $z \le 0$ down to the hull tip at
$-\text{draft}$. The same convention holds for the Parcels-derived sampler and the
analytic `sample_uv` in the idealized notebook.

## Level counts

`n_air` and `n_water` (default 3 / 4) are `Physics` fields, but they are fixed by
the `State` layout: `SparBuoyState` carries one current per level (`U_air_0..2`,
`U_water_0..3`), so a lambdified EOM with its fixed argument signature is wired for
exactly those counts. Constructing `SparBuoySimple` with any other counts raises —
no silent dropping of levels. (Making the count freely configurable would mean
generating the `State` and the symbolic level list dynamically; that is deferred
with the tilt work.)

## Parcels coupling: wind and current on one signed-depth axis

`_max_depth` returns `draft`, so the Parcels coupling samples down to the hull tip
(see [parcels-v4-coupling.md](parcels-v4-coupling.md)). Wind forcing works today,
provided the fieldset supplies both media: put the **air column at negative
`depth`** and the **water column at positive `depth`** on one `depth` axis. The
coupling's profile extractor flips depth to the model's $z$-positive-up frame
(`depth_up = -depth[::-1]`), so air lands at $z > 0$ and water at $z \le 0$ exactly
where the model samples them — air drag reads the wind, hull drag reads the
current. The [`02_parcels_wind_and_current`](../examples/spar_buoy/02_parcels_wind_and_current.ipynb)
example runs this end to end (10 m/s wind over a depth-decaying current, giving a
drag-weighted blended drift).

A plain ocean-only fieldset (water at $z \le 0$, nothing above) leaves the air
levels with no distinct field to sample, so they extrapolate the near-surface
current — fine when wind is irrelevant, but not real wind forcing. Driving the air
column from *real* wind data (e.g. ERA5 merged with a Copernicus ocean field on a
common signed-depth grid) is future work tracked in [BACKLOG.md](../plans/BACKLOG.md).
