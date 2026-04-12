# Stokes drift profile

`drogued_drifters.stokes.compute_stokes_profile` computes a depth-dependent
Stokes drift profile from surface Stokes drift components and a peak wave
period.  It implements the deep-water monochromatic exponential decay model
described in Liu et al. (2021) and Breivik et al. (2016).

## Physics

Under deep-water dispersion the angular frequency `ω = 2π/T_p` and
wavenumber `k` are related by

```
k = ω² / g
```

The Stokes drift at depth `z` (z-up convention, `z ≤ 0`) decays as

```
u_S(z) = u_S(0) · exp(2kz)
```

Because `z ≤ 0`, the exponent `2kz = -2k|z|` is always non-positive and the
profile decays monotonically from the surface value toward zero.

The surface components `(u_S(0), v_S(0))` and the peak period `T_p` are the
only inputs required; no spectral wave data is needed.  This makes the model
compatible with standard Copernicus Marine (CMEMS) wave products, which
provide surface Stokes drift and peak period as gridded fields.

References:

- Liu, Q., et al. (2021). Bulk, Spectral and Deep Water Approximations for
  Stokes Drift: Implications for Coupled Ocean Circulation and Surface Wave
  Models. *Journal of Advances in Modeling Earth Systems*, 13, e2020MS002172.
  https://doi.org/10.1029/2020MS002172
- Breivik, O., Bidlot, J.-R., & Janssen, P. A. E. M. (2016). A Stokes drift
  approximation based on the Phillips spectrum. *Ocean Modelling*, 100, 49–56.
  https://doi.org/10.1016/j.ocemod.2016.01.005

## Function signature

```python
def compute_stokes_profile(surface_u, surface_v, peak_period, depth_levels, g=None):
```

**Parameters**

| Parameter | Description |
|---|---|
| `surface_u` | Surface Stokes drift eastward component, any shape `(...)`. |
| `surface_v` | Surface Stokes drift northward component, same shape. |
| `peak_period` | Peak wave period [s], same shape as `surface_u`. |
| `depth_levels` | Vertical positions [m], positive upward (0 = surface, negative = below MSL), shape `(D,)`. Must be sorted ascending (deepest first, e.g. `[-20, -10, -5, 0]`). |
| `g` | Gravitational acceleration [m/s²]. Defaults to 9.81. |

**Returns**

Tuple `(stokes_u, stokes_v)` of arrays with shape `(D, ...)`, giving the
east and north Stokes drift components at each depth level.

## Multiple wave partitions

For sea states with multiple wave partitions, call `compute_stokes_profile`
once per partition and sum the results:

```python
u_total = np.zeros((len(depth_levels), *surface_u_p1.shape))
v_total = np.zeros_like(u_total)
for u_s, v_s, T_p in partitions:
    du, dv = compute_stokes_profile(u_s, v_s, T_p, depth_levels)
    u_total += du
    v_total += dv
```

## Where it is used

**Baltic pipeline — notebook 02** (`02_derive_effective_currents`): combines
CMEMS Eulerian current fields with Stokes drift profiles computed from CMEMS
wave data to produce effective current fields at each model depth level.
These fields are the direct input to the drogued-drifter simulations in
notebook 04.

**Idealized flow — notebook 03** (`03_drogued_drifter_in_wave_orbitals`):
constructs a velocity field from wave orbital velocities to demonstrate how
the drogued drifter responds to wave-induced motion at different depths.

## Shallow-water caveat

The deep-water dispersion relation `k = ω²/g` overestimates the true
wavenumber in shallow water, which causes the modelled profile to decay
faster with depth than the physical profile.  The approximation holds when
the water depth `h` exceeds half the deep-water wavelength:

```
h > g · T_p² / (8π)
```

For long swell (T_p ≈ 10 s) this threshold is about 39 m; for short wind
waves (T_p ≈ 4 s) it is only about 6 m.  In shallow basins such as the
Baltic Sea, users should check whether this condition is satisfied at the
depths of interest, particularly for long-period swell.

A more accurate alternative is the full finite-depth dispersion relation
`ω² = gk·tanh(kh)`, solved iteratively for `k`.  The monochromatic
deep-water model is preferred here for its simplicity and because CMEMS
products do not always provide bathymetry-corrected wave parameters at the
required resolution.
