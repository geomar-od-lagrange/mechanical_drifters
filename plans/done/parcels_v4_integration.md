# Plan: Drogued Drifter in Parcels v4

## Goal

Run the `DroguedDrifter` model inside OceanParcels v4 so that Lagrangian
particles are advected at the **steady-state drift velocity** of a buoy+drogue
system rather than at the raw surface current.

## Background

The `DroguedDrifter` solves an 8-DOF ODE (buoy position, pole angles, and their
rates) via `scipy.integrate.solve_ivp`. Given velocities at the buoy (surface)
and drogue (depth), `get_final_drift()` returns an xarray Dataset whose final
`xd`, `yd` values are the steady-state horizontal drift velocities.

Parcels v4 is a complete rewrite:
- Kernels are **vectorized** pure-Python functions with signature
  `(particles, fieldset)`.
- Position updates go through **delta accumulators** `dlon`, `dlat`, `dz`.
- FieldSets are built from `xr.Dataset` objects with SGRID metadata.
- Custom particle variables via `Particle.add_variable(Variable(...))`.

**Vectorization note:** `scipy.integrate.solve_ivp` does **not** broadcast over
multiple initial conditions — it solves one IVP at a time. The 8-DOF ODE is per
drifter (each has its own pole angles, velocities, etc.), so we cannot batch
them into a single system without coupling unrelated drifters. A per-particle
loop inside the kernel is necessary. This is fine for O(10–100) particles; for
larger ensembles, a lookup table or analytical steady-state approximation would
be the path forward.

## Design

### Coupling strategy

At each parcels timestep, for every particle:

1. **Sample** `(U_b, V_b)` at the buoy position `(lon, lat, z=0)` and
   `(U_d, V_d)` at the drogue position `(lon, lat, z=drogue_depth)` from the
   fieldset.
2. **Run** `DroguedDrifter.get_final_drift()` with a **constant-velocity
   callback** — a closure that ignores its `(t, z_d, y_b, x_b)` arguments and
   always returns the four sampled values `(U_b, V_b, U_d, V_d)`. This freezes
   the ocean field during the ~120 s mechanical relaxation so the ODE solver
   sees steady currents while the buoy+drogue system settles.
3. **Apply** the resulting drift velocities as `dlon += u_drift * dt`,
   `dlat += v_drift * dt`.

Key assumption: the ocean velocity field varies slowly compared to the ~120 s
mechanical relaxation time, so treating it as frozen is valid.

### Vertical motion

The drogued drifter model is purely horizontal. W is ignored — drifters stay at
z=0.

## Steps

### 1. Create a simple 3D FieldSet

File: `examples/parcels_3d_flow.py` (or notebook)

Build a synthetic 3D velocity field on a flat (Cartesian) grid:

```
U(x, y, z) = U_0 * exp(-z / H)      # eastward, decaying with depth
V(x, y, z) = 0                        # no meridional flow
```

- Domain: ~10 km × 10 km × 100 m deep
- Grid: ~50 × 50 × 20 points
- Time-independent (single snapshot)
- `U_0 = 0.5 m/s`, `H = 30 m` (e-folding depth)

Build the `xr.Dataset` with proper axis attributes (`axis: "X"`, `"Y"`, `"Z"`,
`"T"`) and SGRID metadata, then call `FieldSet.from_sgrid_conventions(ds,
mesh="flat")`.

### 2. Define custom particle type

No extra particle variables needed for the minimal example — the kernel only
needs `lon, lat, z, dt` which are already on the default `Particle`.

If we later want to track the drogue state between timesteps (theta, phi, etc.)
for warm-starting the ODE, add:

```python
DrifterParticle = Particle.add_variable([
    Variable("theta",  dtype=np.float64, initial=0.999 * np.pi),
    Variable("phi",    dtype=np.float64, initial=0.0),
    Variable("u_drift", dtype=np.float64, initial=0.0, to_write=True),
    Variable("v_drift", dtype=np.float64, initial=0.0, to_write=True),
])
```

But start without these — cold-start the ODE each timestep.

### 3. Write the custom kernel

Instantiate a single `DroguedDrifter` with default physical parameters. In the
per-particle loop, swap only the `get_uv` callback (which is cheap — just a
closure returning four floats).

```python
dd = DroguedDrifter()  # one instance, default Callies et al. params

def DroguedDrifterKernel(particles, fieldset):
    drogue_depth = fieldset.drogue_depth  # scalar constant, e.g. 3.0 m

    # Sample velocities at buoy (surface) and drogue (depth)
    (u_b, v_b) = fieldset.UV[particles.time, 0.0,          particles.lat, particles.lon, particles]
    (u_d, v_d) = fieldset.UV[particles.time, drogue_depth,  particles.lat, particles.lon, particles]

    # Loop over particles — solve_ivp is serial, cannot batch
    dt = particles.dt
    n = len(particles.lon)
    dlon = np.zeros(n)
    dlat = np.zeros(n)

    for i in range(n):
        dd.get_uv = lambda *, _ub=u_b[i], _vb=v_b[i], _ud=u_d[i], _vd=v_d[i], **kw: (_ub, _vb, _ud, _vd)
        ds = dd.get_final_drift(t_span=(0, 120))
        dlon[i] = float(ds.xd.isel(time=-1)) * dt[i]
        dlat[i] = float(ds.yd.isel(time=-1)) * dt[i]

    particles.dlon += dlon
    particles.dlat += dlat
```

**Notes:**
- Single `DroguedDrifter` instance reused across all particles and timesteps.
  Only the `get_uv` attribute is swapped each iteration (a cheap closure).
- `dt` in parcels is the outer timestep (e.g. 1 hour). The inner 120 s is the
  mechanical relaxation window.
- Could speed up later with warm-starting or pre-computed lookup tables.

### 4. Assemble and run

```python
pset = ParticleSet(fieldset=fieldset, pclass=Particle,
                   lon=[5000], lat=[5000], z=[0])

pset.execute(
    kernels=DroguedDrifterKernel,
    dt=np.timedelta64(1, "h"),
    runtime=np.timedelta64(24, "h"),
    output_file=ParticleFile(store="drifter_output.zarr",
                             outputdt=np.timedelta64(1, "h")),
)
```

### 5. Validate

- In the synthetic field, the surface current is `U_0 = 0.5 m/s` and the
  current at drogue depth (`z = 3 m`) is `0.5 * exp(-3/30) ≈ 0.45 m/s`.
- The drogued drifter drift velocity should be **between** these two values,
  biased toward the drogue current (since `k_d >> k_b`).
- Compare against a pure-advection particle at `z=0` (faster) and at
  `z=drogue_depth` (slower). The drogued drifter should sit in between but
  close to the drogue-depth result.

## Risks / Open questions

1. **Performance:** Each particle calls `solve_ivp` every outer timestep. Fine
   for O(10–100) particles. For larger ensembles: lookup table or analytical
   steady-state.

2. **Parcels v4 alpha stability:** The API may change. Pin to a specific commit.

3. **Field sampling syntax:** The exact indexing syntax
   `fieldset.UV[time, z, lat, lon, particles]` needs verification against the
   current v4 alpha — the docs are sparse and the API is in flux.

4. **Coordinate system:** The drogued drifter model works in meters (Cartesian).
   Using `mesh="flat"` keeps parcels in meters too. For real ocean applications
   we'll need to convert between degrees and meters.
