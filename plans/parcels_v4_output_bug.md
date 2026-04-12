# Parcels v4 alpha: zarr output bug with custom kernels

## Problem

When using `ParticleFile(store=..., outputdt=...)` with a custom kernel
(not the built-in AdvectionRK4), the zarr writer crashes with a
`KeyError: 'lon'` deep in `parcels._core.particlefile`. The built-in
advection kernels work fine with zarr output.

## Observed with

- Parcels installed from `git+https://github.com/OceanParcels/parcels.git@main`
  (commit around March 2026)
- Python 3.14, macOS ARM64
- Custom kernel signature `(particles, fieldset)` modifying `dlon`/`dlat`

## Workaround

Store trajectories in a Python dict inside the kernel:

```python
drifter_trajectory = {"lon": [], "lat": [], "time": []}

def MyKernel(particles, fieldset):
    # ... compute dlon, dlat ...
    drifter_trajectory["lon"].append(np.asarray(particles.lon).copy())
    drifter_trajectory["lat"].append(np.asarray(particles.lat).copy())
    drifter_trajectory["time"].append(np.asarray(particles.time).copy())
```

This records positions at the start of each timestep (before the
position update). Append the final position after `execute()` returns.

The built-in `AdvectionRK4` runs use `ParticleFile` without issues,
so only the custom kernel output path is affected.

## To reproduce

```python
pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=[0], lat=[0], z=[0])
pset.execute(
    kernels=MyCustomKernel,
    dt=300.0,
    runtime=3600.0,
    output_file=ParticleFile(store="output.zarr", outputdt=300.0),
)
# -> KeyError: 'lon' in parcels._core.particlefile
```

## TODO

- Open issue / PR in https://github.com/OceanParcels/parcels
- Likely related to how the zarr schema is initialized when the first
  kernel is not a built-in advection kernel
