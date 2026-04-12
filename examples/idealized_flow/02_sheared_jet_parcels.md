---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region papermill={"duration": 0.006432, "end_time": "2026-04-11T15:45:20.520639+00:00", "exception": false, "start_time": "2026-04-11T15:45:20.514207+00:00", "status": "completed"} -->
# Drogued Drifter in an Idealized Sheared Flow with Parcels

A drogued drifter is a surface buoy connected by a rigid pole to a subsurface drogue. The drogue anchors the drifter to a target depth so that the buoy tracks currents at that depth rather than being blown by wind and surface waves. But the buoy still feels surface drag, so the actual drift velocity is a compromise between the surface current (acting on the buoy) and the deeper current (acting on the drogue). Under sufficient shear, the pole can tilt, and the drogue can drift at depths shallower than its equilibrium depth. The `DroguedDrifter` model computes this compromise from the full equations of motion.

This notebook demonstrates the drogued drifter model coupled to [Parcels v4](https://github.com/OceanParcels/parcels) in a synthetic 3D velocity field. We compare three types of Lagrangian particles:

1. **Drogued drifters**: advected at the steady-state drift velocity of the buoy+drogue system.
2. **Surface point particles**: advected at z = 0 (the fastest current).
3. **Drogue-depth point particles**: advected at the equilibrium drogue depth.

The drogued drifter should travel at an intermediate speed, between the surface and drogue-depth point particles.
<!-- #endregion -->

<!-- #region papermill={"duration": 0.001931, "end_time": "2026-04-11T15:45:20.525156+00:00", "exception": false, "start_time": "2026-04-11T15:45:20.523225+00:00", "status": "completed"} -->
## Imports
<!-- #endregion -->

```python papermill={"duration": 4.801162, "end_time": "2026-04-11T15:45:25.328051+00:00", "exception": false, "start_time": "2026-04-11T15:45:20.526889+00:00", "status": "completed"}
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleFile, ParticleSet, StatusCode
from parcels.kernels import AdvectionEE, AdvectionRK4
from scipy.special import erf

from drogued_drifters import DroguedDrifter
from drogued_drifters.parcels import make_dd_kernel
```

<!-- #region papermill={"duration": 0.001266, "end_time": "2026-04-11T15:45:25.330906+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.329640+00:00", "status": "completed"} -->
## Physical parameters

The velocity field consists of two opposing meandering jets on a flat Cartesian grid. The current decays exponentially with depth and rotates clockwise (Ekman-like), creating vertical shear that the drogued drifter model must resolve.

**Flow parameters:**
- `U_0`: peak surface current speed.
- `H`: e-folding depth for vertical decay. At depth `H`, the current is reduced to ~37% of its surface value.
- `L_Y`: meridional Gaussian half-width of each jet.
- `JET_SEP`: meridional distance between jet centres.
- `A_MEANDER`, `MEANDER_WAVELENGTH`: amplitude and wavelength of the sinusoidal jet meander.
- `MEANDER_PHASE_DEG`: phase offset between the two jets (in degrees).

**Drifter parameter:**
- `DROGUE_DEPTH`: depth of the drogue below the surface (3 m for the [Callies et al. (2017)](https://doi.org/10.5194/os-13-799-2017) drifter design).
<!-- #endregion -->

```python papermill={"duration": 0.004462, "end_time": "2026-04-11T15:45:25.336580+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.332118+00:00", "status": "completed"}
U_0 = 2.0              # peak surface current [m/s]
H = 3.0                # e-folding depth [m]
L_Y = 5_000.0          # jet half-width [m]
JET_SEP = 12_000.0     # separation between jet centres [m]
A_MEANDER = 3_000.0    # meander amplitude [m]
MEANDER_WAVELENGTH = 30_000.0  # meander wavelength [m]
MEANDER_PHASE_DEG = 60.0       # phase offset between jets [deg]

ROTATION_DEG_0 = 20.0     # rotation at surface [deg/e-fold]
ROTATION_DEG_DEEP = 90.0  # rotation at depth [deg/e-fold]
Z_ROT = 5.0               # depth scale for rotation increase [m]

DROGUE_DEPTH = 3.0  # drogue depth [m]

NX = 300              # grid x-dimension
NY = 150              # grid y-dimension
NZ = 20               # grid z-dimension
DT = 300.0            # timestep: 5 min [s]
RUNTIME = 86400.0     # total: 24 hours [s]
OUTPUTDT = 300.0      # output every 5 min
OUTPUT_DIR = "output"
```

<!-- #region papermill={"duration": 0.001287, "end_time": "2026-04-11T15:45:25.339260+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.337973+00:00", "status": "completed"} -->
## Build the velocity field

We use a **streamfunction** to prescribe the surface flow — not because non-divergence is important here, but because it is a convenient way to define a smooth, interesting flow pattern with meandering jets.

Each jet has a Gaussian cross-section with sinusoidal meanders. The streamfunction is

$$\psi_i(x, y) = -U_i \, L_Y \, \frac{\sqrt{\pi}}{2} \, \operatorname{erf}\!\left(\frac{y - y_{c,i}(x)}{L_Y}\right)$$

where $y_{c,i}(x)$ is the meandering centreline and $\operatorname{erf}$ is the error function (the antiderivative of the Gaussian $e^{-t^2}$, up to normalization). The velocity components follow from $U_s = -\partial\psi/\partial y$ and $V_s = \partial\psi/\partial x$.

The two jets flow in opposite directions (jet 1 eastward, jet 2 westward) and have different meander phases. The surface velocity decays exponentially with depth and rotates clockwise, mimicking Ekman dynamics:

$$\begin{pmatrix} U \\ V \end{pmatrix} = e^{-z/H} \begin{pmatrix} \cos\alpha(z) & -\sin\alpha(z) \\ \sin\alpha(z) & \cos\alpha(z) \end{pmatrix} \begin{pmatrix} U_s \\ V_s \end{pmatrix}$$

where the rotation rate increases with depth:

$$\alpha(z) = -\frac{z}{H}\left[\alpha_0 + (\alpha_\infty - \alpha_0)\left(1 - e^{-z/z_r}\right)\right]$$

with $\alpha_0 = 20°$/e-fold at the surface increasing to $\alpha_\infty = 90°$/e-fold at depth, over a scale $z_r = 5$ m.
<!-- #endregion -->

```python papermill={"duration": 0.056862, "end_time": "2026-04-11T15:45:25.397349+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.340487+00:00", "status": "completed"}
K_MEANDER = 2 * np.pi / MEANDER_WAVELENGTH
MEANDER_PHASE = np.deg2rad(MEANDER_PHASE_DEG)

x = np.linspace(-200_000, 200_000, NX)
y = np.linspace(-50_000, 50_000, NY)
depth = np.linspace(0, 100, NZ)
time = np.array([0.0])

Z, Y, X = np.meshgrid(depth, y, x, indexing="ij")

# Meandering jet centrelines
y_c1 = JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X)
y_c2 = -JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X + MEANDER_PHASE)
dy_c1_dx = A_MEANDER * K_MEANDER * np.cos(K_MEANDER * X)
dy_c2_dx = A_MEANDER * K_MEANDER * np.cos(K_MEANDER * X + MEANDER_PHASE)

eta1 = (Y - y_c1) / L_Y
eta2 = (Y - y_c2) / L_Y

# Surface velocity from streamfunction derivatives
U_surface = U_0 * np.exp(-eta1**2) - U_0 * np.exp(-eta2**2)
V_surface = U_0 * np.exp(-eta1**2) * dy_c1_dx - U_0 * np.exp(-eta2**2) * dy_c2_dx

# Depth decay with Ekman-like rotation
rot_deg_z = ROTATION_DEG_0 + (ROTATION_DEG_DEEP - ROTATION_DEG_0) * (1 - np.exp(-Z / Z_ROT))
angle = -np.radians(rot_deg_z) * Z / H
decay = np.exp(-Z / H)

U_data = (U_surface * decay * np.cos(angle) - V_surface * decay * np.sin(angle))[np.newaxis, ...]
V_data = (U_surface * decay * np.sin(angle) + V_surface * decay * np.cos(angle))[np.newaxis, ...]

# Build xarray Dataset with SGRID metadata
ds = xr.Dataset(
    {
        "U": (["time", "depth", "y", "x"], U_data),
        "V": (["time", "depth", "y", "x"], V_data),
        "grid": xr.DataArray(
            data=0,
            attrs={
                "cf_role": "grid_topology",
                "topology_dimension": 2,
                "node_dimensions": "x y",
                "face_dimensions": "x:x (padding: none) y:y (padding: none)",
                "vertical_dimensions": "depth:depth (padding: none)",
                "node_coordinates": "x y",
            },
        ),
    },
    coords={
        "x": ("x", x, {"axis": "X"}),
        "y": ("y", y, {"axis": "Y"}),
        "depth": ("depth", depth, {"axis": "Z"}),
        "time": ("time", time, {"axis": "T"}),
    },
)

fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
```

<!-- #region papermill={"duration": 0.001265, "end_time": "2026-04-11T15:45:25.400072+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.398807+00:00", "status": "completed"} -->
## Define the drogued drifter kernel

We use the `DDAdvectEE` kernel from `parcels_v4` to advect particles using the drogued drifter model. At each Parcels timestep, the kernel:

1. Extracts the velocity profile at all relevant depth levels using `fieldset.UV.eval()` (grid-agnostic).
2. Builds a fast depth interpolator from the sampled profiles.
3. Runs `DroguedDrifter.get_final_drift_batch` with the profile sampler.
4. Applies an Euler forward position update with the steady-state drift velocity.

The kernel auto-detects spherical/flat mesh from the FieldSet.
<!-- #endregion -->

```python papermill={"duration": 0.004038, "end_time": "2026-04-11T15:45:25.405323+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.401285+00:00", "status": "completed"}
dd = DroguedDrifter()
dd_kernel = make_dd_kernel(dd)


def DeleteOOB(particles, fieldset):
    """Convert out-of-bounds errors to Delete status."""
    state = np.asarray(particles.state)
    oob = (state == StatusCode.ErrorOutOfBounds) | (state == StatusCode.ErrorThroughSurface)
    if np.any(oob):
        particles.state = np.where(oob, StatusCode.Delete, state)
```

<!-- #region papermill={"duration": 0.001242, "end_time": "2026-04-11T15:45:25.407891+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.406649+00:00", "status": "completed"} -->
## Run simulations

We release particles in a grid spanning both jets at several x-positions. Three simulations:

1. **Drogued drifter**: `DDAdvectEE` kernel on the standard FieldSet (the kernel extracts velocity profiles and computes drift internally).
2. **Surface point particle**: `AdvectionRK4` at z=0 on the same FieldSet.
3. **Drogue-depth point particle**: `AdvectionRK4` at z=3m on the same FieldSet.
<!-- #endregion -->

```python papermill={"duration": 0.00431, "end_time": "2026-04-11T15:45:25.413678+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.409368+00:00", "status": "completed"}
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# Release grid: 5 x-positions, 10 y-positions spanning both jets
_lats = np.linspace(-JET_SEP / 2 - 2 * L_Y, JET_SEP / 2 + 2 * L_Y, 10)
_x_positions = np.linspace(5_000, 25_000, 5)
_lon_grid, _lat_grid = np.meshgrid(_x_positions, _lats)
release_lons = _lon_grid.ravel().tolist()
release_lats = _lat_grid.ravel().tolist()
n_particles = len(release_lats)
```

```python papermill={"duration": 74.910591, "end_time": "2026-04-11T15:46:40.325515+00:00", "exception": false, "start_time": "2026-04-11T15:45:25.414924+00:00", "status": "completed"}
# Drogued drifter run — DDAdvectEE kernel
dd_store = str(output_dir / "01_drogued_drifter.zarr")
shutil.rmtree(dd_store, ignore_errors=True)

pset_drifter = ParticleSet(
    fieldset=fieldset,
    pclass=Particle,
    lon=release_lons,
    lat=release_lats,
    z=[0] * n_particles,
)
pset_drifter.execute(
    kernels=[dd_kernel, DeleteOOB],
    dt=DT,
    runtime=RUNTIME,
    output_file=ParticleFile(store=dd_store, outputdt=OUTPUTDT),
    verbose_progress=False,
)
```

```python papermill={"duration": 3.545334, "end_time": "2026-04-11T15:46:43.872731+00:00", "exception": false, "start_time": "2026-04-11T15:46:40.327397+00:00", "status": "completed"}
surface_store = str(output_dir / "01_surface_pp.zarr")
shutil.rmtree(surface_store, ignore_errors=True)

pset_surface = ParticleSet(
    fieldset=fieldset,
    pclass=Particle,
    lon=release_lons,
    lat=release_lats,
    z=[0] * n_particles,
)
pset_surface.execute(
    kernels=[AdvectionRK4, DeleteOOB],
    dt=DT,
    runtime=RUNTIME,
    output_file=ParticleFile(store=surface_store, outputdt=OUTPUTDT),
    verbose_progress=False,
)
```

```python papermill={"duration": 3.569999, "end_time": "2026-04-11T15:46:47.444535+00:00", "exception": false, "start_time": "2026-04-11T15:46:43.874536+00:00", "status": "completed"}
drogue_store = str(output_dir / "01_drogue_depth_pp.zarr")
shutil.rmtree(drogue_store, ignore_errors=True)

pset_drogue = ParticleSet(
    fieldset=fieldset,
    pclass=Particle,
    lon=release_lons,
    lat=release_lats,
    z=[DROGUE_DEPTH] * n_particles,
)
pset_drogue.execute(
    kernels=[AdvectionRK4, DeleteOOB],
    dt=DT,
    runtime=RUNTIME,
    output_file=ParticleFile(store=drogue_store, outputdt=OUTPUTDT),
    verbose_progress=False,
)
```

<!-- #region papermill={"duration": 0.001493, "end_time": "2026-04-11T15:46:47.447787+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.446294+00:00", "status": "completed"} -->
## Plot trajectories

Streamfunction contours show the flow structure. Surface point particles (blue) travel furthest, drogue-depth point particles (orange) travel least, and the drogued drifters (green) fall in between.
<!-- #endregion -->

```python papermill={"duration": 21.836693, "end_time": "2026-04-11T15:47:09.286152+00:00", "exception": false, "start_time": "2026-04-11T15:46:47.449459+00:00", "status": "completed"}
ds_drifter = xr.open_zarr(dd_store)
ds_surface = xr.open_zarr(surface_store)
ds_drogue = xr.open_zarr(drogue_store)

# Streamfunction contours for flow context
x_plot = np.linspace(x.min(), x.max(), 400)
y_plot = np.linspace(y.min(), y.max(), 400)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
yc1_plot = JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X_plot)
yc2_plot = -JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X_plot + MEANDER_PHASE)
C_psi = U_0 * L_Y * np.sqrt(np.pi) / 2
psi = -C_psi * erf((Y_plot - yc1_plot) / L_Y) + C_psi * erf((Y_plot - yc2_plot) / L_Y)

fig, ax = plt.subplots()
fig.set_dpi(300)
ax.contour(X_plot / 1000, Y_plot / 1000, psi, levels=25, colors="0.8", linewidths=0.5)

for i in range(ds_surface.sizes["trajectory"]):
    lon_i = ds_surface.lon.isel(trajectory=i)
    lat_i = ds_surface.lat.isel(trajectory=i)
    ax.plot(lon_i / 1000, lat_i / 1000, color="tab:blue", linewidth=0.3,
            label="Surface point particle" if i == 0 else None)

for i in range(ds_drogue.sizes["trajectory"]):
    lon_i = ds_drogue.lon.isel(trajectory=i)
    lat_i = ds_drogue.lat.isel(trajectory=i)
    ax.plot(lon_i / 1000, lat_i / 1000, color="tab:orange", linewidth=0.3,
            label=f"Drogue-depth point particle (z={DROGUE_DEPTH}m)" if i == 0 else None)

for i in range(ds_drifter.sizes["trajectory"]):
    lon_i = ds_drifter.lon.isel(trajectory=i)
    lat_i = ds_drifter.lat.isel(trajectory=i)
    ax.plot(lon_i / 1000, lat_i / 1000, color="tab:green", linewidth=0.3,
            label="Drogued drifter" if i == 0 else None)

ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.legend()
plt.tight_layout()
plt.show()
```

```python papermill={"duration": 28.819712, "end_time": "2026-04-11T15:47:38.110126+00:00", "exception": false, "start_time": "2026-04-11T15:47:09.290414+00:00", "status": "completed"}
# Compute mean drift speeds from final positions
def compute_mean_drift_speed(ds, runtime):
    """Compute mean drift speed from final particle positions.
    
    Args:
        ds: Zarr dataset from ParticleFile output
        runtime: Total simulation time [seconds]
    
    Returns:
        Mean drift speed [m/s] across all particles
    """
    # Get initial and final positions for each trajectory
    lon_init = ds.lon.isel(obs=0)
    lat_init = ds.lat.isel(obs=0)
    
    # Find the last valid (non-NaN) position for each trajectory
    lon_final = np.full(ds.sizes["trajectory"], np.nan)
    lat_final = np.full(ds.sizes["trajectory"], np.nan)
    
    for i in range(ds.sizes["trajectory"]):
        lon_traj = ds.lon.isel(trajectory=i).dropna("obs")
        lat_traj = ds.lat.isel(trajectory=i).dropna("obs")
        if lon_traj.sizes["obs"] > 0:
            lon_final[i] = float(lon_traj.isel(obs=-1))
            lat_final[i] = float(lat_traj.isel(obs=-1))
    
    # Compute displacement for each particle
    dx = lon_final - lon_init.values
    dy = lat_final - lat_init.values
    displacement = np.sqrt(dx**2 + dy**2)
    
    # Compute mean drift speed
    mean_speed = np.nanmean(displacement) / runtime
    return mean_speed

speed_dd = compute_mean_drift_speed(ds_drifter, RUNTIME)
speed_surface = compute_mean_drift_speed(ds_surface, RUNTIME)
speed_drogue = compute_mean_drift_speed(ds_drogue, RUNTIME)

print(f"Mean drift speed — Surface point particle:       {speed_surface:.4f} m/s")
print(f"Mean drift speed — Drogue-depth point particle:  {speed_drogue:.4f} m/s")
print(f"Mean drift speed — Drogued drifter:              {speed_dd:.4f} m/s")
print(f"\nDrogued drifter is {100 * (speed_surface - speed_dd) / (speed_surface - speed_drogue):.1f}% between surface and drogue-depth.")
```
