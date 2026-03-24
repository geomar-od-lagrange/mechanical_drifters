"""Drogued drifter model coupled to Parcels v4 in a synthetic 3D flow.

Creates a depth-decaying eastward current with Ekman-like clockwise rotation
on a flat Cartesian grid, then runs three particles for comparison:

1. **Drogued drifter** -- advected at the steady-state drift velocity of the
   buoy+drogue system (DroguedDrifter model).
2. **Surface tracer** -- pure AdvectionRK4 at z = 0 (fastest).
3. **Drogue-depth tracer** -- pure AdvectionRK4 at z = drogue_depth.

The surface velocity is derived from a streamfunction:
    psi(x, y) = -U_0 * L_y * sqrt(pi)/2 * erf((y - A*sin(k*x)) / L_y)

giving a Gaussian jet with sinusoidal meanders (amplitude A, wavenumber k).
U = -dpsi/dy, V = dpsi/dx. The flow decays exponentially with depth and
rotates clockwise (Ekman-like).
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleFile, ParticleSet, StatusCode, Variable
from parcels._core.statuscodes import FieldOutOfBoundError
from parcels.kernels import AdvectionRK4

from drogued_drifters.drifter import DroguedDrifter

# Custom particle with effective drogue depth diagnostic
DrifterParticle = Particle.add_variable(Variable("z_eff", dtype=np.float64, initial=0.0))

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
U_0 = 2.0  # peak surface current [m/s]
H = 3.0  # e-folding depth [m]
L_Y = 5_000.0  # jet half-width [m]
JET_SEP = 12_000.0  # separation between jet centres [m]
A_MEANDER = 3_000.0  # meander amplitude [m]
K_MEANDER = 2 * np.pi / 30_000.0  # meander wavenumber [1/m] (wavelength 30km)
MEANDER_PHASE = np.pi / 3  # phase offset between the two jets
ROTATION_DEG_0 = 20.0  # rotation at surface [deg/e-fold]
ROTATION_DEG_DEEP = 90.0  # rotation at depth [deg/e-fold]
Z_ROT = 5.0  # depth scale over which rotation increases [m]
DROGUE_DEPTH = 3.0  # drogue depth [m]

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
NX, NY, NZ = 300, 150, 20
x = np.linspace(-200_000, 200_000, NX)
y = np.linspace(-50_000, 50_000, NY)
depth = np.linspace(0, 100, NZ)
time = np.array([0.0])

# ---------------------------------------------------------------------------
# 1. Build synthetic 3D velocity field from streamfunction
# ---------------------------------------------------------------------------
Z, Y, X = np.meshgrid(depth, y, x, indexing="ij")

# Two opposing meandering jets, offset by JET_SEP with different meander phases.
# Jet 1: eastward, centred at y = +JET_SEP/2
# Jet 2: westward, centred at y = -JET_SEP/2
#
# Streamfunction is additive:
#   psi = psi_1 + psi_2
#   psi_i = -U_i * L_Y * sqrt(pi)/2 * erf(eta_i)
# with U_1 = +U_0, U_2 = -U_0 (opposing).

y_c1 = JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X)
y_c2 = -JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X + MEANDER_PHASE)
dy_c1_dx = A_MEANDER * K_MEANDER * np.cos(K_MEANDER * X)
dy_c2_dx = A_MEANDER * K_MEANDER * np.cos(K_MEANDER * X + MEANDER_PHASE)

eta1 = (Y - y_c1) / L_Y
eta2 = (Y - y_c2) / L_Y

# U_s = -dpsi/dy = sum of Gaussian jet profiles (jet 2 is negative = westward)
# V_s =  dpsi/dx = sum of meander-induced cross-jet velocities
U_surface = U_0 * np.exp(-eta1**2) - U_0 * np.exp(-eta2**2)
V_surface = U_0 * np.exp(-eta1**2) * dy_c1_dx - U_0 * np.exp(-eta2**2) * dy_c2_dx

# Apply depth decay with depth-varying Ekman-like rotation
# Rotation rate increases from ROTATION_DEG_0 at surface to ROTATION_DEG_DEEP at depth
rot_deg_z = ROTATION_DEG_0 + (ROTATION_DEG_DEEP - ROTATION_DEG_0) * (1 - np.exp(-Z / Z_ROT))
angle = -np.radians(rot_deg_z) * Z / H  # negative = clockwise
decay = np.exp(-Z / H)
U_data = (U_surface * decay * np.cos(angle) - V_surface * decay * np.sin(angle))[np.newaxis, ...]
V_data = (U_surface * decay * np.sin(angle) + V_surface * decay * np.cos(angle))[np.newaxis, ...]

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
fieldset.add_constant("drogue_depth", DROGUE_DEPTH)

# ---------------------------------------------------------------------------
# 2. Drogued-drifter custom kernel
# ---------------------------------------------------------------------------
dd = DroguedDrifter()  # single instance, default Callies et al. params
_warm_state = {}  # warm-start state for get_final_drift_batch

# Storage for drogued-drifter trajectory (workaround for parcels v4 alpha
# zarr bug that can corrupt the output store with custom kernels).
# Each entry is an (n_particles,) array for one timestep.
drifter_trajectory = {"lon": [], "lat": [], "time": [], "z_eff": []}


def DroguedDrifterKernel(particles, fieldset):
    """Advect particles at the steady-state drift velocity of a buoy+drogue."""
    drogue_depth = fieldset.drogue_depth

    # Sample UV at the buoy (surface, z=0) and drogue (z=drogue_depth)
    n = len(np.asarray(particles.lon))
    z_surface = np.zeros(n)
    z_drogue = np.full(n, drogue_depth)
    try:
        (u_b, v_b) = fieldset.UV[
            particles.time, z_surface, particles.lat, particles.lon, particles
        ]
        (u_d, v_d) = fieldset.UV[
            particles.time, z_drogue, particles.lat, particles.lon, particles
        ]
    except FieldOutOfBoundError:
        # Mark all particles for deletion — can't distinguish which one is OOB
        particles.state = StatusCode.Delete
        return

    # Convert ParticleSetViewArray objects to plain numpy arrays
    u_b, v_b = np.asarray(u_b), np.asarray(v_b)
    u_d, v_d = np.asarray(u_d), np.asarray(v_d)
    dt = np.asarray(particles.dt)
    lon_arr = np.asarray(particles.lon)
    lat_arr = np.asarray(particles.lat)
    time_arr = np.asarray(particles.time)

    # Vectorized batch solve — one solve_ivp call for all particles
    n = len(u_b)
    y0_warm = _warm_state.get("Y") if _warm_state.get("n") == n else None
    xd_drift, yd_drift, theta_final, Y_final = dd.get_final_drift_batch(
        U_b=u_b, V_b=v_b, U_d=u_d, V_d=v_d, y0=y0_warm,
    )
    _warm_state["Y"] = Y_final
    _warm_state["n"] = n
    z_eff = -dd.l * np.cos(theta_final)

    particles.dlon += xd_drift * dt
    particles.dlat += yd_drift * dt
    particles.z_eff = z_eff

    # Record trajectory for later plotting (one snapshot per timestep)
    drifter_trajectory["lon"].append(lon_arr.copy())
    drifter_trajectory["lat"].append(lat_arr.copy())
    drifter_trajectory["time"].append(time_arr.copy())
    drifter_trajectory["z_eff"].append(z_eff.copy())


def DeleteOOB(particles, fieldset):
    """Convert out-of-bounds error states to Delete so parcels doesn't crash."""
    state = np.asarray(particles.state)
    oob = (state == StatusCode.ErrorOutOfBounds) | (state == StatusCode.ErrorThroughSurface)
    if np.any(oob):
        particles.state = np.where(oob, StatusCode.Delete, state)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT = 300.0  # timestep: 5 min [s]
RUNTIME = 24 * 3600.0  # total: 24 hours [s]
OUTPUTDT = 300.0  # output every 5 min

OUTPUT_DIR = Path("examples/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Release particles across both jets at three x-positions
_lats = np.linspace(-JET_SEP / 2 - 2 * L_Y, JET_SEP / 2 + 2 * L_Y, 21)
_x_positions = np.linspace(5_000, 55_000, 10)
_lon_grid, _lat_grid = np.meshgrid(_x_positions, _lats)
release_lons = _lon_grid.ravel().tolist()
release_lats = _lat_grid.ravel().tolist()
n_particles = len(release_lats)

# ---------------------------------------------------------------------------
# 3a. Run drogued drifters (no zarr output -- collected in-kernel)
# ---------------------------------------------------------------------------
print(f"Running {n_particles} drogued drifters across jet...")

pset_drifter = ParticleSet(
    fieldset=fieldset,
    pclass=DrifterParticle,
    lon=release_lons,
    lat=release_lats,
    z=[0] * n_particles,
)
pset_drifter.execute(
    kernels=[DroguedDrifterKernel, DeleteOOB],
    dt=DT,
    runtime=RUNTIME,
    verbose_progress=True,
)
n_remaining = len(pset_drifter)
print(f"  {n_remaining}/{n_particles} particles remaining")

# ---------------------------------------------------------------------------
# 3b. Run surface tracers (z=0)
# ---------------------------------------------------------------------------
print(f"Running {n_particles} surface tracers (z=0)...")
surface_store = str(OUTPUT_DIR / "surface_tracer.zarr")
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
    verbose_progress=True,
)

# ---------------------------------------------------------------------------
# 3c. Run drogue-depth tracers (z=drogue_depth)
# ---------------------------------------------------------------------------
print(f"Running {n_particles} drogue-depth tracers (z={DROGUE_DEPTH}m)...")
drogue_store = str(OUTPUT_DIR / "drogue_depth_tracer.zarr")
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
    verbose_progress=True,
)

# ---------------------------------------------------------------------------
# 4. Validation plot
# ---------------------------------------------------------------------------
print("Creating validation plot...")

ds_surface = xr.open_zarr(surface_store)
ds_drogue = xr.open_zarr(drogue_store)

# Drifter trajectories are ragged (particles may be deleted). Each entry in
# drifter_trajectory["lon"] is an array whose length may shrink over time.
# We plot each particle's trajectory from its per-step snapshots.

fig, ax = plt.subplots(figsize=(14, 8))

# Plot streamfunction contours (surface, z=0)
from scipy.special import erf

x_plot = np.linspace(x.min(), x.max(), 400)
y_plot = np.linspace(y.min(), y.max(), 400)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
yc1_plot = JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X_plot)
yc2_plot = -JET_SEP / 2 + A_MEANDER * np.sin(K_MEANDER * X_plot + MEANDER_PHASE)
eta1_plot = (Y_plot - yc1_plot) / L_Y
eta2_plot = (Y_plot - yc2_plot) / L_Y
C = U_0 * L_Y * np.sqrt(np.pi) / 2
psi = -C * erf(eta1_plot) + C * erf(eta2_plot)
ax.contour(X_plot, Y_plot, psi, levels=25, colors="0.7", linewidths=0.5)

# Plot zarr-based trajectories (surface and drogue-depth tracers)
n_surface = ds_surface.sizes["trajectory"]
for i in range(n_surface):
    lon_i = ds_surface.lon.values[i, :]
    lat_i = ds_surface.lat.values[i, :]
    valid = np.isfinite(lon_i)
    ax.plot(lon_i[valid], lat_i[valid], color="tab:red", linewidth=1, alpha=0.4,
            label="Surface tracer" if i == 0 else None)

n_drogue = ds_drogue.sizes["trajectory"]
for i in range(n_drogue):
    lon_i = ds_drogue.lon.values[i, :]
    lat_i = ds_drogue.lat.values[i, :]
    valid = np.isfinite(lon_i)
    ax.plot(lon_i[valid], lat_i[valid], color="tab:green", linewidth=1, alpha=0.4,
            label=f"Drogue-depth tracer (z={DROGUE_DEPTH}m)" if i == 0 else None)

# Plot drogued drifter trajectories from in-kernel snapshots.
# Build per-particle trajectories by matching across snapshots.
# All particles start alive; the array shrinks as particles are deleted.
# We track by initial index using the first snapshot as reference.
if drifter_trajectory["lon"]:
    # For simplicity, stack snapshots that have the same length as the first
    n0 = len(drifter_trajectory["lon"][0])
    full_steps = [i for i, arr in enumerate(drifter_trajectory["lon"]) if len(arr) == n0]
    if full_steps:
        d_lon = np.array([drifter_trajectory["lon"][i] for i in full_steps])
        d_lat = np.array([drifter_trajectory["lat"][i] for i in full_steps])
        for i in range(n0):
            ax.plot(d_lon[:, i], d_lat[:, i], color="tab:blue", linewidth=1,
                    linestyle="--", alpha=0.4,
                    label="Drogued drifter" if i == 0 else None)

# Plot start positions
for i in range(n_particles):
    ax.plot(release_lons[i], release_lats[i], "ko", markersize=3)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(
    f"Drogued drifter vs. passive tracers in opposing meandering jets (24h)\n"
    f"U\u2080={U_0} m/s, H={H} m, L_y={L_Y/1000:.0f} km, "
    f"sep={JET_SEP/1000:.0f} km, A={A_MEANDER/1000:.0f} km"
)
# Auto-zoom to trajectory extent with padding
all_lons = []
all_lats = []
for i in range(n_surface):
    v = np.isfinite(ds_surface.lon.values[i, :])
    all_lons.extend(ds_surface.lon.values[i, v])
    all_lats.extend(ds_surface.lat.values[i, v])
for i in range(n_drogue):
    v = np.isfinite(ds_drogue.lon.values[i, :])
    all_lons.extend(ds_drogue.lon.values[i, v])
    all_lats.extend(ds_drogue.lat.values[i, v])
if drifter_trajectory["lon"] and full_steps:
    all_lons.extend(d_lon.ravel())
    all_lats.extend(d_lat.ravel())
pad = 5_000
ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "parcels_3d_flow.png"), dpi=150)
print(f"  Plot saved to {OUTPUT_DIR / 'parcels_3d_flow.png'}")
