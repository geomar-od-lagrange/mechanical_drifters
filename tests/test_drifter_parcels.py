"""Tests for Parcels coupling (kernel, make_profile_sampler).

These tests verify:
- Profile sampler linear interpolation in depth
- Boundary handling (searchsorted clipping)
- Kernel with synthetic FieldSets (uniform and sheared flow)
"""

import numpy as np
import pytest

from mechanical_drifters.models.drogued_drifter import DroguedDrifter
from mechanical_drifters.models.spar_buoy_simple import SparBuoySimple
from mechanical_drifters.parcels import make_kernel
from mechanical_drifters.parcels import _make_profile_sampler as make_profile_sampler


def test_make_profile_sampler_basic():
    """Profile sampler should interpolate linearly in depth (z-up convention)."""
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 3
    U_profiles = np.array(
        [
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7],
        ]
    )
    V_profiles = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ]
    )

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(0.0)
    np.testing.assert_allclose(U, [0.5, 0.6, 0.7], rtol=1e-10)
    np.testing.assert_allclose(V, [0.1, 0.2, 0.3], rtol=1e-10)

    U, V = sample_uv(-10.0)
    np.testing.assert_allclose(U, [0.3, 0.4, 0.5], rtol=1e-10)
    np.testing.assert_allclose(V, [0.1, 0.2, 0.3], rtol=1e-10)


def test_make_profile_sampler_linear_interpolation():
    """Verify linear interpolation between depth levels."""
    depth_levels = np.array([-10.0, 0.0])
    N = 1
    U_profiles = np.array([[1.0], [0.0]])
    V_profiles = np.array([[0.0], [0.0]])

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(-2.5)
    np.testing.assert_allclose(U, [0.25], rtol=1e-10)


def test_make_profile_sampler_vectorized_z():
    """Sample at multiple z values for N particles simultaneously."""
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 2
    U_profiles = np.array(
        [
            [0.5, 0.6],
            [0.3, 0.4],
            [0.1, 0.2],
        ]
    )
    V_profiles = np.zeros((3, 2))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(-5.0)
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"
    np.testing.assert_allclose(U, [0.3, 0.4], rtol=1e-10)


def test_make_profile_sampler_boundary_shallow():
    """At z values shallower than shallowest level, should clip."""
    depth_levels = np.array([-10.0, -5.0, -2.0])
    N = 1
    U_profiles = np.array([[0.3], [0.2], [0.1]])
    V_profiles = np.zeros((3, 1))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(0.0)
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"
    assert np.isfinite(U[0])


def test_make_profile_sampler_boundary_deep():
    """At depths deeper than deepest level, should clip to deepest."""
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 1
    U_profiles = np.array([[0.3], [0.2], [0.1]])
    V_profiles = np.zeros((3, 1))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(-100.0)
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"
    assert np.isfinite(U[0])
    assert U[0] >= 0.2


def test_make_profile_sampler_multiple_particles():
    """Sample with multiple particles in profile."""
    depth_levels = np.array([-5.0, 0.0])
    N = 5
    U_profiles = (
        np.arange(N, dtype=float).reshape(1, -1) * 0.1
    )
    U_profiles = np.vstack([U_profiles + 0.05, U_profiles])
    V_profiles = np.zeros_like(U_profiles)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U_surface, _ = sample_uv(0.0)
    assert U_surface.shape == (N,), f"Expected shape ({N},), got {U_surface.shape}"
    np.testing.assert_allclose(U_surface, np.arange(N) * 0.1, rtol=1e-10)


def test_make_profile_sampler_degenerate_interval():
    """Handle degenerate depth interval (two identical depths at surface)."""
    depth_levels = np.array([-5.0, 0.0, 0.0])
    N = 1
    U_profiles = np.array([[0.2], [0.1], [0.1]])
    V_profiles = np.zeros((3, 1))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(-2.5)
    assert np.isfinite(U[0]), f"Expected finite U, got {U}"


def test_profile_sampler_broadcast_scalar_z():
    """Profile sampler should broadcast scalar z to (N,) for N particles."""
    depth_levels = np.array([-10.0, 0.0])
    N = 3
    U_profiles = np.array([[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]])
    V_profiles = np.zeros((2, N))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(-5.0)
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"


def test_profile_sampler_vectorized_batch_z():
    """Sampler should accept vector z and return vector (U, V)."""
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 4
    U_profiles = np.array(
        [
            [0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    V_profiles = np.array(
        [
            [0.05, 0.1, 0.15, 0.2],
            [0.05, 0.1, 0.15, 0.2],
            [0.05, 0.1, 0.15, 0.2],
        ]
    )

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    z_batch = np.array([0.0, -5.0, -10.0, -2.5])
    U, V = sample_uv(z_batch)

    assert U.shape == (4,)
    assert V.shape == (4,)
    np.testing.assert_allclose(U[0], 0.1, rtol=1e-10)
    np.testing.assert_allclose(U[1], 0.3, rtol=1e-10)


def test_profile_sampler_preserves_velocity_profiles():
    """Sampler should preserve the original velocity profiles at sampled depths."""
    depth_levels = np.array([-15.0, -10.0, -5.0, 0.0])
    N = 2
    rng = np.random.default_rng(42)
    U_profiles = rng.uniform(0, 1, size=(4, N))
    V_profiles = rng.uniform(0, 1, size=(4, N))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    for iz, z in enumerate(depth_levels):
        U, V = sample_uv(z)
        np.testing.assert_allclose(U, U_profiles[iz], rtol=1e-10, err_msg=f"U mismatch at z={z}")
        np.testing.assert_allclose(V, V_profiles[iz], rtol=1e-10, err_msg=f"V mismatch at z={z}")


def test_profile_sampler_nan_handling():
    """Profile sampler should propagate NaN in velocity profiles."""
    depth_levels = np.array([-10.0, 0.0])
    N = 2
    U_profiles = np.array([[0.2, np.nan], [0.1, 0.3]])
    V_profiles = np.zeros((2, N))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    U, V = sample_uv(-5.0)
    assert np.isnan(U[1]), "Should preserve NaN from velocity profile"


# ---------------------------------------------------------------------------
# Integration tests for kernel with synthetic FieldSets
# ---------------------------------------------------------------------------


def _make_flat_fieldset(U_data_4d, V_data_4d, x, y, depth, time):
    """Build a flat-mesh FieldSet from 4-D numpy arrays."""
    import xarray as xr
    from parcels import FieldSet

    ds = xr.Dataset(
        {
            "U": (["time", "depth", "y", "x"], U_data_4d),
            "V": (["time", "depth", "y", "x"], V_data_4d),
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
    return FieldSet.from_sgrid_conventions(ds, mesh="flat")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_uniform_flow_dd_kernel(backend):
    """Uniform flow: buoy and drogue see the same current, drift = current."""
    from parcels import FieldSet, Particle, ParticleSet

    U_const = 0.5
    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    depth = np.array([0.0, 5.0, 10.0])
    time = np.array([0.0])

    U_data = np.full((1, len(depth), len(y), len(x)), U_const)
    V_data = np.zeros_like(U_data)

    fieldset = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    dd = DroguedDrifter(backend=backend)

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=[500.0],
        lat=[500.0],
        z=[0.0],
    )

    DT = 60.0
    pset.execute(
        kernels=[make_kernel(dd)],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )

    lon_final = float(np.asarray(pset.lon)[0])
    displacement = lon_final - 500.0
    expected = U_const * DT
    np.testing.assert_allclose(displacement, expected, rtol=0.05)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_sheared_flow_dd_kernel(backend):
    """Sheared flow: drift velocity should be between surface and bottom current."""
    from parcels import FieldSet, Particle, ParticleSet

    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    depth = np.array([0.0, 1.5, 3.0, 5.0, 10.0])
    time = np.array([0.0])

    U_surface = 1.0
    U_data = np.zeros((1, len(depth), len(y), len(x)))
    for iz, d in enumerate(depth):
        U_data[0, iz, :, :] = U_surface * (1.0 - d / 10.0)
    V_data = np.zeros_like(U_data)

    fieldset = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    dd = DroguedDrifter(backend=backend)

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=[500.0],
        lat=[500.0],
        z=[0.0],
    )

    DT = 60.0
    pset.execute(
        kernels=[make_kernel(dd)],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )

    lon_final = float(np.asarray(pset.lon)[0])
    displacement = lon_final - 500.0
    drift_speed = displacement / DT

    assert 0.0 < drift_speed < U_surface, f"Drift speed {drift_speed:.4f} should be between 0 and {U_surface}"
    assert drift_speed < 0.9, f"Drift speed {drift_speed:.4f} should be < 0.9 (drogue effect)"


def _make_spherical_fieldset(U_data_4d, V_data_4d, lon, lat, depth, time):
    """Build a spherical-mesh FieldSet from 4-D numpy arrays."""
    import xarray as xr
    from parcels import FieldSet

    ds = xr.Dataset(
        {
            "U": (["time", "depth", "lat", "lon"], U_data_4d),
            "V": (["time", "depth", "lat", "lon"], V_data_4d),
            "grid": xr.DataArray(
                data=0,
                attrs={
                    "cf_role": "grid_topology",
                    "topology_dimension": 2,
                    "node_dimensions": "lon lat",
                    "face_dimensions": "lon:lon (padding: none) lat:lat (padding: none)",
                    "vertical_dimensions": "depth:depth (padding: none)",
                    "node_coordinates": "lon lat",
                },
            ),
        },
        coords={
            "lon": ("lon", lon, {"axis": "X"}),
            "lat": ("lat", lat, {"axis": "Y"}),
            "depth": ("depth", depth, {"axis": "Z"}),
            "time": ("time", time, {"axis": "T"}),
        },
    )
    return FieldSet.from_sgrid_conventions(ds, mesh="spherical")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_dd_kernel_spherical_auto(backend):
    """Spherical mesh: kernel auto-detects deg/s and converts correctly."""
    from parcels import Particle, ParticleSet

    U_ms = 0.5
    lat0 = 5.0
    lon0 = 10.0

    lon = np.linspace(lon0 - 1, lon0 + 1, 5)
    lat = np.linspace(lat0 - 1, lat0 + 1, 5)
    depth = np.array([0.0, 5.0, 10.0])
    time = np.array([0.0])

    U_data = np.full((1, len(depth), len(lat), len(lon)), U_ms)
    V_data = np.zeros_like(U_data)

    fieldset = _make_spherical_fieldset(U_data, V_data, lon, lat, depth, time)
    dd = DroguedDrifter(backend=backend)

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=[lon0],
        lat=[lat0],
        z=[0.0],
    )

    DT = 60.0
    pset.execute(
        kernels=[make_kernel(dd)],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )

    DEG2M = 1852.0 * 60.0
    cos_lat = np.cos(np.deg2rad(lat0))
    expected_dlon = U_ms * DT / (DEG2M * cos_lat)

    dlon = float(np.asarray(pset.lon)[0]) - lon0
    dlat = float(np.asarray(pset.lat)[0]) - lat0

    np.testing.assert_allclose(dlon, expected_dlon, rtol=0.10)
    np.testing.assert_allclose(dlat, 0.0, atol=1e-8)


def test_numba_numpy_kernel_consistency():
    """numpy and numba backends should produce bitwise-identical results."""
    from parcels import Particle, ParticleSet

    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    depth = np.array([0.0, 1.5, 3.0, 5.0, 10.0])
    time = np.array([0.0])

    U_surface = 1.0
    U_data = np.zeros((1, len(depth), len(y), len(x)))
    for iz, d in enumerate(depth):
        U_data[0, iz, :, :] = U_surface * (1.0 - d / 10.0)
    V_data = np.zeros_like(U_data)

    DT = 60.0
    lon0, lat0 = 500.0, 500.0

    dd_np = DroguedDrifter(backend="numpy")
    fieldset_np = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    pset_np = ParticleSet(fieldset=fieldset_np, pclass=Particle, lon=[lon0], lat=[lat0], z=[0.0])
    pset_np.execute(kernels=[make_kernel(dd_np)], dt=DT, runtime=DT, verbose_progress=False)
    lon_np = float(np.asarray(pset_np.lon)[0])
    lat_np = float(np.asarray(pset_np.lat)[0])

    dd_nb = DroguedDrifter(backend="numba")
    fieldset_nb = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    pset_nb = ParticleSet(fieldset=fieldset_nb, pclass=Particle, lon=[lon0], lat=[lat0], z=[0.0])
    pset_nb.execute(kernels=[make_kernel(dd_nb)], dt=DT, runtime=DT, verbose_progress=False)
    lon_nb = float(np.asarray(pset_nb.lon)[0])
    lat_nb = float(np.asarray(pset_nb.lat)[0])

    np.testing.assert_allclose(lon_nb, lon_np, rtol=1e-10)
    np.testing.assert_allclose(lat_nb, lat_np, rtol=1e-10)


def test_invalid_backend():
    """DroguedDrifter should raise ValueError for an unrecognised backend."""
    with pytest.raises(ValueError):
        DroguedDrifter(backend="invalid")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_spar_buoy_wind_and_current_kernel(backend):
    """SparBuoySimple through Parcels with wind above and current below.

    Regression for signed-depth air sampling: the air column is encoded at
    *negative* ``depth`` and the coupling's ``depth_up = -depth[::-1]`` maps
    it onto the model's z-positive-up air levels.  So the emergent column must
    feel the wind (z > 0) and the hull the current (z <= 0).

    With a uniform 1 m/s eastward current, a uniform 10 m/s northward wind,
    and equal ``k_air``/``k_water``, the drag balance ``|v-u_water|(v-u_water)
    = -|v-u_air|(v-u_air)`` gives drift ``(0.5, 5.0)`` m/s.  If air sampling
    were broken (e.g. extrapolating the current into the air), the northward
    drift would vanish.
    """
    from parcels import Particle, ParticleSet

    U_current, V_wind = 1.0, 10.0
    x = np.linspace(0.0, 1000.0, 5)
    y = np.linspace(0.0, 1000.0, 5)
    # One signed axis: air at negative depth, water at positive depth.
    depth = np.concatenate(
        [np.linspace(-10.0, 0.0, 11, endpoint=False), np.linspace(0.0, 15.0, 8)]
    )
    time = np.array([0.0])

    Z = depth[None, :, None, None]
    ones = np.ones((1, len(depth), len(y), len(x)))
    U_data = np.where(Z < 0, 0.0, U_current) * ones
    V_data = np.where(Z < 0, V_wind, 0.0) * ones

    fieldset = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    spar = SparBuoySimple(backend=backend)

    pset = ParticleSet(
        fieldset=fieldset, pclass=Particle, lon=[500.0], lat=[500.0], z=[0.0]
    )
    DT = 60.0
    pset.execute(
        kernels=[make_kernel(spar)], dt=DT, runtime=DT, verbose_progress=False
    )

    drift_x = (float(np.asarray(pset.lon)[0]) - 500.0) / DT
    drift_y = (float(np.asarray(pset.lat)[0]) - 500.0) / DT

    assert drift_x > 0.0, "eastward drift from the water current"
    assert drift_y > 0.0, "northward drift from the wind (air must be sampled)"
    np.testing.assert_allclose([drift_x, drift_y], [0.5, 5.0], rtol=0.1)
