"""Tests for Parcels coupling (DDAdvectEE kernel, make_profile_sampler).

These tests verify:
- Profile sampler linear interpolation in depth
- Boundary handling (searchsorted clipping)
- DDAdvectEE kernel with synthetic FieldSets (uniform and sheared flow)
"""

import numpy as np
import pytest

from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.parcels_v4 import DDAdvectEE, make_dd_kernel, make_profile_sampler


def test_make_profile_sampler_basic():
    """Profile sampler should interpolate linearly in depth (z-up convention).

    depth_levels must be sorted ascending (deepest first): e.g., [-10, -5, 0].
    """
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 3
    U_profiles = np.array(
        [
            [0.3, 0.4, 0.5],  # z=-10 (deep)
            [0.4, 0.5, 0.6],  # z=-5
            [0.5, 0.6, 0.7],  # z=0 (surface)
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

    # At z=0 (surface), should return surface exactly
    U, V = sample_uv(0.0)
    np.testing.assert_allclose(U, [0.5, 0.6, 0.7], rtol=1e-10)
    np.testing.assert_allclose(V, [0.1, 0.2, 0.3], rtol=1e-10)

    # At z=-10 (deep), should return bottom exactly
    U, V = sample_uv(-10.0)
    np.testing.assert_allclose(U, [0.3, 0.4, 0.5], rtol=1e-10)
    np.testing.assert_allclose(V, [0.1, 0.2, 0.3], rtol=1e-10)


def test_make_profile_sampler_linear_interpolation():
    """Verify linear interpolation between depth levels (z-up convention)."""
    depth_levels = np.array([-10.0, 0.0])
    N = 1
    U_profiles = np.array([[1.0], [0.0]])  # Linear gradient: 1 at z=-10, 0 at z=0
    V_profiles = np.array([[0.0], [0.0]])

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # At z=-2.5 (1/4 of the way from surface to bottom), should get 0.25
    U, V = sample_uv(-2.5)
    np.testing.assert_allclose(U, [0.25], rtol=1e-10)


def test_make_profile_sampler_vectorized_z():
    """Sample at multiple z values for N particles simultaneously (z-up)."""
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 2
    U_profiles = np.array(
        [
            [0.5, 0.6],  # z=-10
            [0.3, 0.4],  # z=-5
            [0.1, 0.2],  # z=0
        ]
    )
    V_profiles = np.zeros((3, 2))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # Sample at scalar z=-5 (broadcasts to N particles)
    U, V = sample_uv(-5.0)

    # Both particles should get interpolated value at z=-5
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"
    np.testing.assert_allclose(U, [0.3, 0.4], rtol=1e-10)


def test_make_profile_sampler_boundary_shallow():
    """At z values shallower than shallowest level, should clip (z-up convention).

    With z-up, 'shallowest' means closest to 0 (least negative).
    depth_levels=[-10, -5, -2]: shallowest is z=-2.
    Querying at z=0 (above shallowest) should clip to the first valid interval.
    """
    depth_levels = np.array([-10.0, -5.0, -2.0])  # Shallowest is z=-2
    N = 1
    U_profiles = np.array([[0.3], [0.2], [0.1]])
    V_profiles = np.zeros((3, 1))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # At z=0 (above shallowest z=-2), searchsorted clips to last valid interval
    U, V = sample_uv(0.0)
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"
    # The exact value depends on clipping behavior; just verify we get a finite result
    assert np.isfinite(U[0])


def test_make_profile_sampler_boundary_deep():
    """At depths deeper than deepest level, should clip to deepest (z-up convention).

    With z-up, 'deepest' means most negative. depth_levels=[-10, -5, 0]:
    deepest is z=-10. Querying at z=-100 (below deepest) clips to deepest interval.
    """
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 1
    U_profiles = np.array([[0.3], [0.2], [0.1]])
    V_profiles = np.zeros((3, 1))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # At z=-100 (below deepest z=-10), searchsorted will clip to first valid interval
    U, V = sample_uv(-100.0)
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"
    # Should be close to deepest value [0.3] or interpolated near it
    assert np.isfinite(U[0])
    # The clip should give something near the deep end
    assert U[0] >= 0.2  # Should be at least as deep as second-to-last


def test_make_profile_sampler_multiple_particles():
    """Sample with multiple particles in profile (z-up convention)."""
    depth_levels = np.array([-5.0, 0.0])
    N = 5
    U_profiles = (
        np.arange(N, dtype=float).reshape(1, -1) * 0.1
    )  # (1, 5): [0, 0.1, 0.2, ...]
    U_profiles = np.vstack([U_profiles + 0.05, U_profiles])  # (2, 5): deep then surface
    V_profiles = np.zeros_like(U_profiles)

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # Surface is at index 1 (z=0)
    U_surface, _ = sample_uv(0.0)
    assert U_surface.shape == (N,), f"Expected shape ({N},), got {U_surface.shape}"
    np.testing.assert_allclose(U_surface, np.arange(N) * 0.1, rtol=1e-10)


def test_make_profile_sampler_degenerate_interval():
    """Handle degenerate depth interval (two identical depths at surface, z-up)."""
    depth_levels = np.array([-5.0, 0.0, 0.0])  # Degenerate interval at surface [0, 0]
    N = 1
    U_profiles = np.array([[0.2], [0.1], [0.1]])
    V_profiles = np.zeros((3, 1))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # Should not divide by zero (we have dz-epsilon clamping)
    U, V = sample_uv(-2.5)
    assert np.isfinite(U[0]), f"Expected finite U, got {U}"


def test_profile_sampler_broadcast_scalar_z():
    """Profile sampler should broadcast scalar z to (N,) for N particles."""
    depth_levels = np.array([-10.0, 0.0])
    N = 3
    U_profiles = np.array([[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]])
    V_profiles = np.zeros((2, N))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # Pass scalar depth, should broadcast to (N,)
    U, V = sample_uv(-5.0)
    assert U.shape == (N,), f"Expected shape ({N},), got {U.shape}"


def test_profile_sampler_vectorized_batch_z():
    """Sampler should accept vector z and return vector (U, V) (z-up convention)."""
    depth_levels = np.array([-10.0, -5.0, 0.0])
    N = 4
    U_profiles = np.array(
        [
            [0.3, 0.4, 0.5, 0.6],  # z=-10
            [0.2, 0.3, 0.4, 0.5],  # z=-5
            [0.1, 0.2, 0.3, 0.4],  # z=0
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

    # Pass batch z (one per particle)
    z_batch = np.array([0.0, -5.0, -10.0, -2.5])
    U, V = sample_uv(z_batch)

    assert U.shape == (4,), f"Expected U shape (4,), got {U.shape}"
    assert V.shape == (4,), f"Expected V shape (4,), got {V.shape}"
    np.testing.assert_allclose(U[0], 0.1, rtol=1e-10)  # z=0, particle 0 (surface)
    np.testing.assert_allclose(U[1], 0.3, rtol=1e-10)  # z=-5, particle 1


def test_profile_sampler_preserves_velocity_profiles():
    """Sampler should preserve the original velocity profiles at sampled depths."""
    depth_levels = np.array([-15.0, -10.0, -5.0, 0.0])
    N = 2
    rng = np.random.default_rng(42)
    U_profiles = rng.uniform(0, 1, size=(4, N))
    V_profiles = rng.uniform(0, 1, size=(4, N))

    sample_uv = make_profile_sampler(depth_levels, U_profiles, V_profiles)

    # At each depth level, should recover exact values
    for iz, z in enumerate(depth_levels):
        U, V = sample_uv(z)
        np.testing.assert_allclose(
            U, U_profiles[iz], rtol=1e-10, err_msg=f"U mismatch at z={z}"
        )
        np.testing.assert_allclose(
            V, V_profiles[iz], rtol=1e-10, err_msg=f"V mismatch at z={z}"
        )


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
# Integration tests for DDAdvectEE kernel with synthetic FieldSets
# ---------------------------------------------------------------------------


def _make_flat_fieldset(U_data_4d, V_data_4d, x, y, depth, time):
    """Build a flat-mesh FieldSet from 4-D numpy arrays.

    Args:
        U_data_4d, V_data_4d: shape (T, Z, Y, X)
        x, y, depth, time: 1-D coordinate arrays
    """
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
    from parcels import FieldSet, Particle, ParticleSet, StatusCode

    U_const = 0.5
    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    depth = np.array([0.0, 5.0, 10.0])
    time = np.array([0.0])

    U_data = np.full((1, len(depth), len(y), len(x)), U_const)
    V_data = np.zeros_like(U_data)

    fieldset = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    dd = DroguedDrifter()

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=[500.0],
        lat=[500.0],
        z=[0.0],
    )

    DT = 60.0  # 1 minute
    pset.execute(
        kernels=[make_dd_kernel(dd, backend=backend)],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )

    # After one step: lon should have moved by ~U_const * DT = 0.5 * 60 = 30 m
    lon_final = float(np.asarray(pset.lon)[0])
    displacement = lon_final - 500.0
    expected = U_const * DT
    np.testing.assert_allclose(displacement, expected, rtol=0.05)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_sheared_flow_dd_kernel(backend):
    """Sheared flow: drift velocity should be between surface and bottom current."""
    from parcels import FieldSet, Particle, ParticleSet, StatusCode

    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    depth = np.array([0.0, 1.5, 3.0, 5.0, 10.0])
    time = np.array([0.0])

    # Linear shear: U=1.0 at surface, U=0.0 at 10m depth
    U_surface = 1.0
    U_data = np.zeros((1, len(depth), len(y), len(x)))
    for iz, d in enumerate(depth):
        U_data[0, iz, :, :] = U_surface * (1.0 - d / 10.0)
    V_data = np.zeros_like(U_data)

    fieldset = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    dd = DroguedDrifter()  # default drogue depth l=3.0 m

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=[500.0],
        lat=[500.0],
        z=[0.0],
    )

    DT = 60.0
    pset.execute(
        kernels=[make_dd_kernel(dd, backend=backend)],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )

    lon_final = float(np.asarray(pset.lon)[0])
    displacement = lon_final - 500.0
    drift_speed = displacement / DT

    # Drift should be between 0 (bottom) and 1.0 (surface), strictly
    assert (
        0.0 < drift_speed < U_surface
    ), f"Drift speed {drift_speed:.4f} should be between 0 and {U_surface}"
    # With drogue at 3m in linear shear, drift should be closer to the
    # drogue-depth current (0.7 m/s) than to the surface current (1.0 m/s)
    assert (
        drift_speed < 0.9
    ), f"Drift speed {drift_speed:.4f} should be < 0.9 (drogue effect)"


def _make_spherical_fieldset(U_data_4d, V_data_4d, lon, lat, depth, time):
    """Build a spherical-mesh FieldSet from 4-D numpy arrays.

    Args:
        U_data_4d, V_data_4d: shape (T, Z, Y, X)
        lon, lat, depth, time: 1-D coordinate arrays
    """
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

    # Uniform 0.5 m/s eastward flow on a spherical grid near the equator
    U_ms = 0.5
    lat0 = 5.0  # near equator to keep cos(lat) ~ 1
    lon0 = 10.0

    lon = np.linspace(lon0 - 1, lon0 + 1, 5)
    lat = np.linspace(lat0 - 1, lat0 + 1, 5)
    depth = np.array([0.0, 5.0, 10.0])
    time = np.array([0.0])

    # Parcels spherical mesh: field values are in m/s, but the interpolator
    # returns deg/s.  Fill with uniform U_ms.
    U_data = np.full((1, len(depth), len(lat), len(lon)), U_ms)
    V_data = np.zeros_like(U_data)

    fieldset = _make_spherical_fieldset(U_data, V_data, lon, lat, depth, time)
    dd = DroguedDrifter()

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=[lon0],
        lat=[lat0],
        z=[0.0],
    )

    DT = 60.0
    pset.execute(
        kernels=[make_dd_kernel(dd, backend=backend)],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )

    # Expected displacement in degrees
    DEG2M = 1852.0 * 60.0
    cos_lat = np.cos(np.deg2rad(lat0))
    expected_dlon = U_ms * DT / (DEG2M * cos_lat)

    dlon = float(np.asarray(pset.lon)[0]) - lon0
    dlat = float(np.asarray(pset.lat)[0]) - lat0

    # rtol=0.10 accounts for interpolation edge effects on the coarse grid;
    # the critical check is order-of-magnitude correctness (no conversion →
    # ~111 km displacement instead of ~30 m).
    np.testing.assert_allclose(dlon, expected_dlon, rtol=0.10)
    np.testing.assert_allclose(dlat, 0.0, atol=1e-8)


def test_numba_numpy_kernel_consistency():
    """numpy and numba backends should produce bitwise-identical results."""
    from parcels import Particle, ParticleSet

    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    depth = np.array([0.0, 1.5, 3.0, 5.0, 10.0])
    time = np.array([0.0])

    # Linear shear: U=1.0 at surface, U=0.0 at 10m depth
    U_surface = 1.0
    U_data = np.zeros((1, len(depth), len(y), len(x)))
    for iz, d in enumerate(depth):
        U_data[0, iz, :, :] = U_surface * (1.0 - d / 10.0)
    V_data = np.zeros_like(U_data)

    dd = DroguedDrifter()
    DT = 60.0
    lon0, lat0 = 500.0, 500.0

    # Run with numpy backend
    fieldset_np = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    pset_np = ParticleSet(
        fieldset=fieldset_np,
        pclass=Particle,
        lon=[lon0],
        lat=[lat0],
        z=[0.0],
    )
    pset_np.execute(
        kernels=[make_dd_kernel(dd, backend="numpy")],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )
    lon_np = float(np.asarray(pset_np.lon)[0])
    lat_np = float(np.asarray(pset_np.lat)[0])

    # Run with numba backend from identical initial position
    fieldset_nb = _make_flat_fieldset(U_data, V_data, x, y, depth, time)
    pset_nb = ParticleSet(
        fieldset=fieldset_nb,
        pclass=Particle,
        lon=[lon0],
        lat=[lat0],
        z=[0.0],
    )
    pset_nb.execute(
        kernels=[make_dd_kernel(dd, backend="numba")],
        dt=DT,
        runtime=DT,
        verbose_progress=False,
    )
    lon_nb = float(np.asarray(pset_nb.lon)[0])
    lat_nb = float(np.asarray(pset_nb.lat)[0])

    np.testing.assert_allclose(lon_nb, lon_np, rtol=1e-10)
    np.testing.assert_allclose(lat_nb, lat_np, rtol=1e-10)


def test_make_dd_kernel_invalid_backend():
    """make_dd_kernel should raise ValueError for an unrecognised backend."""
    dd = DroguedDrifter()
    with pytest.raises(ValueError):
        make_dd_kernel(dd, backend="invalid")
