"""Tests for the SparBuoySimple model."""

import numpy as np

from mechanical_drifters.base import LagrangianMechanicsModel
from mechanical_drifters.models.spar_buoy_simple import (
    SparBuoySimple,
    SparBuoyPhysics,
    SparBuoyState,
)


# ---------------------------------------------------------------------------
# Structural / static tests
# ---------------------------------------------------------------------------


def test_class_exists():
    """SparBuoySimple is a subclass of LagrangianMechanicsModel."""
    assert issubclass(SparBuoySimple, LagrangianMechanicsModel)


def test_physics_fields_and_defaults():
    """SparBuoyPhysics fields and default values."""
    assert SparBuoyPhysics._fields == (
        "m", "m_tilde", "k_air", "k_water", "draft", "height_air", "n_air", "n_water",
    )
    assert SparBuoyPhysics() == (1.0, 1.0, 10.0, 10.0, 10.0, 9.0, 3, 4)


def test_invalid_level_counts_rejected():
    """n_air/n_water are fixed by the State layout; other values fail loudly."""
    import pytest
    with pytest.raises(ValueError):
        SparBuoySimple(SparBuoyPhysics(n_air=5))


def test_custom_physics_round_trips():
    """Constructs with explicit SparBuoyPhysics; fields are kept."""
    phys = SparBuoyPhysics(k_air=20.0, k_water=30.0, draft=5.0)
    sbd = SparBuoySimple(physics=phys)
    assert sbd.physics == phys


def test_state_field_names():
    """State is xd, yd then per-level currents for n_air air + n_water water levels."""
    expected = ["xd", "yd"]
    for i in range(SparBuoyPhysics().n_air):
        expected += [f"U_air_{i}", f"V_air_{i}"]
    for i in range(SparBuoyPhysics().n_water):
        expected += [f"U_water_{i}", f"V_water_{i}"]
    assert list(SparBuoyState._fields) == expected


def test_n_q_and_state_size():
    """Two generalized coordinates; integrated state is [x, y, xd, yd]."""
    assert SparBuoySimple.n_q == 2
    assert SparBuoySimple().state_size == 4


def test_max_depth_tracks_draft():
    """_max_depth equals the draft (so Parcels samples the full hull)."""
    assert SparBuoySimple(SparBuoyPhysics(draft=7.5))._max_depth == 7.5


def test_water_sampled_below_air_above():
    """z-sign convention: air levels are queried above the surface (z>0), water
    levels at/below it (z<=0), reaching the hull tip at -draft.

    sample_uv(z) is z-positive-up, so the submerged hull must be sampled at
    non-positive z. Pins the sign directly via the z values _rhs_batch queries;
    test_sign_regression_water_vs_calm_air is the behavioural guard.
    """
    sbd = SparBuoySimple()  # draft=10, height_air=9
    seen = []

    def recording_uv(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        seen.append(float(z[0]))
        return np.zeros(len(z)), np.zeros(len(z))

    sbd._rhs_batch(np.zeros((1, sbd.state_size)), recording_uv)
    air, water = seen[: sbd.physics.n_air], seen[sbd.physics.n_air:]
    assert all(z > 0 for z in air), "air levels must be sampled above the surface"
    assert all(z <= 0 for z in water), "water levels must not be sampled in the air"
    assert min(water) == -sbd.physics.draft, "deepest water level is the hull tip"


def test_drift_velocity():
    """drift_velocity extracts [xd, yd] from the state array."""
    Y = np.array([[0.0, 0.0, 0.3, -0.1]])
    np.testing.assert_allclose(SparBuoySimple().drift_velocity(Y), [[0.3, -0.1]])


def test_derive_symbolic_returns_correct_shapes():
    """_derive_symbolic returns (M 2x2, F 2x1, args) with one arg per field."""
    M, F, args = SparBuoySimple()._derive_symbolic()
    assert M.shape == (2, 2)
    assert F.shape == (2, 1)
    assert len(args) == len(SparBuoyPhysics._fields) + len(SparBuoyState._fields)


# ---------------------------------------------------------------------------
# EOM evaluator tests
# ---------------------------------------------------------------------------


def _uniform_state(xd=0.0, yd=0.0, u_water=0.0, v_water=0.0, u_air=0.0, v_air=0.0):
    """SparBuoyState with one current in the air levels and one in the water levels."""
    fields = {"xd": xd, "yd": yd}
    for i in range(SparBuoyPhysics().n_air):
        fields[f"U_air_{i}"], fields[f"V_air_{i}"] = u_air, v_air
    for i in range(SparBuoyPhysics().n_water):
        fields[f"U_water_{i}"], fields[f"V_water_{i}"] = u_water, v_water
    return SparBuoyState(**fields)


def test_qdd_drag_direction():
    """Buoy at rest in an eastward water current: x-acceleration > 0, y == 0."""
    from mechanical_drifters.eom import get_eom_callables

    sbd = SparBuoySimple()
    qdd_func = get_eom_callables(sbd, "numpy")[0]
    qdd = qdd_func(sbd.physics, _uniform_state(u_water=0.5), batch=False)
    assert qdd.shape == (2,)
    assert qdd[0] > 0, "water drags the buoy eastward"
    assert abs(qdd[1]) < 1e-12, "no northward forcing"


def test_qdd_zero_relative_velocity():
    """Drift equals the current everywhere: drag is zero, so qdd == 0."""
    from mechanical_drifters.eom import get_eom_callables

    sbd = SparBuoySimple()
    qdd_func = get_eom_callables(sbd, "numpy")[0]
    state = _uniform_state(xd=0.3, yd=0.1, u_water=0.3, v_water=0.1, u_air=0.3, v_air=0.1)
    qdd = qdd_func(sbd.physics, state, batch=False)
    np.testing.assert_allclose(qdd, [0.0, 0.0], atol=1e-12)


def test_make_kernel_creates_callable():
    """make_kernel returns a callable for SparBuoySimple."""
    from mechanical_drifters.parcels import make_kernel

    assert callable(make_kernel(SparBuoySimple()))


# ---------------------------------------------------------------------------
# Behaviour / integration tests
# ---------------------------------------------------------------------------


def test_steady_state_uniform_flow():
    """Uniform flow (same U, V at every z): steady-state drift == that current."""
    sbd = SparBuoySimple(SparBuoyPhysics(k_air=10000.0, k_water=10000.0))
    U_const, V_const = 0.3, -0.1

    def sample_uv(z):
        N = len(np.atleast_1d(z))
        return np.full(N, U_const), np.full(N, V_const)

    t, Y, _ = sbd.integrate(sample_uv, t_span=(0, 1200))
    np.testing.assert_allclose(sbd.drift_velocity(Y[-1])[0], [U_const, V_const], rtol=1e-4)


def test_steady_state_zero_flow():
    """Zero flow everywhere: steady-state drift is zero."""
    def sample_uv(z):
        N = len(np.atleast_1d(z))
        return np.zeros(N), np.zeros(N)

    t, Y, _ = SparBuoySimple().integrate(sample_uv, t_span=(0, 1200))
    np.testing.assert_allclose(SparBuoySimple().drift_velocity(Y[-1])[0], [0.0, 0.0], atol=1e-10)


def test_sign_regression_water_vs_calm_air():
    """Water=(1,0), calm air: steady drift must be ~0.5 east, not ~0.27.

    With equal k_air=k_water the force balance is k_water(1-v)^2 = k_air v^2,
    i.e. (1-v)=v => v=0.5. If the submerged hull were sampled at positive z
    (in the air) the balance shifts toward the calm air and gives ~0.27, so
    this pins the water/air z-sign behaviourally.
    """
    def sample_uv(z):
        z = np.atleast_1d(z)
        return np.where(z <= 0, 1.0, 0.0), np.zeros_like(z, dtype=float)

    t, Y, _ = SparBuoySimple().integrate(sample_uv, t_span=(0, 1200))
    drift = SparBuoySimple().drift_velocity(Y[-1])[0]
    np.testing.assert_allclose(drift[0], 0.5, rtol=1e-2,
                               err_msg="~0.27 means the hull is sampled in air (sign bug)")
    np.testing.assert_allclose(drift[1], 0.0, atol=1e-3)
