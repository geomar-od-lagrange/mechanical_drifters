"""Tests for the PointSurfaceDrifter model."""

import numpy as np
import pytest

from mechanical_drifters.base import LagrangianMechanicsModel
from mechanical_drifters.models.point_surface_drifter import (
    PointSurfaceDrifter,
    PointSurfacePhysics,
    PointSurfaceState,
)


def test_class_exists():
    """PointSurfaceDrifter is importable and is a subclass of LagrangianMechanicsModel."""
    assert issubclass(PointSurfaceDrifter, LagrangianMechanicsModel)


def test_physics_namedtuple():
    """PointSurfacePhysics has fields m, m_tilde, k (in that order)."""
    assert PointSurfacePhysics._fields == ("m", "m_tilde", "k")


def test_state_namedtuple():
    """PointSurfaceState has fields xd, yd, U, V (in that order)."""
    assert PointSurfaceState._fields == ("xd", "yd", "U", "V")


def test_n_q():
    """PointSurfaceDrifter.n_q == 2."""
    assert PointSurfaceDrifter.n_q == 2


def test_drift_velocity():
    """PointSurfaceDrifter.drift_velocity extracts xd, yd from state."""
    psd = PointSurfaceDrifter()
    import numpy as np
    Y = np.array([[0.0, 0.0, 0.3, -0.1]])
    vel = psd.drift_velocity(Y)
    np.testing.assert_allclose(vel, [[0.3, -0.1]])


def test_state_size():
    """Instance state_size == 4."""
    psd = PointSurfaceDrifter()
    assert psd.state_size == 4


def test_default_physics():
    """Constructs with no args; physics has sensible positive values for all fields."""
    psd = PointSurfaceDrifter()
    assert psd.physics.m > 0
    assert psd.physics.m_tilde > 0
    assert psd.physics.k > 0


def test_custom_physics():
    """Constructs with explicit PointSurfacePhysics; fields match."""
    phys = PointSurfacePhysics(m=2.0, m_tilde=5.0, k=10.0)
    psd = PointSurfaceDrifter(physics=phys)
    assert psd.physics.m == 2.0
    assert psd.physics.m_tilde == 5.0
    assert psd.physics.k == 10.0


def test_is_lagrangian_model():
    """Instance is a LagrangianMechanicsModel."""
    psd = PointSurfaceDrifter()
    assert isinstance(psd, LagrangianMechanicsModel)


# ---------------------------------------------------------------------------
# Behavior tests (Red phase 2)
# ---------------------------------------------------------------------------


def test_derive_symbolic_returns_correct_shapes():
    """_derive_symbolic returns (M_2x2, F_2x1, args_tuple)."""
    psd = PointSurfaceDrifter()
    M, F, args = psd._derive_symbolic()
    assert M.shape == (2, 2)
    assert F.shape == (2, 1)
    assert len(args) == len(PointSurfacePhysics._fields) + len(PointSurfaceState._fields)


def test_qdd_scalar():
    """qdd evaluator returns (2,) array for scalar input."""
    from mechanical_drifters.eom import _make_qdd_func

    psd = PointSurfaceDrifter()
    _qdd = _make_qdd_func(psd, "numpy")
    state = PointSurfaceState(xd=0.0, yd=0.0, U=0.5, V=0.0)
    qdd = _qdd(psd.physics, state)
    assert qdd.shape == (2,)


def test_qdd_drag_direction():
    """Particle at rest in eastward current: drag accelerates it eastward."""
    from mechanical_drifters.eom import _make_qdd_func

    psd = PointSurfaceDrifter()
    _qdd = _make_qdd_func(psd, "numpy")
    state = PointSurfaceState(xd=0.0, yd=0.0, U=0.5, V=0.0)
    qdd = _qdd(psd.physics, state)
    assert qdd[0] > 0, "x-acceleration should be positive (toward eastward current)"
    assert abs(qdd[1]) < 1e-12, "y-acceleration should be zero (no north current)"


def test_qdd_zero_relative_velocity():
    """When drift == current, drag is zero, so qdd == 0."""
    from mechanical_drifters.eom import _make_qdd_func

    psd = PointSurfaceDrifter()
    _qdd = _make_qdd_func(psd, "numpy")
    state = PointSurfaceState(xd=0.3, yd=0.1, U=0.3, V=0.1)
    qdd = _qdd(psd.physics, state)
    np.testing.assert_allclose(qdd, [0.0, 0.0], atol=1e-12)


def test_steady_state_uniform_flow():
    """Uniform flow: steady-state drift == surface current.

    Quadratic drag converges algebraically (O(1/t)), so we use high drag
    and long integration to get close, then check with rtol=1e-4.
    """
    psd = PointSurfaceDrifter(
        physics=PointSurfacePhysics(m=1.0, m_tilde=1.0, k=10000.0),
    )

    U_const, V_const = 0.3, -0.1

    def sample_uv(z):
        N = len(np.atleast_1d(z))
        return np.full(N, U_const), np.full(N, V_const)

    t, Y, max_accel = psd.integrate(
        sample_uv, t_span=(0, 600),
    )

    drift_vel = psd.drift_velocity(Y[-1])
    np.testing.assert_allclose(drift_vel[0, 0], U_const, rtol=1e-4)
    np.testing.assert_allclose(drift_vel[0, 1], V_const, rtol=1e-4)


def test_steady_state_zero_flow():
    """Zero flow: steady-state drift is zero."""
    psd = PointSurfaceDrifter()

    def sample_uv(z):
        N = len(np.atleast_1d(z))
        return np.zeros(N), np.zeros(N)

    t, Y, max_accel = psd.integrate(sample_uv)

    drift_vel = psd.drift_velocity(Y[-1])
    np.testing.assert_allclose(drift_vel[0], [0.0, 0.0], atol=1e-10)


def test_steady_state_multiple_particles():
    """Multiple particles converge to their respective surface currents."""
    psd = PointSurfaceDrifter(
        physics=PointSurfacePhysics(m=1.0, m_tilde=1.0, k=10000.0),
    )
    N = 3

    U_vals = np.array([0.1, 0.3, 0.5])
    V_vals = np.array([0.05, -0.1, 0.0])

    def sample_uv(z):
        return U_vals.copy(), V_vals.copy()

    t, Y, max_accel = psd.integrate(
        sample_uv, t_span=(0, 600),
    )

    drift_vel = psd.drift_velocity(Y[-1])
    assert drift_vel.shape == (N, 2)
    np.testing.assert_allclose(drift_vel[:, 0], U_vals, rtol=1e-4)
    np.testing.assert_allclose(drift_vel[:, 1], V_vals, rtol=1e-4)


def test_make_kernel_creates_callable():
    """make_kernel should return a callable."""
    from mechanical_drifters.parcels import make_kernel
    psd = PointSurfaceDrifter()
    kernel = make_kernel(psd)
    assert callable(kernel)


def test_no_max_depth():
    """Surface-only model: no _max_depth method."""
    psd = PointSurfaceDrifter()
    assert not hasattr(psd, '_max_depth')
