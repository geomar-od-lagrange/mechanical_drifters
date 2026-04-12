"""Shared test fixtures and constants."""

import pytest

from drogued_drifters.models.drogued_drifter import DroguedDrifter, DrifterPhysics, EOMState

DEFAULT_PHYSICS = DrifterPhysics(
    m_b=1.0,
    m_d=2.7,
    m_hat_d=1.0,
    m_tilde_d=101.0,
    m_tilde_b=1.9,
    l=3.0,
    g=9.81,
    k_b=12.0,
    k_d=154.0,
)


@pytest.fixture
def dd():
    """Return a DroguedDrifter with default parameters."""
    return DroguedDrifter()


@pytest.fixture
def default_eom_state():
    """Return a simple equilibrium EOMState (drogue hanging straight down, at rest).

    All stereographic coordinates are zero (equilibrium = drogue pointing down),
    velocities are zero, and currents are zero.  This is the simplest valid state
    for testing EOM callables without physical forcing.
    """
    return EOMState(
        u_stereo=0.0,
        v_stereo=0.0,
        xd=0.0,
        yd=0.0,
        ud_stereo=0.0,
        vd_stereo=0.0,
        U_b=0.0,
        V_b=0.0,
        U_d=0.0,
        V_d=0.0,
    )
