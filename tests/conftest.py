"""Shared test fixtures and constants."""

from drogued_drifters.lagrange_model import DrifterPhysics

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
