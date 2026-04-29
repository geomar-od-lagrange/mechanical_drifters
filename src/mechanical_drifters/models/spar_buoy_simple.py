"""SparBuoyDrifter model: spar buoy at the surface with two quadratic drag forces (surface/water)."""

from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel


class SparBuoyPhysics(NamedTuple):
    """Physical constants for a spar buoy.

    Default values: ``SparBuoyPhysics()`` gives m=1, m_tilde=1, k_surface=10, k_water=10, draft=7, n_z=7.
    """

    m: float = 1.0  # mass [kg]
    m_tilde: float = 1.0  # added mass [kg]
    k_surface: float = 10.0  # drag coefficient (surface) [kg/m]
    k_water: float = 10.0 # drag coefficient (water) [kg/m]
    draft: float = 7.0 # draft of the buoy
    n_z: int = 7 # sampling rate under water


class SparBuoyState(NamedTuple):
    """Per-timestep state variables and forcing."""

    xd: float | np.ndarray  # eastward drift velocity [m/s]
    yd: float | np.ndarray  # northward drift velocity [m/s]
    U: float | np.ndarray  # current at surface, east [m/s]
    V: float | np.ndarray  # current at surface, north [m/s]
    Fx_water: float | np.ndarray
    Fy_water: float | np.ndarray


# State vector layout: [x, y, xd, yd]
IX, IY, IXD, IYD = range(4)


class SparBuoyDrifter(LagrangianMechanicsModel):
    """A spar buoy with 7m draft.

    Two generalized coordinates (x, y). The Lagrangian is
    L = 1/2 (m + m_tilde)(xd^2 + yd^2), drag force F = -k|v-u|(v-u).

    State vector: [x, y, xd, yd]
    """

    Physics = SparBuoyPhysics
    State = SparBuoyState
    n_q = 2
    state_names = ("x", "y", "xd", "yd")

    def __init__(self, physics=SparBuoyPhysics(), *, backend="numpy"):
        super().__init__(physics, backend=backend)

    def _derive_symbolic(self):
        """Derive M, F for a spar buoy with two quadratic drag forces.

        Lagrangian: L = 1/2 (m + m_tilde)(xd^2 + yd^2)
        No potential energy (surface-constrained particle).
        Drag: F_drag = -k |v - u| (v - u) where v = (xd, yd), u = (U, V).
        """
        t = dynamicsymbols._t
        x, y = dynamicsymbols("x y")

        m = sp.Symbol("m", positive=True)
        m_tilde = sp.Symbol("m_tilde", positive=True)
        
        k_surface = sp.Symbol("k_surface", positive=True)
        k_water = sp.Symbol("k_water", positive=True)
        draft = sp.Symbol("draft", positive=True)
        n_z = sp.Symbol("n_z", positive=True)
        
        U, V = sp.symbols("U V", real=True)
        Fx_water, Fy_water = sp.symbols("Fx_water Fy_water", real = True)

        q = sp.Matrix([x, y])
        qd = q.diff(t)

        # Lagrangian: pure kinetic energy
        L = sp.Rational(1, 2) * (m + m_tilde) * qd.dot(qd)

        # Velocity and current vectors
        v = sp.Matrix([qd[0], qd[1]])
        u = sp.Matrix([U, V])
        rel = v - u
        rel_speed = sp.sqrt(rel.dot(rel))

        F_surface = -k_surface * rel_speed * rel

        # Drag under water
        F_water = sp.Matrix([Fx_water, Fy_water])

        # Total generalized force
        Q = F_surface + F_water

        # Euler-Lagrange: d/dt(dL/dqd) - dL/dq = Q
        # => (m + m_tilde) * qdd = Q
        qdd = qd.diff(t)
        eoms = sp.Matrix([
            L.diff(qdj).diff(t) - L.diff(qj) - Qj
            for qj, qdj, Qj in zip(q, qd, Q)
        ])

        M_sym, F_sym = sp.linear_eq_to_matrix(eoms, list(qdd))

        # Substitute dynamic symbols with static ones for lambdify
        xd_dyn, yd_dyn = x.diff(t), y.diff(t)
        xd_static = sp.Symbol("xd", real=True)
        yd_static = sp.Symbol("yd", real=True)

        subs = {xd_dyn: xd_static, yd_dyn: yd_static}
        M_static = M_sym.subs(subs)
        F_static = F_sym.subs(subs)

        symbol_map = {"m": m, "m_tilde": m_tilde, "k_surface": k_surface,
                      "k_water": k_water, "draft": draft, "n_z": n_z, "xd":
                      xd_static, "yd": yd_static, "U": U, "V": V,
                      "Fx_water": Fx_water, "Fy_water": Fy_water
        }
        all_fields = (list(SparBuoyPhysics._fields) +
                      list(SparBuoyState._fields))
        args = tuple(symbol_map[field] for field in all_fields)

        return M_static, F_static, args

    def _rhs_batch(self, Y, sample_uv):
        """Compute dY/dt for N particles.

        Args:
            Y: ``(N, state_size)`` state array [x, y, xd, yd].
            sample_uv: callable(z) -> (U, V) for (N,) arrays.

        Returns:
            dY: ``(N, state_size)`` derivatives.
        """
        N = Y.shape[0]
        xd = Y[:, IXD]
        yd = Y[:, IYD]

        # Oberfläche
        U, V = sample_uv(np.zeros(N))

        # perfekt senkrechter Zylinder: Tiefen von Oberfläche bis draft
        z_levels = np.linspace(0.0, -self.physics.draft,
                               int(self.physics.n_z))

        Fx_levels = []
        Fy_levels = []

        for z in z_levels:
            Uz, Vz = sample_uv(np.full(N, z))

            rel_x = xd - Uz
            rel_y = yd - Vz
            rel_speed = np.sqrt(rel_x**2 + rel_y**2)

            Fx_levels.append(-self.physics.k_water * rel_speed * rel_x)
            Fy_levels.append(-self.physics.k_water * rel_speed * rel_y)

        Fx_water = np.mean(Fx_levels, axis=0)
        Fy_water = np.mean(Fy_levels, axis=0)
    
        state = PointSurfaceState(xd=xd, yd=yd, U=U, V=V, Fx_water=Fx_water,
                              Fy_water=Fy_water,)

        qdd = self._qdd_func(self.physics, state, batch=True)

        bad = ~np.isfinite(qdd).all(axis=1)
        if np.any(bad):
            qdd[bad] = 0.0

        n_q = self.n_q
        dY = np.empty_like(Y)
        dY[:, :n_q] = Y[:, n_q:]   # d/dt(q) = qd  (kinematic identity)
        dY[:, n_q:] = qdd          # d/dt(qd) = qdd
        return dY

    def drift_velocity(self, Y):
        """Extract drift velocity from state array.

        Args:
            Y: State array, shape ``(N, state_size)``.

        Returns:
            Drift velocity array, shape ``(N, 2)``.
        """
        return Y[:, [IXD, IYD]]

