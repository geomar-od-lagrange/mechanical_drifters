# Primary model
from .models.drogued_drifter import (
    DroguedDrifter,
    DrifterPhysics,
    EOMState,
    drogue_horizontal_added_mass,
    buoy_horizontal_added_mass,
    drogue_horizontal_drag_coeff,
    buoy_horizontal_drag_coeff,
)

# Point surface drifter model
from .models.point_surface_drifter import (
    PointSurfaceDrifter,
    PointSurfacePhysics,
    PointSurfaceState,
)

# Base class (for building new models)
from .base import LagrangianMechanicsModel

# EOM evaluation (takes model instance as first argument)
from .eom import eval_qdd, eval_M, eval_F

# Utilities
from .stokes import compute_stokes_profile

