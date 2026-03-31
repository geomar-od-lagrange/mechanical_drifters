from .drifter import DroguedDrifter
from .lagrange_model import M_func, F_func

# For backward compatibility, alias to the new API
compute_M = M_func
compute_F = F_func
