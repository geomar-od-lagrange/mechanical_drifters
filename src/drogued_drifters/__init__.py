from .drifter import DroguedDrifter

try:
    from ._generated_eom import compute_F, compute_M
except ImportError:
    def _not_generated(*args, **kwargs):
        raise ImportError(
            "The generated EOM module is missing. "
            "Run: pixi run python scripts/generate_eom.py"
        )
    compute_M = _not_generated
    compute_F = _not_generated
