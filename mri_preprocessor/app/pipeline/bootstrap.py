"""
Importing this module triggers self-registration of every built-in
step-implementation module (via the @step decorator's side effects).
Import this once, early, before anything queries the registry.
"""
from app.pipeline import steps_structural  # noqa: F401
from app.pipeline import steps_diffusion  # noqa: F401
from app.pipeline import steps_functional  # noqa: F401
from app.pipeline import steps_privacy  # noqa: F401 -- PR11: defacing, anonymization, intensity_filtering
