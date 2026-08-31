"""
fMRI motion correction engine family.

Engines:
  simpleitk  -- the existing reference implementation (SimpleITK rigid
                registration of every volume to a reference frame).
                Pure Python, works out of the box.
                Maturity: reference.

  mcflirt    -- FSL MCFLIRT. Faster, more validated, widely used in
                published research. NOT IMPLEMENTED in PR5 -- stub
                documenting where the integration goes.
                Maturity: external_engine (planned).

Note: simpleitk is the FIRST engine registered, so it's the default.
Users wanting MCFLIRT pass `params["engine"] = "mcflirt"` once the
real wrapper is implemented.
"""
from app.engines.base import EngineRegistry
from app.engines.fmri_motion.simpleitk import SimpleITKMotionCorrection
from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection


FMRI_MOTION_ENGINES = EngineRegistry("fmri_motion")

# Registration order: built-in pure-Python first, so the default is
# always the one that works without external dependencies.
FMRI_MOTION_ENGINES.register(SimpleITKMotionCorrection())
FMRI_MOTION_ENGINES.register(MCFLIRTMotionCorrection())


def available_engines():
    """Returns a JSON-serializable list of engine metadata for this family."""
    return FMRI_MOTION_ENGINES.available_engines()
