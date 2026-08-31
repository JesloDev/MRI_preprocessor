"""
DWI motion / eddy-current correction engine family.

Engines:
  dipy       -- the existing reference implementation (DIPY affine
                registration of each diffusion volume to the mean b0).
                Pure Python, works out of the box.
                Maturity: reference.

  fsl_eddy   -- FSL's eddy + topup. Joint model of susceptibility and
                eddy-current distortions. The validated standard for
                production diffusion preprocessing. NOT IMPLEMENTED in
                PR5 -- stub.
                Maturity: external_engine (planned).
"""
from app.engines.base import EngineRegistry
from app.engines.dwi_correction.dipy import DIPYEddyCorrection
from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection


DWI_CORRECTION_ENGINES = EngineRegistry("dwi_correction")

DWI_CORRECTION_ENGINES.register(DIPYEddyCorrection())
DWI_CORRECTION_ENGINES.register(FSLEddyCorrection())


def available_engines():
    """Returns a JSON-serializable list of engine metadata for this family."""
    return DWI_CORRECTION_ENGINES.available_engines()
