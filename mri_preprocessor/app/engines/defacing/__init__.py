"""
Defacing engine family (PR11).

Engines:
  placeholder -- a built-in pure-Python naive proxy. Blurs the inferior
                 portion of the volume. NOT real defacing -- just enough
                 to validate the dispatcher pattern in test environments
                 without external tools. Maturity: experimental.

  pydeface    -- the popular pydeface Python package. Real defacing using
                 a face template registration. Requires separate
                 installation + an FSL license (pydeface shells out to
                 FSL tools). Maturity: external_engine. Stub in PR11 --
                 real subprocess wrapper would be a follow-up PR.

  mri_deface  -- FreeSurfer's defacing tool. Real defacing via an atlas-
                 based registration. Requires FreeSurfer installation.
                 Maturity: external_engine. Stub in PR11.

Same pattern as brain_extraction: dispatcher in steps_privacy.py reads
params["engine"] (default "placeholder") and looks up the engine in
DEFACING_ENGINES. Adding a new defacing engine is a 1-file change to
this directory.
"""
from app.engines.base import EngineRegistry
from app.engines.defacing.placeholder import PlaceholderDefacing
from app.engines.defacing.pydeface import PyDefaceEngine
from app.engines.defacing.mri_deface import MRIDefaceEngine


DEFACING_ENGINES = EngineRegistry("defacing")

# Registration order: built-in placeholder first (so it's the default),
# then external engines.
DEFACING_ENGINES.register(PlaceholderDefacing())
DEFACING_ENGINES.register(PyDefaceEngine())
DEFACING_ENGINES.register(MRIDefaceEngine())


def available_engines():
    """Returns a JSON-serializable list of engine metadata for this family."""
    return DEFACING_ENGINES.available_engines()
