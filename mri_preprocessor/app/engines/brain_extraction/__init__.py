"""
Brain extraction engine family.

Engines:
  otsu        -- the existing placeholder (SimpleITK Otsu + largest CC).
                Pure Python, works out of the box, but unvalidated.
                Maturity: experimental.

  synthstrip  -- FreeSurfer's SynthStrip (deep-learning skull strip).
                NOT IMPLEMENTED in PR5 -- this is a stub that documents
                the integration surface. When implemented, this file
                will shell out to the SynthStrip binary; the rest of
                the platform (registry, resolver, dispatcher, API)
                already supports it via the params["engine"] knob.
                Maturity: external_engine (planned).

  hdbet       -- HD-BET (Deep learning Brain Extraction Tool).
                NOT IMPLEMENTED in PR5 -- stub, same pattern as SynthStrip.
                Maturity: external_engine (planned).

Adding the real SynthStrip / HD-BET wrappers later is a 1-file change:
implement the ProcessingEngine Protocol here, register it in
BRAIN_EXTRACTION_ENGINES, and the brain_extraction step automatically
gains the new engine option. No other code in the platform needs to
change.
"""
from app.engines.base import EngineRegistry
from app.engines.brain_extraction.otsu import OtsuBrainExtraction
from app.engines.brain_extraction.synthstrip import SynthStripBrainExtraction
from app.engines.brain_extraction.hdbet import HDBETBrainExtraction


# Family-level registry. The brain_extraction step function reads
# params["engine"] (default "otsu") and looks up the engine here.
BRAIN_EXTRACTION_ENGINES = EngineRegistry("brain_extraction")

# IMPORTANT: registration order determines the default.
# Built-in pure-Python engines (otsu) come first, so the default is
# always the one that works without external dependencies.
BRAIN_EXTRACTION_ENGINES.register(OtsuBrainExtraction())
BRAIN_EXTRACTION_ENGINES.register(SynthStripBrainExtraction())
BRAIN_EXTRACTION_ENGINES.register(HDBETBrainExtraction())


def available_engines():
    """Returns a JSON-serializable list of engine metadata for this family."""
    return BRAIN_EXTRACTION_ENGINES.available_engines()
