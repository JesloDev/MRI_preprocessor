"""
Engine adapter layer (PR5).

Selective abstraction: only used where there are GENUINELY multiple
implementations of a step. Steps with a single sensible implementation
(reorientation, intensity_normalization, resampling, ...) stay as
plain step functions -- premature abstraction is avoided.

Three engine families ship in PR5:
    brain_extraction/   -- Otsu (current placeholder), SynthStrip, HD-BET
    fmri_motion/        -- SimpleITK (current), MCFLIRT (planned)
    dwi_correction/     -- DIPY (current), FSL eddy (planned)

Adding a new engine to an existing family is a one-file change:
implement the ProcessingEngine Protocol and register it in the
family's __init__.py. No step code, no API change, no resolver change.

The Protocol contract is identical to the step-function contract:
    run(input_path, output_dir, params, context) -> dict

so step functions become thin dispatchers:

    def brain_extraction(input_path, output_dir, params, context):
        engine_name = params.get("engine", "otsu")
        return BRAIN_EXTRACTION_ENGINES[engine_name].run(
            input_path, output_dir, params, context,
        )

Each family also exposes `available_engines()` so the API/UI can list
which engines are selectable for a given step.
"""
from app.engines.base import ProcessingEngine, EngineRegistry
from app.engines.brain_extraction import (
    BRAIN_EXTRACTION_ENGINES,
    available_engines as available_brain_extraction_engines,
)
from app.engines.fmri_motion import (
    FMRI_MOTION_ENGINES,
    available_engines as available_fmri_motion_engines,
)
from app.engines.dwi_correction import (
    DWI_CORRECTION_ENGINES,
    available_engines as available_dwi_correction_engines,
)

__all__ = [
    "ProcessingEngine",
    "EngineRegistry",
    "BRAIN_EXTRACTION_ENGINES",
    "FMRI_MOTION_ENGINES",
    "DWI_CORRECTION_ENGINES",
    "available_brain_extraction_engines",
    "available_fmri_motion_engines",
    "available_dwi_correction_engines",
]
