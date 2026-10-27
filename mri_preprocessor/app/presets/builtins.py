"""
Built-in presets shipped with the platform (PR10).

Each built-in is a named configuration that references existing
processing steps. NO PROCESSING LOGIC IS DUPLICATED HERE -- the steps
are defined in app/pipeline/steps_*.py and discovered via the
registry. A preset is just a tuple of step_ids + step_params + a
human-readable description.

The 4 built-ins cover the common pipeline shapes:

    basic-qc            - Just enough to verify data integrity.
                          Runs reorientation only. Useful as a
                          sanity-check before committing to a longer
                          pipeline.

    standard            - The default structural MRI pipeline. Suitable
                          for T1/T2/FLAIR/PD processing when you don't
                          have a specific downstream analysis in mind.
                          Produces bias-corrected, skull-stripped,
                          resampled, intensity-normalized volumes.

    ml-ready            - Cross-modality preprocessing for ML workflows.
                          Same as standard for structural, plus the DWI
                          and fMRI pipelines. Designed for inputs to
                          segmentation / classification models that
                          expect clean, normalized volumes.

    full-neuroimaging   - Everything the platform can run today.
                          Adds registration to standard, and the full
                          DWI/fMRI processing chains to ml-ready.
                          Suitable for research use where downstream
                          analyses need all derived outputs (FA maps,
                          motion params, etc.).

Adding a new built-in preset is a 5-line change here. The CLI and API
pick it up automatically through the registry.
"""
from app.presets.schema import Preset

BUILTIN_PRESETS = [
    Preset(
        name="basic-qc",
        description=(
            "Minimal pipeline for verifying dataset integrity. Runs "
            "only reorientation -- enough to confirm files are "
            "loadable, well-formed NIfTI volumes. Useful as a "
            "sanity-check before committing to a longer pipeline."
        ),
        step_ids=["reorientation"],
        step_params={},
        skip_recommended=[],
        target_modalities=None,  # all modalities
        is_builtin=True,
    ),
    Preset(
        name="standard",
        description=(
            "Standard structural MRI pipeline. Produces bias-corrected, "
            "skull-stripped, resampled, intensity-normalized volumes "
            "for T1/T2/FLAIR/PD acquisitions. Suitable for most "
            "research use cases that don't require DWI or fMRI "
            "processing."
        ),
        step_ids=[
            "reorientation",
            "bias_field_correction",
            "brain_extraction",
            "resampling",
            "intensity_normalization",
        ],
        step_params={},
        skip_recommended=[],
        target_modalities=None,
        is_builtin=True,
    ),
    Preset(
        name="ml-ready",
        description=(
            "Cross-modality preprocessing for machine-learning "
            "workflows. Same as `standard` for structural modalities, "
            "plus the DWI pipeline (denoise -> Gibbs -> eddy -> tensor "
            "fitting) and the fMRI pipeline (dummy volumes -> motion "
            "correction -> temporal filtering -> confound regression "
            "-> smoothing). Designed as input to segmentation / "
            "classification models that expect clean, normalized volumes."
        ),
        step_ids=[
            # Structural
            "reorientation",
            "bias_field_correction",
            "brain_extraction",
            "resampling",
            "intensity_normalization",
            # Diffusion
            "dwi_denoise",
            "gibbs_ringing_correction",
            "dwi_motion_eddy_correction",
            "tensor_fitting",
            # Functional
            "remove_initial_volumes",
            "fmri_motion_correction",
            "temporal_filtering",
            "confound_regression",
            "spatial_smoothing",
        ],
        step_params={
            # ML workflows typically want isotropic 1mm voxels for
            # structural inputs -- pre-fill this as a sensible default
            # so users don't have to remember to pass it.
            "resampling": {"voxel_size": [1.0, 1.0, 1.0]},
            # 6mm FWHM is the standard smoothing for ML-ready fMRI
            "spatial_smoothing": {"fwhm": 6.0},
        },
        skip_recommended=[],
        target_modalities=None,
        is_builtin=True,
    ),
    Preset(
        name="full-neuroimaging",
        description=(
            "Everything the platform can run today. Adds rigid "
            "registration to a reference (default: subject's primary "
            "T1) on top of `ml-ready`. Use this when downstream "
            "analyses need all derived outputs (FA/MD/AD/RD maps, "
            "motion params, registered volumes)."
        ),
        step_ids=[
            # Structural
            "reorientation",
            "bias_field_correction",
            "brain_extraction",
            "registration",
            "resampling",
            "intensity_normalization",
            # Diffusion
            "dwi_denoise",
            "gibbs_ringing_correction",
            "dwi_motion_eddy_correction",
            "tensor_fitting",
            # Functional
            "remove_initial_volumes",
            "fmri_motion_correction",
            "temporal_filtering",
            "confound_regression",
            "spatial_smoothing",
        ],
        step_params={
            "resampling": {"voxel_size": [1.0, 1.0, 1.0]},
            "spatial_smoothing": {"fwhm": 6.0},
            # Default registration: rigid transform (the cheaper option;
            # users wanting affine can override via --param)
            "registration": {"transform": "rigid"},
        },
        skip_recommended=[],
        target_modalities=None,
        is_builtin=True,
    ),
    Preset(
        name="privacy",
        description=(
            "Privacy / de-identification preset. Strips identifying "
            "NIfTI header metadata and removes facial features from "
            "structural MRI. The original raw image is never modified; "
            "the preset produces an anonymized, defaced copy in the "
            "output directory. Suitable for data sharing when "
            "preservation of facial anatomy isn't required."
        ),
        step_ids=[
            "anonymization",
            "defacing",
        ],
        step_params={
            # Use the placeholder defacing engine by default; users with
            # pydeface / mri_deface installed can override via
            # --param defacing.engine=pydeface.
            "defacing": {"engine": "placeholder"},
            "anonymization": {
                "remove_extensions": True,
                "pseudonymize": False,
            },
        },
        skip_recommended=[],
        target_modalities=None,  # privacy applies to all modalities
        is_builtin=True,
    ),
]


def get_builtin_preset(name: str):
    """Returns a built-in Preset by name, or None if not found."""
    for p in BUILTIN_PRESETS:
        if p.name == name:
            return p
    return None
