"""
Functional MRI (BOLD / resting-state / task) processing steps.

Filtering, confound regression, and connectivity use nilearn (BSD
licensed). Motion correction here is a reference SimpleITK
implementation (rigid registration of every volume to a reference
frame) -- functional but slower than FSL's MCFLIRT; treat MCFLIRT as
an optional faster "advanced engine" swap-in behind the same step id.
"""
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from app.pipeline.step_registry import step


@step(
    id="remove_initial_volumes",
    name="Remove Initial (dummy) Volumes",
    modalities=["fMRI"],
    engine="nibabel",
    default_params={"n_volumes": 4},
    description="Discards the first N volumes to allow for T1 signal equilibration.",
    maturity="production_ready",
    validation_notes=None,  # standard nibabel-based volume slicing, well understood
)
def remove_initial_volumes(input_path, output_dir, params, context):
    img = nib.load(input_path)
    data = img.get_fdata()
    n = params.get("n_volumes", 4)
    trimmed = data[..., n:]

    out_path = os.path.join(output_dir, "trimmed.nii.gz")
    nib.save(nib.Nifti1Image(trimmed.astype(np.float32), img.affine, img.header), out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {"volumes_removed": n}}


@step(
    id="fmri_motion_correction",
    name="Motion Correction (rigid, volume-to-reference)",
    modalities=["fMRI"],
    engine="SimpleITK",
    depends_on=["remove_initial_volumes"],
    produces=["motion_params"],
    description=(
        "Rigid-registers every volume to a reference volume (default: "
        "first volume). Reference implementation -- swap for FSL MCFLIRT "
        "as an advanced engine if throughput becomes a concern."
    ),
    maturity="reference",
    validation_notes=(
        "Reference SimpleITK rigid registration of every volume to a "
        "reference. Functionally correct but slower than FSL MCFLIRT, "
        "and does not model the same cost function. For published research "
        "consider swapping to the MCFLIRT engine adapter."
    ),
    # PR5: dispatcher reads params["engine"] (default "simpleitk") and
    # delegates to the matching engine in app/engines/fmri_motion/.
    # Available engines: simpleitk (reference, default), mcflirt (external,
    # stub).
    default_params={"engine": "simpleitk"},
)
def fmri_motion_correction(input_path, output_dir, params, context):
    from app.engines.fmri_motion import FMRI_MOTION_ENGINES

    engine_name = params.get("engine", FMRI_MOTION_ENGINES.get_default())
    engine = FMRI_MOTION_ENGINES.get(engine_name)
    return engine.run(input_path, output_dir, params, context)


@step(
    id="temporal_filtering",
    name="Temporal Filtering",
    modalities=["fMRI"],
    engine="nilearn",
    depends_on=["fmri_motion_correction"],
    default_params={"high_pass": 0.01, "low_pass": 0.1, "t_r": 2.0},
    description="Band-pass temporal filtering of the BOLD signal.",
    maturity="production_ready",
    validation_notes="nilearn.signal.clean is the standard research-grade implementation of BOLD band-pass filtering.",
)
def temporal_filtering(input_path, output_dir, params, context):
    from nilearn.signal import clean

    img = nib.load(input_path)
    data = img.get_fdata()
    shape4d = data.shape
    data_2d = data.reshape(-1, shape4d[-1]).T  # time x voxels

    cleaned = clean(
        data_2d,
        high_pass=params.get("high_pass", 0.01),
        low_pass=params.get("low_pass", 0.1),
        t_r=params.get("t_r", 2.0),
        standardize=False,
    )

    cleaned_4d = cleaned.T.reshape(shape4d)
    out_path = os.path.join(output_dir, "filtered.nii.gz")
    nib.save(nib.Nifti1Image(cleaned_4d.astype(np.float32), img.affine, img.header), out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="confound_regression",
    name="Nuisance / Confound Regression",
    modalities=["fMRI"],
    engine="nilearn",
    depends_on=["temporal_filtering"],
    description=(
        "Regresses out nuisance signals (motion params, optionally CSF/WM "
        "signal) using nilearn.signal.clean. Reads context['confounds_path'] "
        "if provided (CSV with one column per confound regressor)."
    ),
    maturity="production_ready",
    validation_notes="nilearn.signal.clean with confounds is the standard research-grade nuisance regression implementation.",
)
def confound_regression(input_path, output_dir, params, context):
    from nilearn.signal import clean

    img = nib.load(input_path)
    data = img.get_fdata()
    shape4d = data.shape
    data_2d = data.reshape(-1, shape4d[-1]).T

    confounds = None
    confounds_path = context.get("confounds_path")
    if confounds_path and os.path.exists(confounds_path):
        confounds = np.loadtxt(confounds_path, delimiter=",", skiprows=1)

    cleaned = clean(data_2d, confounds=confounds, standardize=False)
    cleaned_4d = cleaned.T.reshape(shape4d)

    out_path = os.path.join(output_dir, "confound_regressed.nii.gz")
    nib.save(nib.Nifti1Image(cleaned_4d.astype(np.float32), img.affine, img.header), out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="spatial_smoothing",
    name="Spatial Smoothing",
    modalities=["fMRI"],
    engine="nilearn",
    depends_on=["confound_regression"],
    default_params={"fwhm": 6.0},
    description="Gaussian spatial smoothing of the functional volumes.",
    maturity="production_ready",
    validation_notes="nilearn.image.smooth_img is the standard research-grade spatial smoothing implementation.",
)
def spatial_smoothing(input_path, output_dir, params, context):
    from nilearn.image import smooth_img

    smoothed = smooth_img(input_path, fwhm=params.get("fwhm", 6.0))
    out_path = os.path.join(output_dir, "smoothed.nii.gz")
    nib.save(smoothed, out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}
