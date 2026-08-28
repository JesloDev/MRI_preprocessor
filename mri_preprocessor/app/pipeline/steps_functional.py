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
)
def fmri_motion_correction(input_path, output_dir, params, context):
    img = nib.load(input_path)
    data = img.get_fdata()
    n_vols = data.shape[-1]
    ref_index = params.get("reference_volume", 0)
    ref_np = data[..., ref_index]
    ref_sitk = sitk.GetImageFromArray(np.transpose(ref_np, (2, 1, 0)))

    corrected = np.zeros_like(data)
    motion_params = []

    for i in range(n_vols):
        vol = data[..., i]
        if i == ref_index:
            corrected[..., i] = vol
            motion_params.append([0.0] * 6)
            continue

        moving_sitk = sitk.GetImageFromArray(np.transpose(vol, (2, 1, 0)))
        moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)
        fixed_sitk = sitk.Cast(ref_sitk, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 100)
        reg.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = reg.Execute(fixed_sitk, moving_sitk)

        resampled = sitk.Resample(
            moving_sitk, fixed_sitk, final_transform, sitk.sitkLinear, 0.0
        )
        corrected[..., i] = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))
        motion_params.append(list(final_transform.GetParameters()))

    out_path = os.path.join(output_dir, "motion_corrected.nii.gz")
    nib.save(nib.Nifti1Image(corrected.astype(np.float32), img.affine, img.header), out_path)

    motion_path = os.path.join(output_dir, "motion_params.csv")
    np.savetxt(motion_path, np.array(motion_params), delimiter=",",
               header="rx,ry,rz,tx,ty,tz", comments="")

    return {
        "output_path": out_path,
        "outputs": {"motion_params": motion_path},
        "metrics": {"mean_abs_translation_mm": float(
            np.mean(np.abs(np.array(motion_params)[:, 3:]))
        )},
    }


@step(
    id="temporal_filtering",
    name="Temporal Filtering",
    modalities=["fMRI"],
    engine="nilearn",
    depends_on=["fmri_motion_correction"],
    default_params={"high_pass": 0.01, "low_pass": 0.1, "t_r": 2.0},
    description="Band-pass temporal filtering of the BOLD signal.",
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
)
def spatial_smoothing(input_path, output_dir, params, context):
    from nilearn.image import smooth_img

    smoothed = smooth_img(input_path, fwhm=params.get("fwhm", 6.0))
    out_path = os.path.join(output_dir, "smoothed.nii.gz")
    nib.save(smoothed, out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}
