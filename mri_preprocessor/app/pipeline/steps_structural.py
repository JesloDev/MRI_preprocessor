"""
Structural MRI processing steps (T1, T2, FLAIR, PD).

Implementation notes:
  - reorientation, intensity normalization: pure nibabel/numpy, no
    external dependency and no licensing concerns.
  - bias_field_correction, resampling, registration: SimpleITK (Apache 2.0).
  - brain_extraction: a simple Otsu-threshold + largest-connected-component
    implementation. This is a PLACEHOLDER -- swap this function's body for
    a SynthStrip or HD-BET call when you integrate those engines. Keeping
    it isolated behind one function is exactly why the step-registry
    pattern is useful: the rest of the pipeline never needs to know which
    engine implements "brain_extraction".
"""
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from app.pipeline.step_registry import step


def _save_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: str) -> str:
    out_img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(out_img, out_path)
    return out_path


@step(
    id="reorientation",
    name="Reorientation to canonical (RAS+)",
    modalities=["T1", "T2", "FLAIR", "PD", "SWI", "GRE", "DWI", "fMRI", "ASL"],
    engine="nibabel",
    description="Reorients the image to closest canonical RAS+ orientation.",
)
def reorientation(input_path, output_dir, params, context):
    img = nib.load(input_path)
    reoriented = nib.as_closest_canonical(img)
    out_path = os.path.join(output_dir, "reoriented.nii.gz")
    nib.save(reoriented, out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="bias_field_correction",
    name="N4 Bias Field Correction",
    modalities=["T1", "T2", "FLAIR", "PD"],
    engine="SimpleITK",
    depends_on=["reorientation"],
    default_params={"shrink_factor": 4, "num_iterations": 50},
    description="Corrects low-frequency intensity non-uniformity (N4ITK).",
)
def bias_field_correction(input_path, output_dir, params, context):
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    shrink = params.get("shrink_factor", 4)
    n_iter = params.get("num_iterations", 50)

    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    shrunk_image = sitk.Shrink(image, [shrink] * image.GetDimension())
    shrunk_mask = sitk.Shrink(mask_image, [shrink] * image.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iter])
    _ = corrector.Execute(shrunk_image, shrunk_mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias_field)

    out_path = os.path.join(output_dir, "n4_corrected.nii.gz")
    sitk.WriteImage(sitk.Cast(corrected, sitk.sitkFloat32), out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="brain_extraction",
    name="Brain Extraction (skull strip)",
    modalities=["T1", "T2", "FLAIR"],
    engine="SimpleITK (placeholder -- replace with SynthStrip/HD-BET)",
    depends_on=["bias_field_correction"],
    produces=["brain_mask"],
    description=(
        "PLACEHOLDER Otsu + largest-connected-component skull strip. "
        "Not clinically validated; replace with SynthStrip or HD-BET "
        "for production use."
    ),
)
def brain_extraction(input_path, output_dir, params, context):
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])

    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    if stats.GetNumberOfLabels() == 0:
        largest_mask = mask
    else:
        largest_label = max(stats.GetLabels(), key=lambda l: stats.GetNumberOfPixels(l))
        largest_mask = sitk.Equal(cc, largest_label)

    largest_mask = sitk.Cast(largest_mask, sitk.sitkUInt8)
    brain = sitk.Mask(image, largest_mask)

    brain_path = os.path.join(output_dir, "brain.nii.gz")
    mask_path = os.path.join(output_dir, "brain_mask.nii.gz")
    sitk.WriteImage(sitk.Cast(brain, sitk.sitkFloat32), brain_path)
    sitk.WriteImage(largest_mask, mask_path)

    return {
        "output_path": brain_path,
        "outputs": {"brain_mask": mask_path},
        "metrics": {},
    }


@step(
    id="registration",
    name="Registration to reference",
    modalities=["T1", "T2", "FLAIR", "DWI", "fMRI", "ASL"],
    engine="SimpleITK",
    depends_on=["brain_extraction"],
    default_params={"transform": "rigid", "reference": None},
    description="Rigid/affine registration to a reference image (e.g. subject T1 or MNI152).",
)
def registration(input_path, output_dir, params, context):
    reference_path = params.get("reference") or context.get("reference_image")
    if not reference_path or not os.path.exists(reference_path):
        raise FileNotFoundError(
            "registration requires a 'reference' image path in params or context['reference_image']"
        )

    fixed = sitk.ReadImage(reference_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(input_path, sitk.sitkFloat32)

    transform_type = params.get("transform", "rigid")
    if transform_type == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    else:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])

    final_transform = reg.Execute(fixed, moving)

    resampled = sitk.Resample(
        moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    out_path = os.path.join(output_dir, "registered.nii.gz")
    sitk.WriteImage(sitk.Cast(resampled, sitk.sitkFloat32), out_path)

    transform_path = os.path.join(output_dir, "transform.tfm")
    sitk.WriteTransform(final_transform, transform_path)

    return {
        "output_path": out_path,
        "outputs": {"transform": transform_path},
        "metrics": {"final_metric_value": reg.GetMetricValue()},
    }


@step(
    id="resampling",
    name="Resampling to target voxel size",
    modalities=["T1", "T2", "FLAIR", "DWI", "fMRI", "ASL", "PD", "SWI"],
    engine="SimpleITK",
    default_params={"voxel_size": [1.0, 1.0, 1.0], "interpolation": "linear"},
    description="Resamples the image to an isotropic or custom target voxel size.",
)
def resampling(input_path, output_dir, params, context):
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    target_spacing = params.get("voxel_size", [1.0, 1.0, 1.0])

    interp_map = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
    }
    interpolator = interp_map.get(params.get("interpolation", "linear"), sitk.sitkLinear)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampled = sitk.Resample(
        image, new_size, sitk.Transform(), interpolator,
        image.GetOrigin(), target_spacing, image.GetDirection(),
        0.0, image.GetPixelID(),
    )

    out_path = os.path.join(output_dir, "resampled.nii.gz")
    sitk.WriteImage(resampled, out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="intensity_normalization",
    name="Intensity Normalization",
    modalities=["T1", "T2", "FLAIR", "PD"],
    engine="nibabel/numpy",
    default_params={"method": "zscore"},
    description="Normalizes voxel intensities (z-score, min-max, or percentile).",
)
def intensity_normalization(input_path, output_dir, params, context):
    img = nib.load(input_path)
    data = img.get_fdata()
    method = params.get("method", "zscore")
    mask = data > 0  # non-background voxels

    if method == "zscore":
        mean, std = data[mask].mean(), data[mask].std()
        norm = (data - mean) / (std + 1e-8)
    elif method == "minmax":
        lo, hi = data[mask].min(), data[mask].max()
        norm = (data - lo) / (hi - lo + 1e-8)
    elif method == "percentile":
        lo, hi = np.percentile(data[mask], 1), np.percentile(data[mask], 99)
        norm = np.clip((data - lo) / (hi - lo + 1e-8), 0, 1)
    else:
        raise ValueError(f"Unknown intensity normalization method: {method}")

    out_path = os.path.join(output_dir, "intensity_normalized.nii.gz")
    _save_like(img, norm, out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}
