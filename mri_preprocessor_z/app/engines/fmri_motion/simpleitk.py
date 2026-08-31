"""
SimpleITK fMRI motion correction engine -- the existing reference
implementation, extracted verbatim from the original fmri_motion_correction
step function.

Rigid-registers every BOLD volume to a reference volume (default: the
first volume). The motion parameters are saved as a CSV for downstream
confound regression.

Maturity: reference (functionally correct, slower than MCFLIRT).
"""
import os
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk


class SimpleITKMotionCorrection:
    """Reference rigid motion correction via SimpleITK.

    Same code as the original fmri_motion_correction step function from
    PR1-PR4; extracted unchanged into an engine class.
    """

    name = "simpleitk"
    maturity = "reference"
    validation_notes = (
        "SimpleITK rigid registration of every volume to a reference. "
        "Functionally correct but slower than FSL MCFLIRT, and does "
        "not model the same cost function. For published research "
        "consider the MCFLIRT engine adapter."
    )
    license_notes = None
    requires_external_tool = False

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        nib.save(nib.Nifti1Image(corrected.astype(np.float32),
                                  img.affine, img.header), out_path)

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
