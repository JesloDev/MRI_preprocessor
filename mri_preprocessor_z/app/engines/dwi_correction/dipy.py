"""
DIPY DWI motion / eddy-current correction engine -- the existing
reference implementation, extracted verbatim from the original
dwi_motion_eddy_correction step function.

Affine-registers each diffusion volume to the mean b0 using DIPY's
AffineRegistration. A pure-Python alternative to FSL eddy; suitable
for research use but not the validated standard.

Maturity: reference.
"""
import os
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np


def _load_gradients(context):
    """Loads bval/bvec from the context and returns a DIPY gradient_table.

    Same helper as the original dwi_motion_eddy_correction step function.
    Kept here (not in a shared module) because the gradient-loading
    convention is specific to the DWI engine family; structural and
    functional engines don't need it.
    """
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table

    bval_path = context.get("bval_path")
    bvec_path = context.get("bvec_path")
    if not bval_path or not bvec_path:
        raise FileNotFoundError(
            "DWI steps require context['bval_path'] and context['bvec_path']"
        )
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    return gradient_table(bvals, bvecs)


class DIPYEddyCorrection:
    """Reference DIPY affine eddy-current + motion correction.

    Same code as the original dwi_motion_eddy_correction step function
    from PR1-PR4; extracted unchanged into an engine class.
    """

    name = "dipy"
    maturity = "reference"
    validation_notes = (
        "DIPY affine registration of each diffusion volume to the mean "
        "b0. Functionally correct but does not model susceptibility "
        "distortions; for published research consider the FSL eddy "
        "engine adapter."
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
        from dipy.align.imaffine import AffineRegistration
        from dipy.align.transforms import AffineTransform3D

        img = nib.load(input_path)
        data = img.get_fdata()
        gtab = _load_gradients(context)

        b0_indices = np.where(gtab.b0s_mask)[0]
        if len(b0_indices) == 0:
            raise ValueError("No b0 volumes found in gradient table (bval == 0).")
        ref_vol = data[..., b0_indices].mean(axis=-1)

        affreg = AffineRegistration()
        corrected = np.zeros_like(data)
        for i in range(data.shape[-1]):
            moving = data[..., i]
            if i in b0_indices:
                corrected[..., i] = moving
                continue
            transform = AffineTransform3D()
            params0 = None
            affine_map = affreg.optimize(
                ref_vol, moving, transform, params0,
                img.affine, img.affine,
            )
            corrected[..., i] = affine_map.transform(moving)

        out_path = os.path.join(output_dir, "motion_eddy_corrected.nii.gz")
        nib.save(nib.Nifti1Image(corrected.astype(np.float32),
                                  img.affine, img.header), out_path)
        return {"output_path": out_path, "outputs": {}, "metrics": {}}
