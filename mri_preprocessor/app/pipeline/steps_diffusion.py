"""
Diffusion MRI (DWI/DTI) processing steps.

All implemented with DIPY (BSD licensed) so this whole modality module
works with zero non-commercial or GPL dependency. FSL's `eddy`/`topup`
remain available as an optional, separately-licensed advanced engine
(see docs/CAPABILITY_MATRIX.yaml) but are NOT required for this module
to function.
"""
import os

import nibabel as nib
import numpy as np

from app.pipeline.step_registry import step


def _load_gradients(context):
    """context must provide bval_path / bvec_path for the DWI series."""
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


@step(
    id="dwi_denoise",
    name="DWI Denoising (Local PCA)",
    modalities=["DWI", "DTI"],
    engine="DIPY",
    depends_on=["reorientation"],
    description="Denoises diffusion-weighted volumes using Local PCA (Manjon et al.).",
)
def dwi_denoise(input_path, output_dir, params, context):
    from dipy.denoise.localpca import localpca
    from dipy.denoise.pca_noise_estimate import pca_noise_estimate

    img = nib.load(input_path)
    data = img.get_fdata()
    gtab = _load_gradients(context)

    sigma = pca_noise_estimate(data, gtab, correct_bias=True)
    denoised = localpca(data, sigma=sigma, patch_radius=2)

    out_path = os.path.join(output_dir, "dwi_denoised.nii.gz")
    nib.save(nib.Nifti1Image(denoised.astype(np.float32), img.affine, img.header), out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="gibbs_ringing_correction",
    name="Gibbs Ringing Correction",
    modalities=["DWI", "DTI"],
    engine="DIPY",
    depends_on=["dwi_denoise"],
    description="Removes Gibbs ringing artifacts (Kellner et al. 2016 method).",
)
def gibbs_ringing_correction(input_path, output_dir, params, context):
    from dipy.denoise.gibbs import gibbs_removal

    img = nib.load(input_path)
    data = img.get_fdata()
    corrected = gibbs_removal(data, slice_axis=2)

    out_path = os.path.join(output_dir, "gibbs_corrected.nii.gz")
    nib.save(nib.Nifti1Image(corrected.astype(np.float32), img.affine, img.header), out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="dwi_motion_eddy_correction",
    name="Motion / Eddy-current Correction (affine registration to b0)",
    modalities=["DWI", "DTI"],
    engine="DIPY",
    depends_on=["gibbs_ringing_correction"],
    description=(
        "Registers each diffusion volume to the mean b0 using DIPY's affine "
        "registration. A pure-Python alternative to FSL eddy; swap for FSL "
        "eddy in the 'advanced' engine tier if you need its specific "
        "susceptibility+eddy joint model."
    ),
)
def dwi_motion_eddy_correction(input_path, output_dir, params, context):
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
    nib.save(nib.Nifti1Image(corrected.astype(np.float32), img.affine, img.header), out_path)
    return {"output_path": out_path, "outputs": {}, "metrics": {}}


@step(
    id="tensor_fitting",
    name="Diffusion Tensor Fitting",
    modalities=["DWI", "DTI"],
    engine="DIPY",
    depends_on=["dwi_motion_eddy_correction"],
    produces=["fa", "md", "ad", "rd"],
    description="Fits the diffusion tensor model and derives FA/MD/AD/RD maps.",
)
def tensor_fitting(input_path, output_dir, params, context):
    from dipy.reconst.dti import (
        TensorModel, fractional_anisotropy, mean_diffusivity,
        axial_diffusivity, radial_diffusivity,
    )

    img = nib.load(input_path)
    data = img.get_fdata()
    gtab = _load_gradients(context)

    mask = data[..., 0] > 0  # crude foreground mask; use brain_extraction output for better results
    model = TensorModel(gtab)
    fit = model.fit(data, mask=mask)

    fa = fractional_anisotropy(fit.evals)
    md = mean_diffusivity(fit.evals)
    ad = axial_diffusivity(fit.evals)
    rd = radial_diffusivity(fit.evals)
    fa[np.isnan(fa)] = 0

    outputs = {}
    for name, arr in [("fa", fa), ("md", md), ("ad", ad), ("rd", rd)]:
        p = os.path.join(output_dir, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(np.float32), img.affine, img.header), p)
        outputs[name] = p

    return {"output_path": outputs["fa"], "outputs": outputs, "metrics": {}}
