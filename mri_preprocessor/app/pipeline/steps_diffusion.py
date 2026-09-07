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
    maturity="production_ready",
    validation_notes="Local PCA denoising (Manjon et al. 2013) as implemented in DIPY is the standard research-grade DWI denoising method.",
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
    maturity="production_ready",
    validation_notes="Kellner et al. 2016 method as implemented in DIPY is the validated standard for Gibbs artifact removal.",
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
    maturity="reference",
    validation_notes=(
        "Reference DIPY affine registration to b0. Suitable for research "
        "use, but FSL eddy is the validated standard for production "
        "diffusion preprocessing (it jointly models susceptibility and "
        "eddy-current distortions). For published research, consider "
        "swapping to the FSL eddy engine adapter."
    ),
    # PR5: dispatcher reads params["engine"] (default "dipy") and delegates
    # to the matching engine in app/engines/dwi_correction/. Available
    # engines: dipy (reference, default), fsl_eddy (external, stub).
    default_params={"engine": "dipy"},
)
def dwi_motion_eddy_correction(input_path, output_dir, params, context):
    from app.engines.dwi_correction import DWI_CORRECTION_ENGINES

    engine_name = params.get("engine", DWI_CORRECTION_ENGINES.get_default())
    engine = DWI_CORRECTION_ENGINES.get(engine_name)
    return engine.run(input_path, output_dir, params, context)


@step(
    id="tensor_fitting",
    name="Diffusion Tensor Fitting",
    modalities=["DWI", "DTI"],
    engine="DIPY",
    depends_on=["dwi_motion_eddy_correction"],
    produces=["fa", "md", "ad", "rd"],
    description="Fits the diffusion tensor model and derives FA/MD/AD/RD maps.",
    maturity="production_ready",
    validation_notes="Standard DTI tensor fitting via DIPY's TensorModel. FA/MD/AD/RD metrics are well-established and widely used in published research.",
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
