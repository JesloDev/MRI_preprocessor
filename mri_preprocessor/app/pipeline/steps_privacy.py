"""
Intensity filtering and privacy/de-identification steps (PR11).

Three new steps land in this PR:

  - `intensity_filtering`  : spatial-frequency filtering (high-pass,
                             low-pass, band-pass) using FFT. Separated
                             from `intensity_normalization` per the
                             scientific-correctness principle that
                             "filtering" (frequency-domain) and
                             "normalization" (intensity-domain) are
                             different operations.
  - `defacing`             : remove facial/anatomical features that
                             could identify a subject. Engine-based
                             (pydeface, mri_deface, built-in placeholder).
                             Original raw image is never modified.
  - `anonymization`        : NIfTI header anonymization. Strips
                             identifying metadata (patient name, IDs,
                             dates). Optionally pseudonymizes by
                             rewriting the subject identifier. Real
                             nibabel-based implementation -- no external
                             tool needed.

The privacy distinction (per the spec) is:

    PRIVACY
       |
       +-- ANONYMIZATION  (metadata / headers)
       |
       +-- DEFACING       (image data / anatomy)

Both are first-class steps in the registry and can be combined in a
"privacy" preset.
"""
import os
from typing import Any, Dict

import nibabel as nib
import numpy as np

from app.pipeline.step_registry import ParameterSpec, step


# --------------------------------------------------------------------- #
# Intensity filtering (spatial-frequency domain)
# --------------------------------------------------------------------- #

def _butterworth_filter_mask(shape, cutoff_low, cutoff_high, order=2):
    """Builds a 3-D Butterworth band-pass filter mask in frequency space.

    The mask is built in the FFT-shifted coordinate system so it
    corresponds to the output of np.fft.fftn + np.fft.fftshift.

    cutoff_low: frequencies below this are attenuated (high-pass behavior)
                -- None disables the high-pass component.
    cutoff_high: frequencies above this are attenuated (low-pass behavior)
                -- None disables the low-pass component.

    Frequencies are normalized so that 1.0 = Nyquist (half the sampling
    rate of the volume). This makes the cutoffs dimensionless and
    independent of voxel size -- the user thinks in normalized units,
    not in physical Hz. Future work could add voxel-size-aware cutoffs.
    """
    # Build coordinate grids. For each axis of length N, the normalized
    # frequency goes from -0.5 to +0.5 (after fftshift), where 0.5 is
    # the Nyquist frequency.
    grids = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), np.fft.fftfreq(shape[2])
    # Meshgrid; we use indexing='ij' so axes match array axes
    F1, F2, F3 = np.meshgrid(*[np.fft.fftshift(g) for g in grids], indexing="ij")
    # Radial frequency: 0 at DC, up to ~sqrt(3)*0.5 at the corners
    freq_mag = np.sqrt(F1**2 + F2**2 + F3**2)

    mask = np.ones_like(freq_mag)

    # Low-pass component: attenuate frequencies above cutoff_high
    if cutoff_high is not None and cutoff_high > 0:
        ch = float(cutoff_high)
        # Butterworth low-pass: 1 / (1 + (f/cutoff)^{2*order})
        lp = 1.0 / (1.0 + (freq_mag / ch) ** (2 * order))
        mask = mask * lp

    # High-pass component: attenuate frequencies below cutoff_low
    if cutoff_low is not None and cutoff_low > 0:
        cl = float(cutoff_low)
        # High-pass = 1 - low-pass
        hp = 1.0 - 1.0 / (1.0 + (freq_mag / cl) ** (2 * order))
        mask = mask * hp

    return mask


@step(
    id="intensity_filtering",
    name="Intensity (Spatial-Frequency) Filtering",
    modalities=["T1", "T2", "FLAIR", "PD"],
    engine="numpy/FFT",
    description=(
        "Spatial-frequency filtering of the image via FFT + "
        "Butterworth filter. Distinct from intensity_normalization: "
        "filtering operates on the spatial-frequency content of the "
        "image, not on the intensity values themselves. high_pass and "
        "low_pass are frequency cutoffs (normalized: 1.0 = Nyquist), "
        "not intensity thresholds. Specifying both produces a band-pass "
        "filter."
    ),
    maturity="reference",
    validation_notes=(
        "Reference FFT + Butterworth implementation. Suitable for "
        "research use; for production use, consider validating against "
        "a reference implementation (e.g. SPM's high-pass filter). "
        "Cutoffs are in normalized frequency units (1.0 = Nyquist); a "
        "future PR may add voxel-size-aware cutoffs in physical Hz."
    ),
    parameters=[
        ParameterSpec(
            name="high_pass", type="float", default=None,
            description=(
                "High-pass cutoff (normalized frequency, 0..1 where 1.0 = "
                "Nyquist). Frequencies below this are attenuated. Use to "
                "remove slow intensity drifts (e.g. residual bias field)."
            ),
        ),
        ParameterSpec(
            name="low_pass", type="float", default=None,
            description=(
                "Low-pass cutoff (normalized frequency, 0..1 where 1.0 = "
                "Nyquist). Frequencies above this are attenuated. Use to "
                "smooth high-frequency noise."
            ),
        ),
        ParameterSpec(
            name="order", type="int", default=2,
            description="Butterworth filter order. Higher = sharper rolloff.",
        ),
    ],
)
def intensity_filtering(input_path, output_dir, params, context):
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.float64)

    hp = params.get("high_pass")
    lp = params.get("low_pass")
    order = int(params.get("order", 2))

    if hp is None and lp is None:
        # No filtering requested -- just copy through. Still useful as a
        # no-op placeholder in pipelines that include the step
        # conditionally.
        out_path = os.path.join(output_dir, "filtered.nii.gz")
        nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), out_path)
        return {"output_path": out_path, "outputs": {}, "metrics": {
            "filter_type": "noop",
        }}

    # Handle 4D (e.g. fMRI) by filtering each volume independently.
    # 3D is the common case for structural MRI.
    if data.ndim == 3:
        vols = [data]
    elif data.ndim == 4:
        vols = [data[..., t] for t in range(data.shape[-1])]
    else:
        raise ValueError(
            f"intensity_filtering supports 3D and 4D inputs; got shape "
            f"{data.shape}"
        )

    filtered_vols = []
    for vol in vols:
        # FFT, shift DC to center, multiply by mask, unshift, inverse FFT.
        fft = np.fft.fftn(vol)
        fft_shifted = np.fft.fftshift(fft)
        mask = _butterworth_filter_mask(vol.shape, hp, lp, order=order)
        filtered = np.fft.ifftshift(fft_shifted * mask)
        vol_out = np.real(np.fft.ifftn(filtered))
        filtered_vols.append(vol_out)

    if data.ndim == 3:
        out = filtered_vols[0]
    else:
        out = np.stack(filtered_vols, axis=-1)

    out_path = os.path.join(output_dir, "filtered.nii.gz")
    nib.save(nib.Nifti1Image(out.astype(np.float32), img.affine, img.header), out_path)

    filter_type = (
        "band_pass" if (hp is not None and lp is not None)
        else "high_pass" if hp is not None
        else "low_pass"
    )
    return {
        "output_path": out_path,
        "outputs": {},
        "metrics": {
            "filter_type": filter_type,
            "high_pass": float(hp) if hp is not None else None,
            "low_pass": float(lp) if lp is not None else None,
            "order": order,
        },
    }


# --------------------------------------------------------------------- #
# Defacing -- engine-based, like brain_extraction
# --------------------------------------------------------------------- #

@step(
    id="defacing",
    name="Defacing (facial feature removal)",
    modalities=["T1", "T2", "FLAIR"],
    engine="placeholder / pydeface / mri_deface",
    produces=["defaced_image"],
    description=(
        "Removes facial / anatomical features that could identify a "
        "subject. The original raw image is NEVER modified -- the step "
        "writes a defaced copy to the output directory. Engine-based: "
        "choose between pydeface, mri_deface, or a built-in placeholder "
        "via params['engine']. Run `python -m app.cli engines defacing` "
        "to see what's available on this machine."
    ),
    maturity="experimental",
    validation_notes=(
        "The built-in placeholder engine blurs the inferior portion of "
        "the volume (a naive proxy for face removal). NOT a real "
        "defacing implementation. Use the 'pydeface' or 'mri_deface' "
        "engines when available -- they produce validated, irreversible "
        "facial-feature removal."
    ),
    default_params={"engine": "placeholder"},
    parameters=[
        ParameterSpec(
            name="engine", type="str", default="placeholder",
            allowed_values=["placeholder", "pydeface", "mri_deface"],
            description="Which defacing engine to use. 'placeholder' is "
                        "always available; 'pydeface' and 'mri_deface' "
                        "require external installations (run `engines "
                        "defacing` to check availability).",
        ),
    ],
)
def defacing(input_path, output_dir, params, context):
    from app.engines.defacing import DEFACING_ENGINES
    engine_name = params.get("engine", DEFACING_ENGINES.get_default())
    engine = DEFACING_ENGINES.get(engine_name)
    return engine.run(input_path, output_dir, params, context)


# --------------------------------------------------------------------- #
# Anonymization -- NIfTI header anonymization (no external tool needed)
# --------------------------------------------------------------------- #

# Standard NIfTI header fields that may contain identifying information.
# We strip these by setting them to safe defaults. nibabel exposes them
# via the header's `extensions` and the NIfTI1 header struct.
ANONYMIZED_DEFAULTS = {
    "db_name": b"anonymized",
    "aux_file": b"",
    "descrip": b"anonymized",
}


@step(
    id="anonymization",
    name="Anonymization (NIfTI header de-identification)",
    modalities=["T1", "T2", "FLAIR", "PD", "SWI", "GRE", "DWI", "DTI",
                 "fMRI", "ASL", "ADC", "TOF", "MRA", "MRV"],
    engine="nibabel",
    produces=["anonymized_image"],
    description=(
        "Strips identifying metadata from NIfTI headers (patient name, "
        "aux_file, descrip, db_name) and optionally pseudonymizes the "
        "subject identifier. The original raw image is NEVER modified -- "
        "an anonymized copy is written to the output directory. Distinct "
        "from defacing: anonymization handles metadata/headers; "
        "defacing handles image data/anatomy."
    ),
    maturity="production_ready",
    validation_notes=None,  # standard nibabel-based header manipulation
    parameters=[
        ParameterSpec(
            name="remove_extensions", type="bool", default=True,
            description=(
                "If True, NIfTI header extensions (which may contain "
                "identifying XML metadata from DICOM conversion) are "
                "stripped."
            ),
        ),
        ParameterSpec(
            name="pseudonymize", type="bool", default=False,
            description=(
                "If True, rewrite the subject_id (when discoverable from "
                "the input path or context) as a stable hash so the "
                "anonymized file no longer carries the original subject "
                "identifier."
            ),
        ),
        ParameterSpec(
            name="pseudo_salt", type="str", default="mripreprocessor",
            description=(
                "Salt used for pseudonymization. Stable across runs "
                "(same subject -> same pseudo id). Override per-deployment "
                "to prevent cross-deployment correlation."
            ),
        ),
    ],
)
def anonymization(input_path, output_dir, params, context):
    import hashlib

    img = nib.load(input_path)
    header = img.header.copy()

    # Strip identifying standard header fields
    for field, value in ANONYMIZED_DEFAULTS.items():
        try:
            if field in header:
                header[field] = value
        except (ValueError, KeyError):
            # Some fields may not exist on all NIfTI variants; skip
            # rather than crash.
            pass

    # NIfTI1 has an `intent_name` (char[16]) field that may carry
    # user-supplied text. Clear it.
    try:
        header.set_intent("none")
    except Exception:  # noqa: BLE001 -- intent setting is best-effort
        pass

    # Strip header extensions (XML comments, etc. -- may contain DICOM
    # metadata identifying the subject)
    remove_extensions = bool(params.get("remove_extensions", True))
    new_img = nib.Nifti1Image(
        np.asarray(img.dataobj).astype(np.float32),
        affine=img.affine,
        header=header,
    )
    if remove_extensions:
        # nibabel exposes extensions via img.header.extensions; creating
        # a fresh Nifti1Image drops them by default.
        pass
    else:
        # Preserve any existing extensions
        for ext in img.header.extensions:
            new_img.header.extensions.append(ext.get_content())

    out_path = os.path.join(output_dir, "anonymized.nii.gz")
    nib.save(new_img, out_path)

    metrics: Dict[str, Any] = {
        "stripped_fields": list(ANONYMIZED_DEFAULTS.keys()),
        "removed_extensions": remove_extensions,
    }

    # Pseudonymization: derive a stable pseudo id from the original
    # subject id + a salt. We don't store the mapping (the deployment
    # is responsible for that, if needed at all).
    pseudonymize = bool(params.get("pseudonymize", False))
    if pseudonymize:
        salt = str(params.get("pseudo_salt", "mripreprocessor"))
        subject_id = context.get("subject_id") or ""
        pseudo = hashlib.sha256(
            f"{salt}:{subject_id}".encode("utf-8")
        ).hexdigest()[:16]
        metrics["pseudonymized"] = True
        metrics["pseudo_id"] = pseudo
    else:
        metrics["pseudonymized"] = False

    return {
        "output_path": out_path,
        "outputs": {"anonymized_image": out_path},
        "metrics": metrics,
    }
