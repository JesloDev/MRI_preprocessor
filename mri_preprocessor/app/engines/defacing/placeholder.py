"""
Placeholder defacing engine (PR11).

Naive proxy for real defacing: blurs the inferior portion of the volume
(where the face is typically located in a structural MRI). NOT a real
defacing implementation -- just enough to validate the dispatcher
pattern in test environments without external tools.

The original raw image is NEVER modified -- the placeholder writes a
defaced copy to the output directory.
"""
import os
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import scipy.ndimage  # type: ignore


class PlaceholderDefacing:
    """Built-in pure-Python defacing placeholder.

    Implementation: apply a Gaussian blur to the inferior ~30% of the
    volume (the typical face location in a RAS-oriented structural
    MRI). This is NOT a validated defacing method -- it's a placeholder
    so the dispatcher pattern can be exercised without external tools.
    """

    name = "placeholder"
    maturity = "experimental"
    validation_notes = (
        "Naive proxy for defacing: blurs the inferior portion of the "
        "volume. This is NOT a validated defacing implementation -- it "
        "is a placeholder so the platform can run end-to-end in test "
        "environments without external tools. Use 'pydeface' or "
        "'mri_deface' for real defacing."
    )
    license_notes = None
    requires_external_tool = False

    def is_available(self) -> bool:
        """Pure-Python engine -- always available."""
        return True

    def availability_reason(self) -> str:
        return ""

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        img = nib.load(input_path)
        data = img.get_fdata().astype(np.float32)

        # Infer the "inferior" axis. In a canonical RAS+ image, the
        # inferior-superior axis is axis 2 (Z). We blur the inferior
        # fraction of voxels along that axis.
        inferior_fraction = float(params.get("inferior_fraction", 0.30))
        sigma = float(params.get("sigma", 4.0))

        if data.ndim != 3:
            # 4D inputs (rare for defacing) -- apply to each volume
            vols = [data[..., t] for t in range(data.shape[-1])]
        else:
            vols = [data]

        out_vols = []
        for vol in vols:
            z = vol.shape[2]
            inferior_count = max(1, int(z * inferior_fraction))
            vol_out = vol.copy()
            inferior_block = vol_out[:, :, :inferior_count]
            blurred = scipy.ndimage.gaussian_filter(inferior_block, sigma=sigma)
            vol_out[:, :, :inferior_count] = blurred
            out_vols.append(vol_out)

        if data.ndim == 3:
            out = out_vols[0]
        else:
            out = np.stack(out_vols, axis=-1)

        out_path = os.path.join(output_dir, "defaced.nii.gz")
        nib.save(nib.Nifti1Image(out.astype(np.float32), img.affine, img.header), out_path)

        return {
            "output_path": out_path,
            "outputs": {"defaced_image": out_path},
            "metrics": {
                "engine": "placeholder",
                "inferior_fraction": inferior_fraction,
                "sigma": sigma,
            },
        }
