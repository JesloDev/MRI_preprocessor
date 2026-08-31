"""
Otsu brain extraction engine -- the existing placeholder implementation,
extracted verbatim from the original brain_extraction step function.

This is the only brain-extraction engine that actually works in PR5.
SynthStrip and HD-BET are stubs documenting where the future validated
implementations will plug in.

Maturity: experimental.
"""
import os
from typing import Any, Dict, Optional

import SimpleITK as sitk


class OtsuBrainExtraction:
    """SimpleITK Otsu threshold + largest connected component.

    Same code as the original brain_extraction step function from
    PR1-PR4; extracted unchanged into an engine class. The dispatcher
    in steps_structural.py delegates to this engine when
    params["engine"] == "otsu" (or when no engine is specified, since
    otsu is the default).
    """

    name = "otsu"
    maturity = "experimental"
    validation_notes = (
        "Otsu thresholding + largest connected component. Placeholder "
        "implementation; not validated against reference brain extraction "
        "tools. Results may be unreliable for subjects with atypical "
        "anatomy. Replace with synthstrip or hdbet for production use."
    )
    license_notes = None  # pure-Python, no external tool needed
    requires_external_tool = False

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        image = sitk.ReadImage(input_path, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])

        cc = sitk.ConnectedComponent(mask)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        if stats.GetNumberOfLabels() == 0:
            largest_mask = mask
        else:
            largest_label = max(stats.GetLabels(),
                                key=lambda l: stats.GetNumberOfPixels(l))
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
