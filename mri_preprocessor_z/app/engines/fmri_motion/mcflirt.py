"""
FSL MCFLIRT motion correction engine (stub).

MCFLIRT (Motion Correction FLIRT) is FSL's reference-implementation
motion correction for BOLD fMRI. Faster and more validated than the
SimpleITK reference; widely used in published research.

PR5 ships this as a STUB. Implementing the real wrapper is a 1-file
change: shell out to `mcflirt -in <input> -out <output> -plots` and
parse the generated motion parameter files.

License caveat: FSL is released under the Oxford Academic non-commercial
license. NOT for commercial use.
"""
from typing import Any, Dict, Optional


class MCFLIRTMotionCorrection:
    """Stub engine for FSL MCFLIRT."""

    name = "mcflirt"
    maturity = "external_engine"
    validation_notes = (
        "FSL MCFLIRT is the validated standard for BOLD motion "
        "correction, widely used in published research. Faster than "
        "the SimpleITK reference implementation. When implemented, this "
        "engine will produce results equivalent to running MCFLIRT "
        "directly."
    )
    license_notes = (
        "FSL is released under the Oxford Academic non-commercial "
        "license. Requires separate FSL installation with $FSLDIR set. "
        "NOT for commercial use without a separate commercial license."
    )
    requires_external_tool = True

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "MCFLIRT engine is a stub in PR5. To enable: install FSL, "
            "set $FSLDIR, and implement the subprocess call in "
            "app/engines/fmri_motion/mcflirt.py. Alternatively, use "
            "the 'simpleitk' engine (default, reference tier) for "
            "development."
        )
