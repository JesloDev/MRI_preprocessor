"""
FSL eddy + topup DWI correction engine (stub).

FSL's `eddy` jointly models susceptibility-induced distortion (via
topup fieldmaps) and eddy-current distortion. It is the validated
standard for production diffusion MRI preprocessing in published
research.

PR5 ships this as a STUB. Implementing the real wrapper is a 1-file
change: shell out to `eddy --imain=<...> --mask=<...> --acqp=<...>
--index=<...> --bvecs=<...> --bvals=<...> --out=<...>` and parse the
output. The FSL eddy CUDA implementation is significantly faster than
the CPU version; the wrapper should detect which is available.

License caveat: FSL is released under the Oxford Academic non-commercial
license. NOT for commercial use.
"""
from typing import Any, Dict, Optional


class FSLEddyCorrection:
    """Stub engine for FSL eddy + topup."""

    name = "fsl_eddy"
    maturity = "external_engine"
    validation_notes = (
        "FSL eddy + topup is the validated standard for diffusion MRI "
        "distortion correction. Jointly models susceptibility (via "
        "topup fieldmaps) and eddy currents. When implemented, this "
        "engine will produce results equivalent to running FSL eddy "
        "directly."
    )
    license_notes = (
        "FSL is released under the Oxford Academic non-commercial "
        "license. Requires separate FSL installation with $FSLDIR set. "
        "CUDA-capable GPU recommended for performance. NOT for "
        "commercial use without a separate commercial license."
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
            "FSL eddy engine is a stub in PR5. To enable: install FSL, "
            "set $FSLDIR, and implement the subprocess call in "
            "app/engines/dwi_correction/fsl_eddy.py. Alternatively, use "
            "the 'dipy' engine (default, reference tier) for development."
        )
