"""
FSL eddy + topup DWI correction engine (stub for PR7).

FSL's `eddy` jointly models susceptibility-induced distortion (via
topup fieldmaps) and eddy-current distortion. It is the validated
standard for production diffusion MRI preprocessing in published
research.

PR7: `is_available()` does real binary discovery via:
  1. params["binary_path"]
  2. $MRI_PREPROCESSOR_FSL_EDDY
  3. PATH (shutil.which("eddy"))
  4. $FSLDIR/bin/eddy (FSL installation layout)

License: FSL Oxford Academic non-commercial.
"""
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


FSL_EDDY_DISCOVERY = DiscoveryConfig(
    engine_name="fsl_eddy",
    binary_name="eddy",  # FSL's eddy binary is literally called "eddy"
    env_var="MRI_PREPROCESSOR_FSL_EDDY",
    tool_suite_env="FSLDIR",
)


class FSLEddyCorrection:
    """Engine wrapper for FSL eddy + topup."""

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

    def is_available(self) -> bool:
        """Probes for the FSL eddy binary. True if found."""
        path, _ = find_binary(FSL_EDDY_DISCOVERY)
        return path is not None

    def availability_reason(self) -> str:
        path, reason = find_binary(FSL_EDDY_DISCOVERY)
        if path:
            return ""
        return reason

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        path, reason = find_binary(FSL_EDDY_DISCOVERY,
                                    params.get("binary_path"))
        if not path:
            raise NotImplementedError(
                f"FSL eddy engine is not available on this machine. "
                f"{reason} Alternatively, use the 'dipy' engine "
                f"(default, reference tier) for development."
            )
        raise NotImplementedError(
            f"FSL eddy binary found at {path} but the subprocess wrapper "
            f"is not implemented yet. Use the 'dipy' engine "
            f"(default, reference tier) for development."
        )
