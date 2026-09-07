"""
FSL MCFLIRT motion correction engine (stub for PR7).

MCFLIRT (Motion Correction FLIRT) is FSL's reference-implementation
motion correction for BOLD fMRI. Faster and more validated than the
SimpleITK reference; widely used in published research.

PR7: `is_available()` does real binary discovery via:
  1. params["binary_path"]
  2. $MRI_PREPROCESSOR_MCFLIRT
  3. PATH (shutil.which("mcflirt"))
  4. $FSLDIR/bin/mcflirt (FSL installation layout)

A future PR will replace the `run()` stub with the real subprocess
call. License: FSL non-commercial.
"""
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


MCFLIRT_DISCOVERY = DiscoveryConfig(
    engine_name="mcflirt",
    binary_name="mcflirt",
    env_var="MRI_PREPROCESSOR_MCFLIRT",
    tool_suite_env="FSLDIR",
)


class MCFLIRTMotionCorrection:
    """Engine wrapper for FSL MCFLIRT."""

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

    def is_available(self) -> bool:
        """Probes for the MCFLIRT binary. True if found."""
        path, _ = find_binary(MCFLIRT_DISCOVERY)
        return path is not None

    def availability_reason(self) -> str:
        path, reason = find_binary(MCFLIRT_DISCOVERY)
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
        path, reason = find_binary(MCFLIRT_DISCOVERY,
                                    params.get("binary_path"))
        if not path:
            raise NotImplementedError(
                f"MCFLIRT engine is not available on this machine. "
                f"{reason} Alternatively, use the 'simpleitk' engine "
                f"(default, reference tier) for development."
            )
        raise NotImplementedError(
            f"MCFLIRT binary found at {path} but the subprocess wrapper "
            f"is not implemented yet. Use the 'simpleitk' engine "
            f"(default, reference tier) for development."
        )
