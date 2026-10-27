"""
HD-BET brain extraction engine (stub for PR7).

HD-BET (Deep learning Brain Extraction Tool) is an alternative to
SynthStrip, developed at the German Cancer Research Center (DKFZ).
Same pattern as the SynthStrip stub: registered in the family so the
API/UI can list it as an option, with real `is_available()` checks.

PR7: `is_available()` probes for the HD-BET binary via:
  1. params["binary_path"]
  2. $MRI_PREPROCESSOR_HDBET
  3. PATH (shutil.which("hd_bet"))
  4. (no tool-suite fallback -- HD-BET doesn't have a unified install dir)

License: HD-BET is released under a custom license (SUSE Linux-like /
non-commercial). A user must install HD-BET separately.
"""
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


HDBET_DISCOVERY = DiscoveryConfig(
    engine_id="hdbet",
    binary_name="hd_bet",
    env_var="MRI_PREPROCESSOR_HDBET",
    # HD-BET doesn't have a unified install dir -- it's a pip-installable
    # Python package. PATH lookup is the only realistic discovery.
    tool_suite_env=None,
)


class HDBETBrainExtraction:
    """Engine wrapper for HD-BET."""

    name = "hdbet"
    maturity = "external_engine"
    validation_notes = (
        "HD-BET is a validated deep-learning skull strip tool developed "
        "at DKFZ. Particularly strong on T1 images. When implemented, "
        "this engine will produce results equivalent to running HD-BET "
        "directly."
    )
    license_notes = (
        "HD-BET: custom non-commercial license. Requires separate "
        "installation of the HD-BET Python package and its PyTorch "
        "model weights. NOT for commercial use."
    )
    requires_external_tool = True

    def is_available(self) -> bool:
        """Probes for the HD-BET binary. True if found."""
        path, _ = find_binary(HDBET_DISCOVERY)
        return path is not None

    def availability_reason(self) -> str:
        path, reason = find_binary(HDBET_DISCOVERY)
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
        path, reason = find_binary(HDBET_DISCOVERY,
                                    params.get("binary_path"))
        if not path:
            raise NotImplementedError(
                f"HD-BET engine is not available on this machine. "
                f"{reason} Alternatively, use the 'otsu' engine "
                f"(default, experimental) for development."
            )
        raise NotImplementedError(
            f"HD-BET binary found at {path} but the subprocess wrapper "
            f"is not implemented yet. Use the 'otsu' engine "
            f"(default, experimental) for development."
        )
