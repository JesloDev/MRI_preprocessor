"""
HD-BET brain extraction engine (stub).

HD-BET (Deep learning Brain Extraction Tool) is an alternative to
SynthStrip, developed at the German Cancer Research Center (DKFZ).
Same pattern as the SynthStrip stub: registered in the family so the
API/UI can list it as an option, but raises NotImplementedError if
actually invoked before the real wrapper is implemented.

License: HD-BET is released under a custom license (SUSE Linux-like /
non-commercial). A user must install HD-BET separately.
"""
from typing import Any, Dict, Optional


class HDBETBrainExtraction:
    """Stub engine for HD-BET."""

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

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "HD-BET engine is a stub in PR5. To enable: install HD-BET "
            "and implement the subprocess call in "
            "app/engines/brain_extraction/hdbet.py. Alternatively, use "
            "the 'otsu' engine (default, experimental) for development."
        )
