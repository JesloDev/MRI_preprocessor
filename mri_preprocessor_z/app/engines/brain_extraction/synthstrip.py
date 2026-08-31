"""
SynthStrip brain extraction engine (stub).

SynthStrip is FreeSurfer's deep-learning skull strip tool. It produces
state-of-the-art results across modalities (T1, T2, FLAIR, PD, ...).

PR5 ships this as a STUB. Implementing the real wrapper is a 1-file
change to this file (shell out to the SynthStrip binary, parse its
output, write the brain + mask). The dispatcher, registry, and API
already support it via params["engine"] == "synthstrip".

License caveat: SynthStrip's model weights are released under a
FreeSurfer-specific license (non-commercial use). The code itself is
also under FreeSurfer's license. A user must install FreeSurfer
separately and provide a license file. See docs/CAPABILITY_MATRIX.yaml.
"""
from typing import Any, Dict, Optional


class SynthStripBrainExtraction:
    """Stub engine for FreeSurfer SynthStrip.

    When implemented, this class will:
      1. Locate the SynthStrip binary (via env var or PATH).
      2. Construct the command line: mri_synthstrip --i <input> --o <brain> --mask <mask>.
      3. Run the binary as a subprocess.
      4. Read the output brain + mask NIfTI files.
      5. Return them in the standard result-dict shape.

    The implementation lives behind one class so all SynthStrip-specific
    concerns (license check, binary discovery, error handling) are
    isolated here. The dispatcher step function never needs to know
    whether SynthStrip is installed -- it just calls .run().
    """

    name = "synthstrip"
    maturity = "external_engine"
    validation_notes = (
        "SynthStrip is a validated deep-learning skull strip tool "
        "from the FreeSurfer team. State-of-the-art across modalities. "
        "When implemented, this engine will produce results equivalent "
        "to running SynthStrip directly."
    )
    license_notes = (
        "SynthStrip model weights: FreeSurfer non-commercial license. "
        "Requires FreeSurfer installation and a license file at "
        "$FREESURFER_HOME/license.txt. NOT for commercial use without "
        "a separate commercial license."
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
            "SynthStrip engine is a stub in PR5. To enable: install "
            "FreeSurfer, set $FREESURFER_HOME, and implement the "
            "subprocess call in app/engines/brain_extraction/synthstrip.py. "
            "Alternatively, use the 'otsu' engine (default, experimental) "
            "for development."
        )
