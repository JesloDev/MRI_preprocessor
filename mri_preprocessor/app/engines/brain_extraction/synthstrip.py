"""
SynthStrip brain extraction engine -- real subprocess wrapper.

FreeSurfer's SynthStrip is a deep-learning skull strip tool that
produces state-of-the-art results across modalities (T1, T2, FLAIR,
PD, ...). It's released as part of FreeSurfer.

This module wraps the `mri_synthstrip` command-line tool as a
ProcessingEngine. The dispatcher in steps_structural.py delegates to
this engine when `params["engine"] == "synthstrip"`.

Execution flow:
    1. find_binary(SYNTHSTRIP_DISCOVERY) -> path to mri_synthstrip
    2. Construct command: mri_synthstrip --i <input> --o <brain> --mask <mask>
    3. subprocess.run(..., check=True)
    4. Validate output brain + mask NIfTI files exist
    5. Return standard result dict

License: SynthStrip's model weights are released under FreeSurfer's
non-commercial license. Requires FreeSurfer installation and a license
file at $FREESURFER_HOME/license.txt.

Tests use a fake "mri_synthstrip" shell script that copies the input
to brain_output and writes a zero mask -- so we can validate the full
subprocess + output-validation flow without an actual FreeSurfer
installation.
"""
import os
import subprocess
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


# Discovery configuration for SynthStrip.
# Checks (in order):
#   1. params["binary_path"] (per-job override)
#   2. $MRI_PREPROCESSOR_SYNTHSTRIP env var (point at the binary directly)
#   3. PATH (shutil.which("mri_synthstrip"))
#   4. $FREESURFER_HOME/bin/mri_synthstrip (FreeSurfer installation layout)
SYNTHSTRIP_DISCOVERY = DiscoveryConfig(
    engine_name="synthstrip",
    binary_name="mri_synthstrip",
    env_var="MRI_PREPROCESSOR_SYNTHSTRIP",
    tool_suite_env="FREESURFER_HOME",
)


class SynthStripBrainExtraction:
    """Engine wrapper for FreeSurfer SynthStrip.

    When called via the dispatcher:
        brain_extraction_fn(input_path, output_dir, params, context)
            -> params["engine"] = "synthstrip"
            -> SynthStripBrainExtraction.run(...)
            -> subprocess.run(["mri_synthstrip", ...])
            -> {output_path: brain.nii.gz, outputs: {brain_mask: mask.nii.gz}}
    """

    name = "synthstrip"
    maturity = "external_engine"
    validation_notes = (
        "SynthStrip is a validated deep-learning skull strip tool "
        "from the FreeSurfer team. State-of-the-art across modalities. "
        "Produces results equivalent to running mri_synthstrip directly."
    )
    license_notes = (
        "SynthStrip model weights: FreeSurfer non-commercial license. "
        "Requires FreeSurfer installation and a license file at "
        "$FREESURFER_HOME/license.txt. NOT for commercial use without "
        "a separate commercial license."
    )
    requires_external_tool = True

    def is_available(self) -> bool:
        """Probes for the SynthStrip binary. True if found."""
        path, _ = find_binary(SYNTHSTRIP_DISCOVERY)
        return path is not None

    def availability_reason(self) -> str:
        """Returns the discovery-failure message when is_available() is False."""
        path, reason = find_binary(SYNTHSTRIP_DISCOVERY)
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
        """Executes mri_synthstrip as a subprocess.

        Raises:
            FileNotFoundError: if the SynthStrip binary isn't found.
            RuntimeError: if the subprocess fails or the expected output
                files aren't produced.
        """
        # 1. Find the binary
        binary_path, reason = find_binary(
            SYNTHSTRIP_DISCOVERY,
            params.get("binary_path"),
        )
        if not binary_path:
            raise FileNotFoundError(
                f"SynthStrip engine is not available on this machine. "
                f"{reason} Alternatively, use the 'otsu' engine "
                f"(default, experimental) for development."
            )

        # 2. Build output paths. We use the same filenames the otsu
        # engine produces so downstream steps can find them by the
        # same context key regardless of which engine was selected.
        os.makedirs(output_dir, exist_ok=True)
        brain_path = os.path.join(output_dir, "brain.nii.gz")
        mask_path = os.path.join(output_dir, "brain_mask.nii.gz")

        # 3. Construct the command. SynthStrip's CLI:
        #     mri_synthstrip --i <input> --o <brain> --mask <mask>
        # We use the long form (--i, --o, --mask) for readability.
        # SynthStrip also accepts --no-cuda, --sd (subjects dir), etc.
        # -- these are passed through via params if provided.
        command = [
            binary_path,
            "--i", input_path,
            "--o", brain_path,
            "--mask", mask_path,
        ]

        # Allow advanced users to add extra SynthStrip flags via
        # params["extra_args"] (a list of strings). This is the escape
        # hatch for options we don't model explicitly.
        extra_args = params.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(extra_args)

        # 4. Execute. We capture stdout/stderr so we can include them
        # in the error message if the subprocess fails -- the user
        # needs to see what mri_synthstrip complained about.
        try:
            result = subprocess.run(
                command,
                check=False,  # we check manually so we can include stderr
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            # This shouldn't happen (find_binary already verified it
            # exists), but if the binary disappears between discovery
            # and execution we want a useful error.
            raise FileNotFoundError(
                f"SynthStrip binary disappeared between discovery and "
                f"execution: {binary_path}. {exc}"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"SynthStrip failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 5. Validate outputs. SynthStrip should produce both files
        # when invoked with --o and --mask. If either is missing, we
        # raise a clear error rather than returning a half-success.
        if not os.path.isfile(brain_path):
            raise RuntimeError(
                f"SynthStrip completed but brain output is missing: "
                f"{brain_path}. stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        if not os.path.isfile(mask_path):
            raise RuntimeError(
                f"SynthStrip completed but brain mask output is "
                f"missing: {mask_path}. stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 6. Return the standard result dict. The dispatcher passes
        # this back through dag.py, which writes "brain_mask" into the
        # SubjectContext (via context.update({"brain_mask": mask_path}))
        # -- so subsequent steps operating on sibling acquisitions of
        # the same subject can pick it up via context.get("brain_mask").
        return {
            "output_path": brain_path,
            "outputs": {"brain_mask": mask_path},
            "metrics": {
                "engine": "synthstrip",
                "binary_path": binary_path,
                "returncode": result.returncode,
            },
        }
