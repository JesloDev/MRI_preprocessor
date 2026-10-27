"""
FSL MCFLIRT engine -- real subprocess wrapper (PR12).

MCFLIRT (Motion Correction FLIRT) is FSL's reference-implementation
motion correction for BOLD fMRI. Faster and more validated than the
SimpleITK reference; widely used in published research.

Execution flow:
    1. find_binary(MCFLIRT_DISCOVERY) -> path to mcflirt binary
    2. Construct command: mcflirt -in <input> -out <output_base> -plots
    3. subprocess.run(check=False, capture_output=True)
    4. Validate output exists + read motion params file
    5. Return standard result dict with motion_params path

The -plots flag tells MCFLIRT to write the motion parameters file
(<output_base>.par), which is the standard format consumed by downstream
confound regression steps.

Note: MCFLIRT writes <output_base>.nii.gz (NOT <output_base>.nii.gz.gz
or any other variant). The output_base should NOT include a file
extension -- MCFLIRT appends .nii.gz itself.

License: FSL is released under the Oxford Academic non-commercial
license. Requires separate FSL installation with $FSLDIR set.
"""
import os
import subprocess
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


MCFLIRT_DISCOVERY = DiscoveryConfig(
    engine_id="mcflirt",
    binary_name="mcflirt",
    env_var="MRI_PREPROCESSOR_MCFLIRT",
    tool_suite_env="FSLDIR",
)


class MCFLIRTMotionCorrection:
    """Real subprocess wrapper for FSL MCFLIRT."""

    name = "mcflirt"
    maturity = "external_engine"
    validation_notes = (
        "FSL MCFLIRT is the validated standard for BOLD motion "
        "correction, widely used in published research. Faster than "
        "the SimpleITK reference implementation. Produces results "
        "equivalent to running MCFLIRT directly."
    )
    license_notes = (
        "FSL is released under the Oxford Academic non-commercial "
        "license. Requires separate FSL installation with $FSLDIR set. "
        "NOT for commercial use without a separate commercial license."
    )
    requires_external_tool = True

    def is_available(self) -> bool:
        """Probes for the mcflirt binary. True if found."""
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
        """Executes FSL MCFLIRT as a subprocess.

        Raises:
            FileNotFoundError: if the mcflirt binary isn't found.
            RuntimeError: if the subprocess fails or outputs are missing.
        """
        # 1. Find the binary
        binary_path, reason = find_binary(
            MCFLIRT_DISCOVERY,
            params.get("binary_path"),
        )
        if not binary_path:
            raise FileNotFoundError(
                f"MCFLIRT engine is not available on this machine. "
                f"{reason} Alternatively, use the 'simpleitk' engine "
                f"(default, reference tier) for development."
            )

        os.makedirs(output_dir, exist_ok=True)

        # 2. Build output path. MCFLIRT's convention: -out takes a path
        # WITHOUT an extension, and it appends .nii.gz itself. We use
        # "motion_corrected" as the basename (matches the SimpleITK
        # engine's output naming so downstream code is engine-agnostic).
        output_base = os.path.join(output_dir, "motion_corrected")
        # MCFLIRT writes <output_base>.nii.gz
        out_path = output_base + ".nii.gz"
        # MCFLIRT writes motion params to <output_base>.par when -plots is given
        motion_params_path = output_base + ".par"

        # 3. Construct the command. MCFLIRT's CLI:
        #     mcflirt -in <input> -out <output_base> -plots
        # Optional flags we model explicitly:
        #   -cost <cost>     mutualinfo | woods | coronal | saturn ...
        #   -mats            save transformation matrices
        #   -reffile <file>  use a specific reference volume
        command = [
            binary_path,
            "-in", input_path,
            "-out", output_base,
            "-plots",
        ]

        # Allow the user to specify a reference volume (default: first
        # volume, which is MCFLIRT's default).
        ref_vol = params.get("reference_volume")
        if ref_vol is not None:
            command.extend(["-reffile", str(ref_vol)])

        # Cost function (default: mutualinfo)
        cost = params.get("cost", "mutualinfo")
        if cost:
            command.extend(["-cost", str(cost)])

        # Save transformation matrices (useful for downstream registration)
        if params.get("save_mats", True):
            command.append("-mats")

        # Allow advanced users to add extra flags via params["extra_args"]
        extra_args = params.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(extra_args)

        # 4. Execute
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"mcflirt binary disappeared between discovery and "
                f"execution: {binary_path}. {exc}"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"MCFLIRT failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 5. Validate outputs. MCFLIRT should produce:
        #   - <output_base>.nii.gz (the motion-corrected 4D volume)
        #   - <output_base>.par (the motion parameters: 6 cols per volume)
        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"MCFLIRT completed but motion-corrected output is "
                f"missing: {out_path}. stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        if not os.path.isfile(motion_params_path):
            raise RuntimeError(
                f"MCFLIRT completed but motion parameters file is "
                f"missing: {motion_params_path}. (Was -plots passed?) "
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        # 6. Compute a basic motion summary metric: mean absolute
        # translation (mm). MCFLIRT's .par file format is 6 cols:
        # rx ry rz tx ty tz (rotations in radians, translations in mm).
        mean_abs_translation = None
        try:
            import numpy as np
            params_arr = np.loadtxt(motion_params_path)
            if params_arr.ndim == 1:
                params_arr = params_arr.reshape(1, -1)
            if params_arr.shape[1] >= 6:
                # Cols 4, 5, 6 (1-indexed) are tx, ty, tz
                translations = params_arr[:, 3:6]
                mean_abs_translation = float(
                    np.mean(np.abs(translations))
                )
        except Exception:  # noqa: BLE001 -- metric is best-effort
            pass

        return {
            "output_path": out_path,
            "outputs": {"motion_params": motion_params_path},
            "metrics": {
                "engine": "mcflirt",
                "binary_path": binary_path,
                "returncode": result.returncode,
                "mean_abs_translation_mm": mean_abs_translation,
            },
        }
