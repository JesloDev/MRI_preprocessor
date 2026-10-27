"""
FSL eddy engine -- real subprocess wrapper (PR12).

FSL's `eddy` jointly models susceptibility-induced distortion (via
topup fieldmaps) and eddy-current distortion. It is the validated
standard for production diffusion MRI preprocessing in published
research.

Execution flow:
    1. find_binary(FSL_EDDY_DISCOVERY) -> path to eddy binary
    2. Resolve required inputs from params + SubjectContext:
       - imain   (DWI volume)              -- from input_path
       - mask    (brain mask)              -- from context['brain_mask']
                                              or params['mask_path']
       - acqp    (acquisition parameters)   -- from params['acqp_path']
                                              (NOT auto-discovered; user must supply)
       - index   (volume->acqp mapping)    -- from params['index_path']
       - bvals   (b-values)                -- from context['bval_path']
       - bvecs   (gradient directions)     -- from context['bvec_path']
    3. Construct command:
       eddy --imain=<imain> --mask=<mask> --acqp=<acqp> --index=<index>
            --bvals=<bvals> --bvecs=<bvecs> --out=<output>
       [optionally: --topup=<topup_results_dir>]
    4. subprocess.run(check=False, capture_output=True)
    5. Validate output + rotated bvecs
    6. Return standard result dict

License: FSL is released under the Oxford Academic non-commercial
license. Requires separate FSL installation with $FSLDIR set. CUDA-
capable GPU strongly recommended for performance.

The acqp and index files are dataset-specific and not auto-discoverable
in the current SubjectContext (PR1 discovered bval/bvec but not acqp/
index). The user must pass them via params["acqp_path"] and
params["index_path"]. A future PR could auto-discover them via the
existing sidecar discovery pattern.

Tests use a fake `eddy` shell script (same pattern as SynthStrip).
"""
import os
import subprocess
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


FSL_EDDY_DISCOVERY = DiscoveryConfig(
    engine_id="fsl_eddy",
    binary_name="eddy",
    env_var="MRI_PREPROCESSOR_FSL_EDDY",
    tool_suite_env="FSLDIR",
)


class FSLEddyCorrection:
    """Real subprocess wrapper for FSL eddy + topup."""

    name = "fsl_eddy"
    maturity = "external_engine"
    validation_notes = (
        "FSL eddy + topup is the validated standard for diffusion MRI "
        "distortion correction. Jointly models susceptibility (via "
        "topup fieldmaps) and eddy currents. Produces results equivalent "
        "to running FSL eddy directly."
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

    def _resolve_required_inputs(
        self,
        input_path: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, str]:
        """Resolves all required inputs for the eddy command.

        Returns a dict with keys: imain, mask, acqp, index, bvals, bvecs.
        Raises FileNotFoundError if any required input can't be resolved.

        Resolution sources per input:
            imain  -> input_path (the DWI NIfTI the engine was called on)
            acqp   -> params['acqp_path']  (REQUIRED -- user must supply)
            index  -> params['index_path'] (REQUIRED -- user must supply)
            bvals  -> params['bvals_path'] | context['bval_path']
            bvecs  -> params['bvecs_path'] | context['bvec_path']
            mask   -> params['mask_path']  | context['brain_mask']
                      (the brain mask produced by an earlier brain_extraction
                       step on the same subject)

        The topup results dir is OPTIONAL -- only needed when the user
        has run topup separately and wants eddy to use it.

        Order of checks (load-bearing for UX): we check the user-required
        params (acqp, index) BEFORE the context-derived ones (mask,
        bvals, bvecs). That way if a user forgets --param acqp_path,
        they get the clear 'acqp' error rather than a confusing 'mask'
        error from a system-level auto-resolution failure.
        """
        resolved: Dict[str, str] = {"imain": input_path}

        # acqp (REQUIRED -- not auto-discoverable). Check FIRST so the
        # user gets the actionable error before anything else.
        acqp_path = params.get("acqp_path")
        if not acqp_path:
            raise FileNotFoundError(
                "FSL eddy requires an acquisition parameters file "
                "(--acqp). Pass it via "
                "--param dwi_motion_eddy_correction.acqp_path=<path>. "
                "This file describes the phase-encoding direction and "
                "echo spacing for each acquisition in the DWI series. "
                "It's dataset-specific and must be supplied per job."
            )
        resolved["acqp"] = acqp_path

        # index (REQUIRED -- not auto-discoverable)
        index_path = params.get("index_path")
        if not index_path:
            raise FileNotFoundError(
                "FSL eddy requires an index file. Pass it via "
                "--param dwi_motion_eddy_correction.index_path=<path>. "
                "This file maps each DWI volume to a row in the acqp "
                "file (for multi-acquisition datasets, all 1's for "
                "single-acquisition datasets)."
            )
        resolved["index"] = index_path

        # bvals (auto-resolved from SubjectContext; can be overridden via params)
        bvals_path = params.get("bvals_path") or context.get("bval_path")
        if not bvals_path:
            raise FileNotFoundError(
                "FSL eddy requires a bvals file. The DWI acquisition's "
                "bvals should be auto-discovered from the SubjectContext "
                "(context['bval_path']); if that's not set, pass it via "
                "--param dwi_motion_eddy_correction.bvals_path=<path>."
            )
        resolved["bvals"] = bvals_path

        # bvecs (same)
        bvecs_path = params.get("bvecs_path") or context.get("bvec_path")
        if not bvecs_path:
            raise FileNotFoundError(
                "FSL eddy requires a bvecs file. The DWI acquisition's "
                "bvecs should be auto-discovered from the SubjectContext "
                "(context['bvec_path']); if that's not set, pass it via "
                "--param dwi_motion_eddy_correction.bvecs_path=<path>."
            )
        resolved["bvecs"] = bvecs_path

        # mask
        mask_path = params.get("mask_path") or context.get("brain_mask")
        if not mask_path:
            raise FileNotFoundError(
                "FSL eddy requires a brain mask. Pass it via "
                "--param dwi_motion_eddy_correction.mask_path=<path>, "
                "or ensure brain_extraction ran first (the engine "
                "auto-reads context['brain_mask'])."
            )
        resolved["mask"] = mask_path

        # topup (OPTIONAL -- for susceptibility correction)
        topup_path = params.get("topup_path") or context.get("topup_results")
        if topup_path:
            resolved["topup"] = topup_path

        # Validate all resolved paths exist (skip topup -- it can be a
        # directory, not a file, in FSL's convention).
        for name, path in resolved.items():
            if name == "topup":
                # topup results can be a directory or a basename; FSL
                # appends _fieldcoef.nii.gz etc. We only check that the
                # string is non-empty (already enforced above).
                continue
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"FSL eddy input '{name}' not found at {path!r}."
                )

        return resolved

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes FSL eddy as a subprocess.

        Raises:
            FileNotFoundError: if the eddy binary or any required input
                file isn't found.
            RuntimeError: if the subprocess fails or outputs are missing.
        """
        # 1. Find the binary
        binary_path, reason = find_binary(
            FSL_EDDY_DISCOVERY,
            params.get("binary_path"),
        )
        if not binary_path:
            raise FileNotFoundError(
                f"FSL eddy engine is not available on this machine. "
                f"{reason} Alternatively, use the 'dipy' engine "
                f"(default, reference tier) for development."
            )

        # 2. Resolve required inputs
        inputs = self._resolve_required_inputs(input_path, params, context)

        # 3. Build output paths. eddy writes:
        #   <output_base>.nii.gz            (the corrected DWI)
        #   <output_base>.eddy_parameters   (per-volume parameters)
        # And, if --repol flag is given:
        #   <output_base>.eddy_outlier_report
        # We use "motion_eddy_corrected" as the basename to match the
        # DIPY engine's output naming.
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, "motion_eddy_corrected")
        out_path = output_base + ".nii.gz"

        # 4. Construct the command. eddy's CLI uses --key=value form:
        command = [
            binary_path,
            f"--imain={inputs['imain']}",
            f"--mask={inputs['mask']}",
            f"--acqp={inputs['acqp']}",
            f"--index={inputs['index']}",
            f"--bvals={inputs['bvals']}",
            f"--bvecs={inputs['bvecs']}",
            f"--out={output_base}",
        ]

        # Optional topup integration
        if "topup" in inputs:
            command.append(f"--topup={inputs['topup']}")

        # --repol: detect and replace outlier volumes (recommended default)
        if params.get("repol", True):
            command.append("--repol")

        # --data_is_shelled: hint to eddy that the data is shelled
        # (vs DSI). Auto-detect if not given.
        if "data_is_shelled" in params:
            if params["data_is_shelled"]:
                command.append("--data_is_shelled")

        # Allow advanced users to add extra flags via params["extra_args"]
        extra_args = params.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(extra_args)

        # 5. Execute
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"FSL eddy binary disappeared between discovery and "
                f"execution: {binary_path}. {exc}"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"FSL eddy failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 6. Validate outputs
        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"FSL eddy completed but corrected output is missing: "
                f"{out_path}. stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # eddy also writes a rotated bvecs file when --repol is given:
        # <output_base>.eddy_rotated_bvecs. Surface it as an output if present.
        rotated_bvecs_path = output_base + ".eddy_rotated_bvecs"
        outputs = {}
        if os.path.isfile(rotated_bvecs_path):
            outputs["rotated_bvecs"] = rotated_bvecs_path

        return {
            "output_path": out_path,
            "outputs": outputs,
            "metrics": {
                "engine": "fsl_eddy",
                "binary_path": binary_path,
                "returncode": result.returncode,
                "used_topup": "topup" in inputs,
                "repol": params.get("repol", True),
            },
        }
