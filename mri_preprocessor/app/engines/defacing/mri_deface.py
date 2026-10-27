"""
mri_deface engine -- real subprocess wrapper (PR12).

FreeSurfer's mri_deface is an atlas-based defacing tool. It registers
a face atlas + skull atlas to the subject's image, then removes the
matched region. Produces validated, irreversible facial-feature removal.

Execution flow:
    1. find_binary(MRI_DEFACE_DISCOVERY) -> path to mri_deface binary
    2. Locate the face + skull atlas templates (typically in
       $FREESURFER_HOME/subjects/face/face_gca/face.gca and
       .../aseg.gca)
    3. Construct command: mri_deface <input> <face.gca> <aseg.gca> <output>
    4. subprocess.run(check=False, capture_output=True)
    5. Validate output exists
    6. Return standard result dict

License: mri_deface is part of FreeSurfer (non-commercial). Requires
FreeSurfer installation + a license file at $FREESURFER_HOME/license.txt.

Tests use a fake `mri_deface` shell script (same pattern as the
SynthStrip tests in PR9).
"""
import os
import subprocess
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


MRI_DEFACE_DISCOVERY = DiscoveryConfig(
    engine_id="mri_deface",
    binary_name="mri_deface",
    env_var="MRI_PREPROCESSOR_MRI_DEFACE",
    tool_suite_env="FREESURFER_HOME",
)


# Default atlas locations relative to $FREESURFER_HOME. These are the
# standard paths shipped with FreeSurfer 6/7. Users with non-standard
# installations can override via params["face_template"] and
# params["skull_template"].
DEFAULT_FACE_TEMPLATE = "subjects/face/face_gca/face.gca"
DEFAULT_SKULL_TEMPLATE = "subjects/face/face_gca/aseg.gca"


class MRIDefaceEngine:
    """Real subprocess wrapper for FreeSurfer mri_deface."""

    name = "mri_deface"
    maturity = "external_engine"
    validation_notes = (
        "mri_deface is a validated atlas-based defacing tool from the "
        "FreeSurfer team. Produces results equivalent to running "
        "mri_deface directly."
    )
    license_notes = (
        "mri_deface is part of FreeSurfer (non-commercial license). "
        "Requires FreeSurfer installation and a license file at "
        "$FREESURFER_HOME/license.txt. NOT for commercial use without "
        "a separate commercial license."
    )
    requires_external_tool = True

    def is_available(self) -> bool:
        """Probes for the mri_deface binary. True if found."""
        path, _ = find_binary(MRI_DEFACE_DISCOVERY)
        return path is not None

    def availability_reason(self) -> str:
        path, reason = find_binary(MRI_DEFACE_DISCOVERY)
        if path:
            return ""
        return reason

    def _resolve_templates(
        self,
        params: Dict[str, Any],
    ) -> tuple:
        """Resolves the face and skull atlas template paths.

        Resolution order per template:
            1. params["face_template"] / params["skull_template"] -- explicit
               override (useful for non-standard FreeSurfer installations
               or custom atlases).
            2. $FREESURFER_HOME/<DEFAULT_FACE_TEMPLATE> / .../<DEFAULT_SKULL_TEMPLATE>
               -- standard FreeSurfer install layout.

        Raises FileNotFoundError if neither resolution yields a path.
        """
        face_template = params.get("face_template")
        skull_template = params.get("skull_template")

        if face_template is None or skull_template is None:
            fs_home = os.environ.get("FREESURFER_HOME")
            if not fs_home:
                raise FileNotFoundError(
                    "mri_deface requires the face + skull atlas templates. "
                    "Either pass --param defacing.face_template=<path> and "
                    "--param defacing.skull_template=<path>, or set "
                    "$FREESURFER_HOME so the engine can locate the default "
                    "templates at "
                    f"$FREESURFER_HOME/{DEFAULT_FACE_TEMPLATE} and "
                    f"$FREESURFER_HOME/{DEFAULT_SKULL_TEMPLATE}."
                )
            if face_template is None:
                face_template = os.path.join(fs_home, DEFAULT_FACE_TEMPLATE)
            if skull_template is None:
                skull_template = os.path.join(fs_home, DEFAULT_SKULL_TEMPLATE)

        # Validate the template files exist -- if they don't, mri_deface
        # will fail with a confusing error; better to fail fast here.
        for tmpl, name in ((face_template, "face_template"),
                            (skull_template, "skull_template")):
            if not os.path.isfile(tmpl):
                raise FileNotFoundError(
                    f"mri_deface {name} not found at {tmpl!r}. Pass "
                    f"--param defacing.{name}=<path> to override."
                )

        return face_template, skull_template

    def run(
        self,
        input_path: str,
        output_dir: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes mri_deface as a subprocess.

        Raises:
            FileNotFoundError: if the mri_deface binary or atlas templates
                aren't found.
            RuntimeError: if the subprocess fails or the output isn't
                produced.
        """
        # 1. Find the binary
        binary_path, reason = find_binary(
            MRI_DEFACE_DISCOVERY,
            params.get("binary_path"),
        )
        if not binary_path:
            raise FileNotFoundError(
                f"mri_deface engine is not available on this machine. "
                f"{reason} Alternatively, use the 'placeholder' engine "
                f"(default, experimental) for development."
            )

        # 2. Resolve the atlas templates
        face_template, skull_template = self._resolve_templates(params)

        # 3. Build output path
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "defaced.nii.gz")

        # 4. Construct the command. mri_deface's CLI:
        #     mri_deface <input> <face_template> <skull_template> <output>
        command = [
            binary_path,
            input_path,
            face_template,
            skull_template,
            out_path,
        ]

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
                f"mri_deface binary disappeared between discovery and "
                f"execution: {binary_path}. {exc}"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"mri_deface failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 6. Validate output
        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"mri_deface completed but defaced output is missing: "
                f"{out_path}. stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        return {
            "output_path": out_path,
            "outputs": {"defaced_image": out_path},
            "metrics": {
                "engine": "mri_deface",
                "binary_path": binary_path,
                "face_template": face_template,
                "skull_template": skull_template,
                "returncode": result.returncode,
            },
        }
