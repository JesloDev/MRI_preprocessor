"""
pydeface engine -- real subprocess wrapper (PR12).

pydeface is a popular Python package for defacing structural MRI via
template registration. It shells out to FSL tools (FLIRT + a face
template) to identify and remove facial features.

Execution flow:
    1. find_binary(PYDEFACE_DISCOVERY) -> path to the `pydeface` script
    2. Construct command: pydeface <input> --outfile <output> --force
    3. subprocess.run(check=False, capture_output=True)
    4. Validate output exists
    5. Return standard result dict

License: pydeface itself is BSD-3-Clause, but it requires FSL (Oxford
Academic non-commercial license) for the underlying registration.

Tests use a fake `pydeface` shell script (same pattern as the SynthStrip
tests in PR9) so we can exercise the full subprocess + output-validation
flow without an actual pydeface + FSL installation.
"""
import os
import subprocess
from typing import Any, Dict, Optional

from app.engines.binary_discovery import DiscoveryConfig, find_binary


PYDEFACE_DISCOVERY = DiscoveryConfig(
    engine_id="pydeface",
    binary_name="pydeface",
    env_var="MRI_PREPROCESSOR_PYDEFACE",
    # pydeface doesn't have a unified install dir; PATH lookup is the
    # primary discovery (after the env var override).
    tool_suite_env=None,
)


class PyDefaceEngine:
    """Real subprocess wrapper for pydeface.

    When called via the dispatcher:
        defacing_fn(input_path, output_dir, params, context)
            -> params["engine"] = "pydeface"
            -> PyDefaceEngine.run(...)
            -> subprocess.run(["pydeface", input, "--outfile", out, ...])
            -> {output_path: defaced.nii.gz, outputs: {defaced_image: ...}}
    """

    name = "pydeface"
    maturity = "external_engine"
    validation_notes = (
        "pydeface is a validated, widely-used defacing tool. Produces "
        "results equivalent to running pydeface directly. Uses FSL "
        "under the hood for the registration."
    )
    license_notes = (
        "pydeface is BSD-3-Clause, but requires FSL (Oxford Academic "
        "non-commercial license) for the underlying registration. "
        "Requires separate pydeface + FSL installation. NOT for "
        "commercial use without a separate commercial FSL license."
    )
    requires_external_tool = True

    def is_available(self) -> bool:
        """Probes for the pydeface binary. True if found."""
        path, _ = find_binary(PYDEFACE_DISCOVERY)
        return path is not None

    def availability_reason(self) -> str:
        """Returns the discovery-failure message when is_available() is False."""
        path, reason = find_binary(PYDEFACE_DISCOVERY)
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
        """Executes pydeface as a subprocess.

        Raises:
            FileNotFoundError: if the pydeface binary isn't found.
            RuntimeError: if the subprocess fails or the expected output
                file isn't produced.
        """
        # 1. Find the binary
        binary_path, reason = find_binary(
            PYDEFACE_DISCOVERY,
            params.get("binary_path"),
        )
        if not binary_path:
            raise FileNotFoundError(
                f"pydeface engine is not available on this machine. "
                f"{reason} Alternatively, use the 'placeholder' engine "
                f"(default, experimental) for development."
            )

        # 2. Build output path. Use the same filename the placeholder
        # engine produces so downstream code can find it by the same
        # context key regardless of which engine was selected.
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "defaced.nii.gz")

        # 3. Construct the command. pydeface's CLI:
        #     pydeface <input> --outfile <output> [--force] [--nocleanup]
        # --force: overwrite if the output exists
        # --nocleanup: keep intermediate files (useful for debugging)
        command = [
            binary_path,
            input_path,
            "--outfile", out_path,
            "--force",
        ]

        # Allow advanced users to add extra pydeface flags via
        # params["extra_args"] (a list of strings).
        extra_args = params.get("extra_args", [])
        if isinstance(extra_args, list):
            command.extend(extra_args)

        # 4. Execute. We capture stdout/stderr so we can include them
        # in the error message if the subprocess fails.
        try:
            result = subprocess.run(
                command,
                check=False,  # we check manually so we can include stderr
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"pydeface binary disappeared between discovery and "
                f"execution: {binary_path}. {exc}"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"pydeface failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 5. Validate output. pydeface should produce the defaced file
        # when invoked with --outfile. If it's missing, raise a clear
        # error rather than returning a half-success.
        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"pydeface completed but defaced output is missing: "
                f"{out_path}. stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 6. Return the standard result dict.
        return {
            "output_path": out_path,
            "outputs": {"defaced_image": out_path},
            "metrics": {
                "engine": "pydeface",
                "binary_path": binary_path,
                "returncode": result.returncode,
            },
        }
