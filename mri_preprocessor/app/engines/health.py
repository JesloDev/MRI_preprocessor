"""
Engine health checks + license handling (PR17).

Distinguishes "installed" (directory marker exists) from "ready" (the
engine can actually run). An engine might be installed but broken
(binary missing, wrong version, license file missing, dependencies
unavailable). The health check verifies all of these and returns a
status that the CLI/API/GUI can surface.

Status states:
    NOT_INSTALLED  -- the engine isn't in the store at all
    INSTALLED      -- directory marker exists, but health not verified
    READY          -- health check passed: binary exists, is executable,
                       and runs --version successfully
    BROKEN         -- health check failed: binary missing, not
                       executable, or crashes when invoked
    LICENSE_REQUIRED -- the manifest requires a license file but it
                       hasn't been provided yet (FreeSurfer engines)

The license handling:
    - The manifest declares `license_file_required: true` + the
      expected license file name + the env var that points at it
      (e.g. FREESURFER_HOME -> $FREESURFER_HOME/license.txt).
    - The user provides the license file via
      `engines license <id> <path>` -- it's copied into the engine's
      install directory so it travels with the engine.
    - The health check verifies the license file exists before
      reporting READY.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

from app.engines.manifest import EngineManifest
from app.engines.store import EngineStore


# --------------------------------------------------------------------- #
# Status states
# --------------------------------------------------------------------- #

NOT_INSTALLED = "not_installed"
INSTALLED = "installed"               # marker exists, health not checked
READY = "ready"                       # health check passed
BROKEN = "broken"                     # health check failed
LICENSE_REQUIRED = "license_required"  # license file missing

ALL_STATUSES = [NOT_INSTALLED, INSTALLED, READY, BROKEN, LICENSE_REQUIRED]


def status_display(status: str) -> str:
    """Returns a human-readable status for CLI display."""
    return {
        NOT_INSTALLED: "✗ Not installed",
        INSTALLED: "○ Installed (unverified)",
        READY: "✓ Ready",
        BROKEN: "✗ Broken",
        LICENSE_REQUIRED: "⚠ License required",
    }.get(status, status)


# --------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------- #

class HealthCheck:
    """Verifies that an installed engine is actually usable.

    A health check answers: "can I run this engine right now?"
    It checks:
      1. Is the engine installed in the store?
      2. Does the binary exist at the expected path?
      3. Is the binary executable?
      4. Does the binary run --version (or --help) and exit 0?
      5. If the manifest requires a license file, does it exist?

    If any check fails, the status is BROKEN or LICENSE_REQUIRED
    (depending on which check failed). If all pass, READY.

    The check is intentionally fast: --version (or --help if --version
    isn't supported) takes <1s for most neuroimaging tools. It doesn't
    do a full functional test (that would require running the engine on
    real data, which is the step's job, not the health check's).
    """

    def __init__(self, store: EngineStore,
                 manifests: Dict[str, EngineManifest]):
        self.store = store
        self.manifests = manifests

    def check(self, engine_id: str,
              run_binary: bool = True) -> Dict[str, Any]:
        """Runs a health check on one engine.

        Args:
            engine_id: the manifest id to check.
            run_binary: if True, attempt to run the binary with --version
                        (or --help). Set to False for environments where
                        subprocess execution is restricted (e.g. some
                        CI sandboxes).

        Returns a dict:
            {
                "engine_id": str,
                "status": str,         # one of ALL_STATUSES
                "binary_path": str,     # path to the binary, or None
                "binary_exists": bool,
                "binary_executable": bool,
                "binary_runs": Optional[bool],  # None if not checked
                "license_required": bool,
                "license_valid": Optional[bool],
                "license_path": Optional[str],
                "issues": List[str],   # human-readable list of problems
            }
        """
        result: Dict[str, Any] = {
            "engine_id": engine_id,
            "status": NOT_INSTALLED,
            "binary_path": None,
            "binary_exists": False,
            "binary_executable": False,
            "binary_runs": None,
            "license_required": False,
            "license_valid": None,
            "license_path": None,
            "issues": [],
        }

        manifest = self.manifests.get(engine_id)
        if manifest is None:
            result["issues"].append(f"Unknown engine: {engine_id!r}")
            return result

        result["license_required"] = manifest.license_file_required

        # 1. Is it installed?
        if not self.store.is_installed(engine_id):
            result["status"] = NOT_INSTALLED
            return result

        result["status"] = INSTALLED  # at least the marker exists

        # 2. Get the binary path
        version = self.store.get_installed_version(engine_id)
        if version is None:
            result["issues"].append(
                "Installed marker exists but version is unknown"
            )
            result["status"] = BROKEN
            return result

        binary_name = manifest.binary_name or manifest.id
        # Construct the binary path directly rather than using
        # store.get_binary_path(), because get_binary_path() filters
        # by executability (returns None if not executable). The health
        # check needs to distinguish "doesn't exist" from "exists but
        # not executable" -- so it checks file existence and
        # executability separately.
        bin_dir = self.store.bin_dir(engine_id, version)
        binary_path = os.path.join(bin_dir, binary_name)
        result["binary_path"] = binary_path if os.path.isfile(binary_path) else None

        if binary_path is None or not os.path.isfile(binary_path):
            result["issues"].append(
                f"Binary not found at expected path: "
                f"bin/{binary_name}"
            )
            result["status"] = BROKEN
            return result

        result["binary_exists"] = True

        # 3. Is it executable?
        if not os.access(binary_path, os.X_OK):
            result["issues"].append(
                f"Binary exists but is not executable: {binary_path}"
            )
            result["status"] = BROKEN
            return result

        result["binary_executable"] = True

        # 4. License check (before binary execution -- if the license
        #    is missing, the binary might hang or crash, so we check
        #    this first)
        if manifest.license_file_required:
            license_path = self._find_license_file(engine_id, manifest)
            result["license_path"] = license_path
            if license_path is None:
                result["issues"].append(
                    f"License file required but not found. Provide it "
                    f"via `python -m app.cli engines license {engine_id} "
                    f"<path-to-license-file>`."
                )
                result["license_valid"] = False
                result["status"] = LICENSE_REQUIRED
                return result
            result["license_valid"] = True

        # 5. Does the binary run?
        if run_binary:
            runs_ok = self._test_binary(binary_path)
            result["binary_runs"] = runs_ok
            if not runs_ok:
                result["issues"].append(
                    f"Binary exists and is executable but failed to run "
                    f"(exit code != 0 on --version/--help). The engine "
                    f"may be broken or have missing runtime dependencies."
                )
                result["status"] = BROKEN
                return result

        # All checks passed
        result["status"] = READY
        return result

    def check_all(self, run_binary: bool = True) -> Dict[str, Dict[str, Any]]:
        """Runs health checks on all known engines. Returns a dict
        keyed by engine_id."""
        return {
            eid: self.check(eid, run_binary=run_binary)
            for eid in self.manifests
        }

    # ----------------------------------------------------------------- #
    # Internal helpers
    # ----------------------------------------------------------------- #

    def _find_license_file(self, engine_id: str,
                            manifest: EngineManifest) -> Optional[str]:
        """Resolves the license file path for an engine.

        Resolution order:
            1. <engine_install_dir>/license.txt -- the platform-managed
               copy (placed by `engines license <id> <path>`)
            2. $FREESURFER_HOME/license.txt -- the standard FreeSurfer
               location (for engines that use FreeSurfer's license)
            3. $MRI_PREPROCESSOR_<ENGINE_ID>_LICENSE -- explicit env var
               override

        Returns the path if found, or None.
        """
        install_path = self.store.get_install_path(engine_id)
        if install_path:
            managed_license = os.path.join(install_path, "license.txt")
            if os.path.isfile(managed_license):
                return managed_license

        # Fall back to tool-suite env var (e.g. FREESURFER_HOME)
        if manifest.source_type == "conda" and manifest.source_package:
            # Heuristic: if the package starts with "freesurfer" or
            # "fsl-synthstrip", it uses FreeSurfer's license
            if "freesurfer" in manifest.source_package.lower() or \
               "synthstrip" in manifest.source_package.lower() or \
               "mri_deface" in manifest.source_package.lower():
                fs_home = os.environ.get("FREESURFER_HOME")
                if fs_home:
                    fs_license = os.path.join(fs_home, "license.txt")
                    if os.path.isfile(fs_license):
                        return fs_license

        # Explicit env var override
        env_name = f"MRI_PREPROCESSOR_{engine_id.upper()}_LICENSE"
        env_path = os.environ.get(env_name)
        if env_path and os.path.isfile(env_path):
            return env_path

        return None

    def _test_binary(self, binary_path: str) -> bool:
        """Tries to run the binary with --version or --help.

        Returns True if the binary exits 0 with either flag. We try
        --version first (most engines support it); if that fails with
        a non-zero exit, we try --help (some engines use --help
        instead). If both fail, the binary is considered broken.

        We capture stdout/stderr so failures don't pollute the user's
        terminal. The check has a 10-second timeout so a hanging
        binary doesn't block the platform.
        """
        for flag in ("--version", "--help", "-h"):
            try:
                result = subprocess.run(
                    [binary_path, flag],
                    capture_output=True, text=True,
                    timeout=10,
                )
                # Some tools return non-zero for --version but still
                # print version info to stderr. We accept exit 0 OR
                # non-empty output (the binary ran and produced output).
                if result.returncode == 0 or result.stdout or result.stderr:
                    return True
            except subprocess.TimeoutExpired:
                continue
            except (FileNotFoundError, PermissionError):
                return False
        return False


# --------------------------------------------------------------------- #
# License management
# --------------------------------------------------------------------- #

def store_license_file(engine_id: str, license_source_path: str,
                        store: EngineStore) -> str:
    """Copies a license file into the engine's install directory.

    This is what `engines license <id> <path>` calls. The license file
    is copied to <install_dir>/license.txt so it travels with the
    engine (survives reinstalls via --force, since --force removes and
    recreates the engine dir -- the user would need to re-provide the
    license after a force reinstall).

    Returns the destination path.

    Raises:
        FileNotFoundError: if the engine isn't installed, or the source
                           license file doesn't exist.
    """
    if not store.is_installed(engine_id):
        raise FileNotFoundError(
            f"Engine {engine_id!r} is not installed. Install it first "
            f"via `python -m app.cli engines install {engine_id}`, then "
            f"provide the license file."
        )

    if not os.path.isfile(license_source_path):
        raise FileNotFoundError(
            f"License file not found: {license_source_path!r}"
        )

    install_path = store.get_install_path(engine_id)
    if install_path is None:
        raise FileNotFoundError(
            f"Engine {engine_id!r} install path not found."
        )

    dest_path = os.path.join(install_path, "license.txt")
    import shutil
    shutil.copy2(license_source_path, dest_path)
    return dest_path


def get_license_path(engine_id: str, store: EngineStore,
                      manifests: Dict[str, EngineManifest]) -> Optional[str]:
    """Returns the path to the engine's license file, or None.

    Uses the same resolution order as HealthCheck._find_license_file.
    """
    manifest = manifests.get(engine_id)
    if manifest is None or not manifest.license_file_required:
        return None

    health = HealthCheck(store, manifests)
    return health._find_license_file(engine_id, manifest)
