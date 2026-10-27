"""
Engine store: filesystem layout for installed engines (PR13).

The store is the local directory where the EngineManager installs
external engines. Each engine gets its own subdirectory, with the
version pinned in the path so multiple versions can coexist:

    engines/                                    # store root
    ├── synthstrip/
    │   ├── 7.4.1/                              # version-pinned install
    │   │   ├── bin/
    │   │   │   └── mri_synthstrip
    │   │   ├── lib/
    │   │   └── ...
    │   ├── manifest.yaml                       # copy of the source manifest
    │   └── installed                           # marker file: install date
    ├── eddy/
    │   └── ...
    └── ...

Store location (priority):
    1. $MRI_PREPROCESSOR_ENGINES_DIR env var (test override + custom deployments)
    2. <project_root>/engines/                  # default, alongside app/

Keeping the store at the project root (NOT inside app/) is intentional:
the store holds actual engine installations (potentially gigabytes of
binaries + libs), not Python adapter code. Mixing them would make the
app/ package bloated and confuse packaging tools.
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Optional

from app.engines.manifest import EngineManifest


ENV_VAR = "MRI_PREPROCESSOR_ENGINES_DIR"


def get_default_store_root() -> str:
    """Returns the default store root: <project_root>/engines/.

    The project root is two levels up from this file (app/engines/store.py
    -> app/engines/ -> app/ -> project_root/).
    """
    # app/engines/store.py -> app/engines/ -> app/ -> project_root/
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(project_root, "engines")


def get_store_root() -> str:
    """Returns the active store root, creating it if it doesn't exist.

    Resolution order:
        1. $MRI_PREPROCESSOR_ENGINES_DIR (explicit override)
        2. <project_root>/engines/        (default)
    """
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        store_root = os.path.expanduser(env_path)
    else:
        store_root = get_default_store_root()
    os.makedirs(store_root, exist_ok=True)
    return store_root


class EngineStore:
    """Filesystem operations for the local engine store.

    The store is intentionally dumb: it knows how to create / detect /
    remove engine directories, but it doesn't know how to install the
    actual binaries (that's the Installer's job). This separation lets
    us swap installers (conda, micromamba, docker pull, etc.) without
    touching the store layout.
    """

    def __init__(self, root: Optional[str] = None):
        self.root = root or get_store_root()
        os.makedirs(self.root, exist_ok=True)

    # ----------------------------------------------------------------- #
    # Path helpers
    # ----------------------------------------------------------------- #

    def engine_dir(self, engine_id: str) -> str:
        """Returns the per-engine directory (contains version subdirs
        + manifest.yaml + installed marker)."""
        return os.path.join(self.root, engine_id)

    def version_dir(self, engine_id: str, version: str) -> str:
        """Returns the version-pinned install directory for an engine."""
        return os.path.join(self.engine_dir(engine_id), version)

    def manifest_path(self, engine_id: str) -> str:
        """Returns the path to the engine's local manifest copy."""
        return os.path.join(self.engine_dir(engine_id), "manifest.yaml")

    def installed_marker(self, engine_id: str) -> str:
        """Returns the path to the 'installed' marker file.

        The marker file's mtime is the install date; its content is
        a human-readable install timestamp.
        """
        return os.path.join(self.engine_dir(engine_id), "installed")

    def bin_dir(self, engine_id: str, version: str) -> str:
        """Returns the bin/ directory inside a version-pinned install."""
        return os.path.join(self.version_dir(engine_id, version), "bin")

    def get_binary_path(self, engine_id: str, version: str,
                         binary_name: str) -> Optional[str]:
        """Returns the path to a specific binary inside the engine's
        bin/ directory, or None if the file doesn't exist.

        This is what the binary_discovery chain calls BEFORE checking
        env vars / PATH / tool-suite home -- if an engine is locally
        installed, we use the local install, not whatever happens to be
        on PATH (which might be a different version).
        """
        candidate = os.path.join(self.bin_dir(engine_id, version), binary_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return None

    # ----------------------------------------------------------------- #
    # Installation lifecycle
    # ----------------------------------------------------------------- #

    def is_installed(self, engine_id: str) -> bool:
        """True if the engine has been installed (marker file exists)."""
        return os.path.isfile(self.installed_marker(engine_id))

    def get_installed_version(self, engine_id: str) -> Optional[str]:
        """Returns the installed version string, or None if not installed.

        Reads from the local manifest.yaml copy.
        """
        if not self.is_installed(engine_id):
            return None
        manifest = self._load_local_manifest(engine_id)
        return manifest.version if manifest else None

    def get_install_path(self, engine_id: str) -> Optional[str]:
        """Returns the version-pinned install path, or None if not installed."""
        version = self.get_installed_version(engine_id)
        if version is None:
            return None
        return self.version_dir(engine_id, version)

    def mark_installed(self, manifest: EngineManifest) -> str:
        """Creates the engine directory + version dir + manifest copy +
        installed marker. Returns the version-pinned install path.

        Called by the Installer after it has placed the binaries in
        <version_dir>/bin/. The store is responsible for recording the
        metadata; the installer is responsible for placing the binaries.
        """
        engine_dir = self.engine_dir(manifest.id)
        version_dir = self.version_dir(manifest.id, manifest.version)
        os.makedirs(version_dir, exist_ok=True)
        os.makedirs(os.path.join(version_dir, "bin"), exist_ok=True)

        # Write the manifest copy (so future runs can verify what's
        # installed without re-reading the source manifests)
        with open(self.manifest_path(manifest.id), "w", encoding="utf-8") as f:
            import yaml as _yaml
            _yaml.safe_dump(manifest.model_dump(), f, sort_keys=False)

        # Write the installed marker
        with open(self.installed_marker(manifest.id), "w", encoding="utf-8") as f:
            f.write(f"installed={time.time()}\nversion={manifest.version}\n")

        return version_dir

    def remove(self, engine_id: str) -> bool:
        """Removes the engine directory entirely. Returns True if deleted,
        False if the engine wasn't installed."""
        engine_dir = self.engine_dir(engine_id)
        if not os.path.isdir(engine_dir):
            return False
        shutil.rmtree(engine_dir)
        return True

    def list_installed(self) -> list:
        """Returns the engine_ids of all installed engines."""
        if not os.path.isdir(self.root):
            return []
        result = []
        for entry in sorted(os.listdir(self.root)):
            engine_dir = os.path.join(self.root, entry)
            if not os.path.isdir(engine_dir):
                continue
            if os.path.isfile(os.path.join(engine_dir, "installed")):
                result.append(entry)
        return result

    def _load_local_manifest(self, engine_id: str) -> Optional[EngineManifest]:
        """Loads the local manifest.yaml copy for an installed engine."""
        path = self.manifest_path(engine_id)
        if not os.path.isfile(path):
            return None
        try:
            from app.engines.manifest import load_manifest
            return load_manifest(path)
        except Exception:  # noqa: BLE001 -- corrupt manifest, skip
            return None
