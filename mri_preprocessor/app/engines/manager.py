"""
Engine manager: central API for engine discovery, installation, and
lookup (PR13).

The EngineManager is the single entry point for:
  - Listing known engines (from manifests)
  - Listing installed engines (from the local store)
  - Checking if an engine is installed
  - Installing an engine (via an Installer strategy)
  - Removing an engine
  - Looking up the binary path for an installed engine

CLI / API / future GUI all use this manager. No business logic is
duplicated -- the manager is the only path to the engine store.

Module-level singleton: `engine_manager`. Use this in CLI / API
calls rather than constructing instances directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.engines.installer import CondaInstaller, Installer, StubInstaller
from app.engines.manifest import EngineManifest, load_all_manifests
from app.engines.store import EngineStore


# Cache of loaded manifests -- loaded once at startup, never re-read
# unless explicitly invalidated (future work: hot-reload on file change).
_MANIFESTS_CACHE: Optional[Dict[str, EngineManifest]] = None


def _get_all_manifests() -> Dict[str, EngineManifest]:
    global _MANIFESTS_CACHE
    if _MANIFESTS_CACHE is None:
        _MANIFESTS_CACHE = load_all_manifests()
    return _MANIFESTS_CACHE


class EngineManager:
    """Central API for engine discovery, installation, and lookup.

    Layered design:
      - manifests  : declarative YAML files (what engines exist)
      - store      : filesystem ops (where installed engines live)
      - installer  : strategy for placing binaries in the store
      - manager    : orchestrates the above three

    The manager is intentionally stateless -- every call reads from
    the live store + manifests. So an engine installed from one CLI
    invocation is immediately visible to the next.
    """

    def __init__(
        self,
        store: Optional[EngineStore] = None,
        installer: Optional[Installer] = None,
        manifests: Optional[Dict[str, EngineManifest]] = None,
    ):
        self.store = store or EngineStore()
        # Default to CondaInstaller (production path); callers can pass
        # a StubInstaller for testing.
        self.installer: Installer = installer or CondaInstaller()
        self._manifests = manifests

    # ----------------------------------------------------------------- #
    # Manifest access
    # ----------------------------------------------------------------- #

    @property
    def manifests(self) -> Dict[str, EngineManifest]:
        """All known manifests (cached at module load)."""
        if self._manifests is not None:
            return self._manifests
        return _get_all_manifests()

    def list_known_engines(self) -> List[EngineManifest]:
        """Returns manifests for all engines the platform knows about
        (whether installed or not)."""
        return list(self.manifests.values())

    def get_manifest(self, engine_id: str) -> Optional[EngineManifest]:
        """Returns the manifest for one engine, or None if unknown."""
        return self.manifests.get(engine_id)

    # ----------------------------------------------------------------- #
    # Installation status
    # ----------------------------------------------------------------- #

    def list_installed_engines(self) -> List[EngineManifest]:
        """Returns manifests for engines installed in the local store."""
        installed_ids = self.store.list_installed()
        result = []
        for engine_id in installed_ids:
            # Prefer the source manifest (more authoritative) but fall
            # back to the local copy if the source has been removed.
            manifest = self.manifests.get(engine_id)
            if manifest is None:
                manifest = self.store._load_local_manifest(engine_id)
            if manifest is not None:
                result.append(manifest)
        return result

    def is_installed(self, engine_id: str) -> bool:
        """True if the engine is in the local store."""
        return self.store.is_installed(engine_id)

    def get_install_path(self, engine_id: str) -> Optional[str]:
        """Returns the version-pinned install path, or None if not installed."""
        return self.store.get_install_path(engine_id)

    def get_binary_path(self, engine_id: str, binary_name: str) -> Optional[str]:
        """Returns the path to a binary inside an installed engine's
        bin/ directory, or None if the engine isn't installed or the
        binary doesn't exist.

        This is the integration point with binary_discovery.py -- when
        the discovery chain looks for a binary, it first asks the
        EngineManager: "is this engine installed locally? if so, where
        is its binary?" before falling back to env vars / PATH /
        tool-suite home.
        """
        manifest = self.manifests.get(engine_id)
        if manifest is None:
            # Unknown engine -- can't be installed
            return None
        version = self.store.get_installed_version(engine_id)
        if version is None:
            return None
        return self.store.get_binary_path(engine_id, version, binary_name)

    # ----------------------------------------------------------------- #
    # Installation lifecycle
    # ----------------------------------------------------------------- #

    def install(self, engine_id: str, force: bool = False) -> str:
        """Installs an engine via the configured Installer.

        Args:
            engine_id: the manifest id to install.
            force: if True, reinstall even if already installed
                   (useful for repairing a broken install).

        Returns:
            The version-pinned install path.

        Raises:
            KeyError: if the engine_id isn't in the manifests.
            RuntimeError: if the installer fails (subprocess error,
                          missing prerequisites, etc.).
        """
        manifest = self.manifests.get(engine_id)
        if manifest is None:
            raise KeyError(f"Unknown engine: {engine_id!r}")

        if self.is_installed(engine_id) and not force:
            # Already installed -- return the existing path
            existing = self.get_install_path(engine_id)
            if existing is not None:
                return existing

        if not self.installer.is_available():
            raise RuntimeError(
                f"Installer '{self.installer.name}' is not available. "
                f"{self.installer.availability_reason()}"
            )

        # If forcing a reinstall, remove the old install first
        if force and self.is_installed(engine_id):
            self.store.remove(engine_id)

        install_path = self.installer.install(manifest, self.store)
        return install_path

    def install_bundle(self, engine_id: str, bundle_path: str,
                        symlink: bool = False, force: bool = False,
                        platform_id: Optional[str] = None) -> str:
        """Registers a pre-bundled engine in the store (PR15).

        This is the offline-distribution path. No internet, no conda,
        no micromamba -- the engine binaries must already exist in the
        bundle directory.

        Args:
            engine_id: the manifest id to install.
            bundle_path: path to the pre-bundled engine directory (PR15)
                          OR a bundle_base containing per-platform subdirs
                          (PR16). Platform resolution is automatic.
            symlink: if True, symlink instead of copy (zero disk
                     overhead; ideal for read-only app dirs).
            force: if True, reinstall even if already installed.
            platform_id: optional platform override (PR16). If None,
                         auto-detects. Useful for cross-platform
                         preparation (e.g. pre-install the macOS arm64
                         bundle from a Linux machine).

        Returns:
            The version-pinned install path.

        Raises:
            KeyError: if the engine_id isn't in the manifests.
            FileNotFoundError: if the bundle directory doesn't exist or
                                doesn't contain the expected binary for
                                the current/overridden platform.
        """
        manifest = self.manifests.get(engine_id)
        if manifest is None:
            raise KeyError(f"Unknown engine: {engine_id!r}")

        if self.is_installed(engine_id) and not force:
            existing = self.get_install_path(engine_id)
            if existing is not None:
                return existing

        if force and self.is_installed(engine_id):
            self.store.remove(engine_id)

        from app.engines.installer import BundleInstaller
        installer = BundleInstaller(bundle_path=bundle_path, symlink=symlink)
        install_path = installer.install(
            manifest, self.store, platform_id=platform_id,
        )
        return install_path

    def uninstall(self, engine_id: str) -> bool:
        """Removes an engine from the local store. Returns True if deleted,
        False if it wasn't installed.

        Raises:
            KeyError: if the engine_id isn't in the manifests.
        """
        if engine_id not in self.manifests:
            raise KeyError(f"Unknown engine: {engine_id!r}")
        return self.store.remove(engine_id)

    # ----------------------------------------------------------------- #
    # Info
    # ----------------------------------------------------------------- #

    def info(self, engine_id: str) -> Dict[str, Any]:
        """Returns the full info dict for an engine: manifest + install
        status + install path + health status + license state.
        Suitable for the CLI `engines info` and the API
        `GET /api/engines/<id>/info`."""
        manifest = self.manifests.get(engine_id)
        if manifest is None:
            raise KeyError(f"Unknown engine: {engine_id!r}")
        installed = self.is_installed(engine_id)
        install_path = self.get_install_path(engine_id) if installed else None
        info = manifest.to_info_dict(installed=installed, install_path=install_path)

        # PR17: add health status + license info
        from app.engines.health import HealthCheck
        health = HealthCheck(self.store, self.manifests)
        health_result = health.check(engine_id, run_binary=False)
        info["health"] = {
            "status": health_result["status"],
            "binary_exists": health_result["binary_exists"],
            "binary_executable": health_result["binary_executable"],
            "license_required": health_result["license_required"],
            "license_valid": health_result["license_valid"],
            "license_path": health_result["license_path"],
            "issues": health_result["issues"],
        }

        # Add installer status so the UI can show "Install" vs
        # "Installer not available" appropriately.
        info["installer"] = {
            "name": self.installer.name,
            "available": self.installer.is_available(),
            "reason": self.installer.availability_reason() if not self.installer.is_available() else "",
        }
        return info

    def health_check(self, engine_id: str,
                      run_binary: bool = True) -> Dict[str, Any]:
        """Runs a health check on one engine (PR17).

        Verifies: installed, binary exists, binary executable, license
        file present (if required), binary runs --version.

        Use this to answer "can I use this engine right now?" vs
        `is_installed()` which only checks if the marker file exists.

        Returns the health check result dict (see HealthCheck.check()).
        """
        from app.engines.health import HealthCheck
        health = HealthCheck(self.store, self.manifests)
        return health.check(engine_id, run_binary=run_binary)

    def get_status(self, engine_id: str) -> str:
        """Returns the status string for one engine (PR17).

        One of: not_installed, installed, ready, broken, license_required.
        This is a convenience wrapper around health_check() that returns
        just the status (not the full result dict).
        """
        return self.health_check(engine_id, run_binary=False)["status"]

    def store_license(self, engine_id: str, license_path: str) -> str:
        """Copies a license file into the engine's install directory (PR17).

        This is what `engines license <id> <path>` calls. After this,
        the health check will report READY (or BROKEN if the binary
        is still missing) instead of LICENSE_REQUIRED.

        Returns the destination path.

        Raises:
            FileNotFoundError: if the engine isn't installed or the
                                license file doesn't exist.
        """
        from app.engines.health import store_license_file
        return store_license_file(engine_id, license_path, self.store)


# --------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------- #

# The singleton is lazily constructed on first access so that test
# fixtures can set $MRI_PREPROCESSOR_ENGINES_DIR BEFORE the store root
# is resolved. If we constructed at import time, the store root would
# be pinned to whatever the env var was at import, and later changes
# (e.g. by monkeypatch in tests) wouldn't take effect.
_engine_manager: Optional[EngineManager] = None


def get_engine_manager() -> EngineManager:
    """Returns the singleton EngineManager, constructing it lazily on
    first access. Use this in CLI / API / binary_discovery calls rather
    than constructing instances directly.

    The lazy construction means $MRI_PREPROCESSOR_ENGINES_DIR is read
    at first call time, not at module import time -- so test fixtures
    that set the env var before the first call get the right store root.
    """
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager()
    return _engine_manager


def _reset_engine_manager() -> None:
    """Forces the singleton to be re-constructed on next access.

    Used by tests that change $MRI_PREPROCESSOR_ENGINES_DIR between
    test cases. Not for production use.
    """
    global _engine_manager
    _engine_manager = None
