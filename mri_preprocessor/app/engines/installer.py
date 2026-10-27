"""
Engine installer: knows how to actually place binaries in the store (PR13).

The store is dumb filesystem ops; the installer is the strategy that
places binaries. We support multiple installation strategies:

  - CondaInstaller    : uses `conda create -p <path>` (or micromamba)
                        to install conda packages into an isolated env
                        inside the store. This is the primary strategy
                        for FSL components (FSL publishes individual
                        conda packages) and other conda-distributed tools.
  - StubInstaller     : for development + testing. Creates a fake
                        binary script that mimics the engine's CLI.
                        NOT for production -- the engine appears
                        available but doesn't actually do real work.
  - BundleInstaller   : (PR15) registers a pre-bundled engine directory
                        in the store WITHOUT any download. This is the
                        offline-distribution strategy: engines are
                        pre-packaged in the application installer and
                        registered at install time.
  - (future) DockerInstaller : pulls a docker image and runs via docker run
  - (future) URLInstaller    : downloads a tarball from a URL

The Installer protocol is intentionally minimal so adding a new
strategy is a 1-file change. The EngineManager doesn't care HOW an
installer places binaries -- only that they end up at
<store>/<id>/<version>/bin/<binary>.

PR15 change: BundleInstaller is the load-bearing piece for the
offline-first distribution model. When the platform is packaged as a
standalone application, engines can be pre-bundled in the installer
and registered in the store via BundleInstaller -- no internet access,
no conda, no micromamba needed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, Optional, Protocol

from app.engines.manifest import EngineManifest
from app.engines.store import EngineStore


class Installer(Protocol):
    """Strategy for placing engine binaries in the store."""

    name: str  # "conda", "stub", "docker", "url"

    def is_available(self) -> bool:
        """True if this installer can run on this machine (e.g. conda
        is on PATH). False means `install()` will raise."""
        ...

    def availability_reason(self) -> str:
        """Returns a human-readable reason when is_available() is False."""
        ...

    def install(self, manifest: EngineManifest,
                 store: EngineStore) -> str:
        """Installs the engine to the store. Returns the version-pinned
        install path.

        Raises:
            RuntimeError: if the install fails (subprocess error,
                          missing prerequisites, etc.)
        """
        ...


# --------------------------------------------------------------------- #
# CondaInstaller (the production strategy)
# --------------------------------------------------------------------- #

class CondaInstaller:
    """Installs engines via `conda create -p <path>` or micromamba.

    FSL publishes its components as individual conda packages
    (https://fsl.fmrib.ox.ac.uk/fsl/docs/install/conda.html), so this
    installer is the primary strategy for FSL engines (eddy, topup,
    mcflirt, etc.). It can also install other conda-distributed tools
    like pydeface.

    The installer creates an isolated conda env at
    `<store>/<id>/<version>/` (no shared prefix), so each engine's
    dependencies can't break another's.
    """
    name = "conda"

    def __init__(self, conda_binary: Optional[str] = None):
        """If conda_binary is given, use it; else auto-discover."""
        self._conda_binary = conda_binary

    def _find_conda(self) -> Optional[str]:
        """Finds conda or micromamba. Prefers micromamba (faster, no
        base env activation needed)."""
        if self._conda_binary and os.path.isfile(self._conda_binary):
            return self._conda_binary
        for name in ("micromamba", "conda", "mamba"):
            path = shutil.which(name)
            if path:
                return path
        return None

    def is_available(self) -> bool:
        return self._find_conda() is not None

    def availability_reason(self) -> str:
        if self.is_available():
            return ""
        return (
            "Conda / micromamba / mamba not found on PATH. Install one "
            "of them to enable engine installation. We recommend "
            "micromamba: see https://mamba.gmuell.se/installation/."
        )

    def install(self, manifest: EngineManifest, store: EngineStore) -> str:
        if manifest.source_type != "conda":
            raise RuntimeError(
                f"CondaInstaller can't install engine {manifest.id!r} "
                f"with source_type={manifest.source_type!r}. Use a "
                f"different installer."
            )
        if not manifest.source_package:
            raise RuntimeError(
                f"Engine {manifest.id!r} manifest is missing source_package "
                f"(the conda package name)."
            )

        conda = self._find_conda()
        if not conda:
            raise RuntimeError(self.availability_reason())

        # The install path becomes the conda env prefix -- so the env
        # IS the engine version dir. This means `conda create -p <path>`
        # gives us an isolated Python env per engine, which is exactly
        # what we want for dependency isolation.
        install_path = store.version_dir(manifest.id, manifest.version)
        os.makedirs(install_path, exist_ok=True)

        # Build the command. We use `-p` (prefix) rather than `-n`
        # (name) so multiple engine versions can coexist without
        # polluting the user's named-env namespace.
        command = [
            conda, "create", "-y",
            "-p", install_path,
            "-c", manifest.source_channel or "defaults",
            f"{manifest.source_package}={manifest.version}",
        ]

        # Run with stderr captured so failures include conda's
        # diagnostics.
        result = subprocess.run(
            command, check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Clean up the partial install dir
            shutil.rmtree(install_path, ignore_errors=True)
            raise RuntimeError(
                f"conda install failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # Mark the engine as installed (writes manifest copy + marker)
        store.mark_installed(manifest)
        return install_path


# --------------------------------------------------------------------- #
# StubInstaller (for development + testing)
# --------------------------------------------------------------------- #

class StubInstaller:
    """Creates a fake binary script in the store.

    For development and testing only -- the engine appears installed
    but the binary is a stub that just exits 0. Useful for:
      - Testing the install/uninstall lifecycle without conda
      - Verifying the EngineManager wiring end-to-end
      - CI environments where conda isn't available

    NOT for production. The stub binary doesn't actually do brain
    extraction / motion correction / etc.
    """
    name = "stub"

    def is_available(self) -> bool:
        """The stub installer is always available -- it doesn't need
        any external tools."""
        return True

    def availability_reason(self) -> str:
        return ""

    def install(self, manifest: EngineManifest, store: EngineStore) -> str:
        install_path = store.version_dir(manifest.id, manifest.version)
        bin_dir = os.path.join(install_path, "bin")
        os.makedirs(bin_dir, exist_ok=True)

        # Use the manifest's binary_name (which matches what the discovery
        # config looks for). Fall back to manifest.id if not set.
        binary_name = manifest.binary_name or manifest.id
        stub_path = os.path.join(bin_dir, binary_name)
        with open(stub_path, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(stub_path, 0o755)

        store.mark_installed(manifest)
        return install_path


# --------------------------------------------------------------------- #
# BundleInstaller (PR15 — offline distribution)
# --------------------------------------------------------------------- #

class BundleInstaller:
    """Registers a pre-bundled engine directory in the store.

    This is the offline-distribution strategy. When the platform is
    packaged as a standalone application, engines can be pre-bundled
    in the installer (e.g. in a `bundles/` directory inside the app)
    and registered in the store at install time -- NO internet access,
    NO conda, NO micromamba needed.

    The bundle directory must have this layout::

        <bundle_path>/
        ├── bin/
        │   └── <binary_name>      (the engine executable)
        ├── lib/                   (optional: shared libraries)
        └── ...                    (any other engine files)

    The BundleInstaller copies (or symlinks) this directory into the
    store at `<store>/<id>/<version>/`, then calls `store.mark_installed()`
    so the EngineManager recognizes it.

    Two modes:
        - copy (default): copies the entire bundle dir into the store.
          Safe, portable, but uses 2x disk space.
        - symlink: creates a symlink from the store path to the bundle
          dir. Zero disk overhead; ideal when the bundle lives in a
          read-only application directory. The symlink target must
          remain stable across application restarts.

    Usage via CLI:
        # Register a pre-bundled engine from a local directory
        python -m app.cli engines install synthstrip --bundle /path/to/bundle

        # Or symlink instead of copy (for read-only bundle locations)
        python -m app.cli engines install synthstrip --bundle /path/to/bundle --symlink
    """
    name = "bundle"

    def __init__(self, bundle_path: Optional[str] = None,
                 symlink: bool = False):
        """Creates a BundleInstaller.

        Args:
            bundle_path: path to the pre-bundled engine directory. If
                None, the installer expects the path to be passed via
                the manifest's `source_url` field (which is a slight
                abuse of the field name but avoids adding a new schema
                field just for bundles).
            symlink: if True, symlink the bundle into the store instead
                of copying. Zero disk overhead; ideal for read-only
                application directories where the bundle already lives.
        """
        self._bundle_path = bundle_path
        self._symlink = symlink

    def is_available(self) -> bool:
        """The BundleInstaller is always available -- it doesn't need
        conda, micromamba, or internet. It only needs the bundle path
        to exist on disk, which is checked at install() time."""
        return True

    def availability_reason(self) -> str:
        return ""

    def _resolve_bundle_path(self, manifest: EngineManifest,
                              platform_id: Optional[str] = None) -> str:
        """Resolves the bundle path, including platform-specific resolution.

        Resolution order:
            1. Explicitly passed to the constructor (self._bundle_path)
               -- can be a direct bundle dir OR a bundle_base containing
               per-platform subdirs. If it's a bundle_base, the platform
               is appended automatically.
            2. manifest.bundle_base (PR16) -- the directory containing
               per-platform subdirs (e.g. /app/bundles/synthstrip/).
               The platform is appended: /app/bundles/synthstrip/linux-64/
            3. manifest.source_url (used as a filesystem path for bundles)
            4. manifest.source_package (fallback: treat as a path)

        For options 1 and 2, if the resolved path is a directory, it's
        used directly. If it contains per-platform subdirs (detected by
        checking for <path>/<platform_id>/bin/), the platform subdir is
        used instead.

        Args:
            manifest: the engine manifest (provides bundle_base, etc.)
            platform_id: optional platform override (for cross-platform
                         preparation). If None, auto-detects the current
                         platform.

        Raises FileNotFoundError if no bundle path is available or the
        path doesn't exist.
        """
        from app.engines.platform import detect_platform

        plat = platform_id or detect_platform()

        # Step 1: explicit bundle_path from constructor
        path = self._bundle_path or manifest.bundle_base \
               or manifest.source_url or manifest.source_package

        if not path:
            raise FileNotFoundError(
                f"BundleInstaller requires a bundle path for engine "
                f"{manifest.id!r}. Pass it via the `bundle_path` "
                f"constructor arg, set `bundle_base` in the manifest, "
                f"or set `source_url` to the bundle directory path."
            )
        path = os.path.expanduser(path)

        # Step 2: check if this is a bundle_base (contains per-platform
        # subdirs). If <path>/<platform>/bin/<binary_name> exists, use
        # the platform-specific subdir. Otherwise, use <path> directly
        # (the caller gave us a ready-to-use bundle dir).
        platform_bundle = os.path.join(path, plat)
        binary_name = manifest.binary_name or manifest.id
        platform_binary = os.path.join(
            platform_bundle, "bin", binary_name,
        )
        if os.path.isfile(platform_binary):
            # Found a platform-specific subdirectory with the binary --
            # use it.
            return platform_bundle

        # No platform-specific subdir found. Check if <path> itself is
        # a ready-to-use bundle (has bin/<binary_name> directly).
        direct_binary = os.path.join(path, "bin", binary_name)
        if os.path.isfile(direct_binary):
            return path

        # Neither path/<platform>/bin/<binary> nor path/bin/<binary> exists.
        # Give the most helpful error.
        if os.path.isdir(path):
            # The dir exists but doesn't have the expected binary layout.
            # Check if it has platform subdirs (just not the right one).
            available_platforms = []
            for p in os.listdir(path):
                if os.path.isfile(os.path.join(path, p, "bin", binary_name)):
                    available_platforms.append(p)
            if available_platforms:
                raise FileNotFoundError(
                    f"Bundle at {path!r} has platform subdirs "
                    f"({', '.join(available_platforms)}) but none for "
                    f"the current platform ({plat!r}). Install the "
                    f"correct platform bundle or use --platform to "
                    f"override."
                )
            raise FileNotFoundError(
                f"Bundle at {path!r} does not contain the expected "
                f"binary: bin/{binary_name} (neither directly nor under "
                f"a platform subdir {plat}/). The bundle directory must "
                f"have bin/<binary_name> or <platform>/bin/<binary_name>."
            )
        raise FileNotFoundError(
            f"Bundle directory not found: {path!r}. The bundle "
            f"must be a directory containing bin/<binary_name> "
            f"or <platform>/bin/<binary_name>."
        )

    def _validate_bundle(self, bundle_path: str,
                          manifest: EngineManifest) -> str:
        """Validates that the bundle contains the expected binary.

        Returns the path to the binary inside the bundle.
        Raises FileNotFoundError if the binary is missing.
        """
        binary_name = manifest.binary_name or manifest.id
        binary_path = os.path.join(bundle_path, "bin", binary_name)
        if not os.path.isfile(binary_path):
            raise FileNotFoundError(
                f"Bundle at {bundle_path!r} does not contain the "
                f"expected binary: bin/{binary_name}. The bundle "
                f"directory must have a bin/ subdirectory with the "
                f"engine's executable."
            )
        if not os.access(binary_path, os.X_OK):
            raise FileNotFoundError(
                f"Bundle binary {binary_path!r} exists but is not "
                f"executable. Check file permissions."
            )
        return binary_path

    def install(self, manifest: EngineManifest, store: EngineStore,
                platform_id: Optional[str] = None) -> str:
        """Registers the pre-bundled engine in the store.

        Copies (or symlinks) the bundle directory into
        <store>/<id>/<version>/, then marks it installed.

        No internet access. No conda. No download. The bundle must
        already exist on disk.

        Args:
            manifest: the engine manifest.
            store: the engine store to install into.
            platform_id: optional platform override (for cross-platform
                        preparation). If None, auto-detects.
        """
        bundle_path = self._resolve_bundle_path(manifest, platform_id)
        self._validate_bundle(bundle_path, manifest)

        install_path = store.version_dir(manifest.id, manifest.version)

        # If forcing a reinstall, the EngineManager already removed the
        # old dir. But if the dir somehow exists (e.g. from a previous
        # failed install), clean it up first.
        if os.path.exists(install_path):
            shutil.rmtree(install_path, ignore_errors=True)
        os.makedirs(os.path.dirname(install_path), exist_ok=True)

        if self._symlink:
            # Create a symlink from the store path to the bundle dir.
            # Zero disk overhead; ideal for read-only application dirs.
            os.symlink(bundle_path, install_path, target_is_directory=True)
        else:
            # Copy the entire bundle dir into the store. This makes the
            # engine self-contained in the store -- the original bundle
            # can be deleted after install.
            shutil.copytree(bundle_path, install_path, symlinks=True)

        # Mark the engine as installed (writes manifest copy + marker)
        store.mark_installed(manifest)
        return install_path
