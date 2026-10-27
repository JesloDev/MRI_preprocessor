"""
Tests for PR16: OS/architecture-specific engine bundles.

Covers:
  - Platform detection (detect_platform, env var override, auto-detect)
  - Platform display names
  - EngineManifest.platforms field (declared in YAML)
  - BundleInstaller platform-specific path resolution:
    - Direct bundle (bin/<binary> at top level) still works
    - Bundle base with per-platform subdirs (linux-64/bin/<binary>)
    - Wrong platform raises a clear error listing available platforms
    - --platform override works for cross-platform preparation
  - EngineManager.install_bundle(platform_id=...) API
  - CLI: engines install --bundle <base> --platform <id>
"""
import os
import platform as py_platform
import shutil

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.engines.installer import BundleInstaller
from app.engines.manifest import EngineManifest, load_all_manifests
from app.engines.manager import get_engine_manager
from app.engines.platform import (
    ALL_PLATFORMS, LINUX_64, LINUX_AARCH64, OSX_64, OSX_ARM64, WIN_64,
    detect_platform, get_platform_display_name, get_current_platform_display,
    is_native_platform,
)
from app.engines.store import EngineStore


# --------------------------------------------------------------------- #
# Platform detection
# --------------------------------------------------------------------- #

class TestPlatformDetection:
    def test_detect_returns_valid_id(self):
        result = detect_platform()
        assert result in ALL_PLATFORMS

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "osx-arm64")
        assert detect_platform() == "osx-arm64"

    def test_env_var_override_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "invalid-platform")
        # Invalid env var -> auto-detect
        result = detect_platform()
        assert result in ALL_PLATFORMS

    def test_env_var_override_linux_64(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        assert detect_platform() == "linux-64"

    def test_env_var_override_win_64(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "win-64")
        assert detect_platform() == "win-64"

    def test_is_native_platform(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-aarch64")
        assert is_native_platform("linux-aarch64") is True
        assert is_native_platform("linux-64") is False

    def test_display_names(self):
        assert get_platform_display_name("linux-64") == "Linux x86_64"
        assert get_platform_display_name("osx-arm64") == "macOS Apple Silicon"
        assert get_platform_display_name("win-64") == "Windows x86_64"

    def test_current_platform_display(self):
        name = get_current_platform_display()
        # Should be one of the known display names
        assert any(name in get_platform_display_name(p) for p in ALL_PLATFORMS)

    def test_all_five_platforms_defined(self):
        assert len(ALL_PLATFORMS) == 5
        assert "linux-64" in ALL_PLATFORMS
        assert "linux-aarch64" in ALL_PLATFORMS
        assert "osx-64" in ALL_PLATFORMS
        assert "osx-arm64" in ALL_PLATFORMS
        assert "win-64" in ALL_PLATFORMS


# --------------------------------------------------------------------- #
# Manifest: platforms field
# --------------------------------------------------------------------- #

class TestManifestPlatforms:
    def test_all_manifests_have_platforms(self):
        manifests = load_all_manifests()
        for mid, m in manifests.items():
            assert len(m.platforms) > 0, (
                f"Manifest {mid!r} has empty platforms list"
            )

    def test_synthstrip_manifest_platforms(self):
        manifests = load_all_manifests()
        m = manifests["synthstrip"]
        # SynthStrip ships for Linux + macOS (not Windows natively)
        assert "linux-64" in m.platforms
        assert "osx-arm64" in m.platforms
        # All 6 manifests should list at least 2 platforms
        assert len(m.platforms) >= 4  # 4 platforms: linux-64, linux-aarch64, osx-64, osx-arm64

    def test_manifest_to_info_dict_includes_platforms(self):
        manifests = load_all_manifests()
        m = manifests["synthstrip"]
        info = m.to_info_dict()
        assert "platforms" in info
        assert len(info["platforms"]) > 0

    def test_manifest_bundle_base_field(self):
        m = EngineManifest(
            id="test", name="Test", provider="X",
            capability="y", version="1.0", binary_name="test_bin",
            bundle_base="/app/bundles/test",
        )
        assert m.bundle_base == "/app/bundles/test"


# --------------------------------------------------------------------- #
# BundleInstaller: platform-specific path resolution
# --------------------------------------------------------------------- #

def _make_platform_bundle(base_path, platform_id, binary_name="mri_synthstrip",
                            content="#!/bin/bash\nexit 0\n"):
    """Creates a platform-specific bundle subdir."""
    plat_dir = base_path / platform_id
    bin_dir = plat_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / binary_name
    binary.write_text(content)
    binary.chmod(0o755)
    return str(plat_dir)


class TestBundleInstallerPlatformResolution:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.manifests = load_all_manifests()
        self.manifest = self.manifests["synthstrip"]
        self.bundle_base = tmp_path / "bundles" / "synthstrip"
        self.bundle_base.mkdir(parents=True, exist_ok=True)
        # Create platform-specific subdirs for all 5 platforms
        for plat in ALL_PLATFORMS:
            _make_platform_bundle(
                self.bundle_base, plat,
                content=f"#!/bin/bash\necho '{plat}' >&2\nexit 0\n",
            )

    def test_resolve_finds_current_platform_subdir(self, monkeypatch):
        """When bundle_base has <platform>/bin/<binary>, the resolver
        selects the subdir matching the current platform."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        installer = BundleInstaller(bundle_path=str(self.bundle_base))
        path = installer._resolve_bundle_path(self.manifest)
        assert path.endswith("synthstrip/linux-64")

    def test_resolve_with_explicit_platform(self, monkeypatch):
        """--platform osx-arm64 selects the osx-arm64 subdir."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        installer = BundleInstaller(bundle_path=str(self.bundle_base))
        path = installer._resolve_bundle_path(
            self.manifest, platform_id="osx-arm64",
        )
        assert path.endswith("synthstrip/osx-arm64")

    def test_resolve_falls_back_to_direct_bundle(self, tmp_path, monkeypatch):
        """If the bundle dir has bin/<binary> directly (no platform
        subdirs), the resolver uses it as-is."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        direct_bundle = tmp_path / "direct_bundle"
        (direct_bundle / "bin").mkdir(parents=True)
        binary = direct_bundle / "bin" / "mri_synthstrip"
        binary.write_text("#!/bin/bash\nexit 0\n")
        binary.chmod(0o755)

        installer = BundleInstaller(bundle_path=str(direct_bundle))
        path = installer._resolve_bundle_path(self.manifest)
        assert path == str(direct_bundle)

    def test_resolve_wrong_platform_raises_with_available_list(self, monkeypatch):
        """If the current platform isn't in the bundle, the error lists
        what IS available."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "win-64")
        # Create a bundle_base with only Linux + macOS subdirs
        limited_base = self.bundle_base.parent / "limited"
        limited_base.mkdir()
        _make_platform_bundle(limited_base, "linux-64")
        _make_platform_bundle(limited_base, "osx-arm64")

        installer = BundleInstaller(bundle_path=str(limited_base))
        with pytest.raises(FileNotFoundError) as exc_info:
            installer._resolve_bundle_path(self.manifest)
        msg = str(exc_info.value)
        assert "win-64" in msg  # the requested platform
        assert "linux-64" in msg  # available platforms listed
        assert "osx-arm64" in msg

    def test_install_from_bundle_base_selects_correct_platform(self, monkeypatch):
        """End-to-end: install from a bundle_base with per-platform subdirs
        correctly selects the current platform's binary."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        store = EngineStore()
        installer = BundleInstaller(bundle_path=str(self.bundle_base))
        install_path = installer.install(self.manifest, store)

        # Verify the installed binary is the Linux one
        binary_path = os.path.join(install_path, "bin", "mri_synthstrip")
        with open(binary_path) as f:
            content = f.read()
        assert "linux-64" in content

    def test_install_cross_platform_with_override(self, monkeypatch):
        """Install the macOS ARM64 bundle from a Linux machine using
        --platform override."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        store = EngineStore()
        installer = BundleInstaller(bundle_path=str(self.bundle_base))
        install_path = installer.install(
            self.manifest, store, platform_id="osx-arm64",
        )

        # Verify the installed binary is the macOS ARM64 one
        binary_path = os.path.join(install_path, "bin", "mri_synthstrip")
        with open(binary_path) as f:
            content = f.read()
        assert "osx-arm64" in content


# --------------------------------------------------------------------- #
# EngineManager.install_bundle with platform_id
# --------------------------------------------------------------------- #

class TestEngineManagerPlatformBundle:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.manager = get_engine_manager()
        self.bundle_base = tmp_path / "bundles" / "synthstrip"
        self.bundle_base.mkdir(parents=True)
        for plat in ALL_PLATFORMS:
            _make_platform_bundle(
                self.bundle_base, plat,
                content=f"#!/bin/bash\necho '{plat}' >&2\nexit 0\n",
            )

    def test_install_bundle_auto_detects_platform(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        path = self.manager.install_bundle("synthstrip", str(self.bundle_base))
        binary = os.path.join(path, "bin", "mri_synthstrip")
        with open(binary) as f:
            assert "linux-64" in f.read()

    def test_install_bundle_with_platform_override(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        path = self.manager.install_bundle(
            "synthstrip", str(self.bundle_base), platform_id="osx-arm64",
        )
        binary = os.path.join(path, "bin", "mri_synthstrip")
        with open(binary) as f:
            assert "osx-arm64" in f.read()

    def test_install_bundle_wrong_platform_raises(self, monkeypatch):
        """If the bundle_base doesn't have a subdir for the requested
        platform, install raises FileNotFoundError."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "win-64")
        # Bundle only has linux + mac subdirs
        limited_base = self.bundle_base.parent / "limited_synthstrip"
        limited_base.mkdir()
        _make_platform_bundle(limited_base, "linux-64")
        _make_platform_bundle(limited_base, "osx-arm64")

        with pytest.raises(FileNotFoundError) as exc_info:
            self.manager.install_bundle(
                "synthstrip", str(limited_base), platform_id="win-64",
            )
        assert "win-64" in str(exc_info.value)
        assert "linux-64" in str(exc_info.value)  # lists available


# --------------------------------------------------------------------- #
# CLI: --platform flag
# --------------------------------------------------------------------- #

runner = CliRunner()


class TestCLIPlatformBundle:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.bundle_base = tmp_path / "bundles" / "synthstrip"
        self.bundle_base.mkdir(parents=True)
        for plat in ALL_PLATFORMS:
            _make_platform_bundle(
                self.bundle_base, plat,
                content=f"#!/bin/bash\necho '{plat}' >&2\nexit 0\n",
            )

    def test_cli_bundle_install_auto_detects_platform(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", str(self.bundle_base),
        ])
        assert result.exit_code == 0
        assert "Linux x86_64" in result.stdout
        assert "linux-64" in result.stdout

    def test_cli_bundle_install_with_platform_override(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", str(self.bundle_base),
            "--platform", "osx-arm64",
        ])
        assert result.exit_code == 0
        assert "macOS Apple Silicon" in result.stdout
        assert "osx-arm64" in result.stdout

    def test_cli_bundle_install_wrong_platform_fails(self, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "win-64")
        limited_base = self.bundle_base.parent / "limited"
        limited_base.mkdir()
        _make_platform_bundle(limited_base, "linux-64")
        _make_platform_bundle(limited_base, "osx-arm64")

        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", str(limited_base),
            "--platform", "win-64",
        ])
        assert result.exit_code != 0
        assert "win-64" in result.stdout
        assert "linux-64" in result.stdout  # lists available

    def test_cli_direct_bundle_still_works_without_platform(self, tmp_path, monkeypatch):
        """A direct bundle (bin/<binary> at top level, no platform
        subdirs) still works without --platform."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")
        direct_bundle = tmp_path / "direct"
        (direct_bundle / "bin").mkdir(parents=True)
        binary = direct_bundle / "bin" / "mri_synthstrip"
        binary.write_text("#!/bin/bash\nexit 0\n")
        binary.chmod(0o755)

        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", str(direct_bundle),
        ])
        assert result.exit_code == 0
        assert "Installed" in result.stdout
