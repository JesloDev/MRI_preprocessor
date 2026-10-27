"""
Tests for PR17: engine health checks + license handling.

Covers:
  - Status states (NOT_INSTALLED, INSTALLED, READY, BROKEN, LICENSE_REQUIRED)
  - HealthCheck.check() for each state:
      - not installed -> NOT_INSTALLED
      - installed, binary missing -> BROKEN
      - installed, binary not executable -> BROKEN
      - installed, license required but missing -> LICENSE_REQUIRED
      - installed, license valid -> READY (with --no-run-binary)
      - installed, license not required -> READY
  - License file management:
      - store_license_file() copies license into engine dir
      - get_license_path() resolves from engine dir / FREESURFER_HOME / env var
      - store_license_file() raises if engine not installed
      - store_license_file() raises if source file doesn't exist
  - EngineManager API: health_check(), get_status(), store_license()
  - CLI: engines health, engines license
  - engines info now shows health + license state
"""
import os
import shutil

import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.engines.health import (
    ALL_STATUSES, BROKEN, INSTALLED, LICENSE_REQUIRED, NOT_INSTALLED,
    READY, HealthCheck, get_license_path, status_display,
    store_license_file,
)
from app.engines.installer import StubInstaller
from app.engines.manifest import EngineManifest, load_all_manifests
from app.engines.manager import EngineManager, get_engine_manager
from app.engines.store import EngineStore


# --------------------------------------------------------------------- #
# Status states + display
# --------------------------------------------------------------------- #

def test_all_five_statuses_defined():
    assert len(ALL_STATUSES) == 5
    assert NOT_INSTALLED in ALL_STATUSES
    assert INSTALLED in ALL_STATUSES
    assert READY in ALL_STATUSES
    assert BROKEN in ALL_STATUSES
    assert LICENSE_REQUIRED in ALL_STATUSES

def test_status_display_returns_readable_string():
    assert "Not installed" in status_display(NOT_INSTALLED)
    assert "Ready" in status_display(READY)
    assert "Broken" in status_display(BROKEN)
    assert "License" in status_display(LICENSE_REQUIRED)


# --------------------------------------------------------------------- #
# HealthCheck
# --------------------------------------------------------------------- #

class TestHealthCheck:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.store = EngineStore()
        self.manifests = load_all_manifests()
        self.health = HealthCheck(self.store, self.manifests)

    def test_not_installed_returns_not_installed(self):
        result = self.health.check("synthstrip", run_binary=False)
        assert result["status"] == NOT_INSTALLED
        assert result["binary_exists"] is False

    def test_installed_stub_binary_ready_without_run(self):
        """A stub-installed engine with no license requirement should
        be READY (with run_binary=False)."""
        # Install mcflirt (no license required)
        manager = EngineManager(
            store=self.store, installer=StubInstaller(),
            manifests=self.manifests,
        )
        manager.install("mcflirt")
        result = self.health.check("mcflirt", run_binary=False)
        assert result["status"] == READY
        assert result["binary_exists"] is True
        assert result["binary_executable"] is True

    def test_installed_license_required_reports_license_required(self):
        """SynthStrip requires a license file. Without it, status is
        LICENSE_REQUIRED (not BROKEN -- the binary IS fine, the license
        is the only missing piece)."""
        manager = EngineManager(
            store=self.store, installer=StubInstaller(),
            manifests=self.manifests,
        )
        manager.install("synthstrip")
        result = self.health.check("synthstrip", run_binary=False)
        assert result["status"] == LICENSE_REQUIRED
        assert result["binary_exists"] is True
        assert result["license_required"] is True
        assert result["license_valid"] is False
        assert any("license" in issue.lower() for issue in result["issues"])

    def test_installed_with_license_becomes_ready(self):
        """After providing a license file, status becomes READY."""
        manager = EngineManager(
            store=self.store, installer=StubInstaller(),
            manifests=self.manifests,
        )
        manager.install("synthstrip")

        # Provide a license file
        install_path = self.store.get_install_path("synthstrip")
        license_path = os.path.join(install_path, "license.txt")
        with open(license_path, "w") as f:
            f.write("fake license")

        result = self.health.check("synthstrip", run_binary=False)
        assert result["status"] == READY
        assert result["license_valid"] is True
        assert result["license_path"] is not None

    def test_broken_binary_missing(self):
        """If the binary is deleted after install, status is BROKEN."""
        manager = EngineManager(
            store=self.store, installer=StubInstaller(),
            manifests=self.manifests,
        )
        manager.install("mcflirt")

        # Delete the binary
        install_path = self.store.get_install_path("mcflirt")
        binary_path = os.path.join(install_path, "bin", "mcflirt")
        os.remove(binary_path)

        result = self.health.check("mcflirt", run_binary=False)
        assert result["status"] == BROKEN
        assert result["binary_exists"] is False
        assert any("not found" in issue.lower() for issue in result["issues"])

    def test_broken_binary_not_executable(self):
        """If the binary exists but isn't executable, status is BROKEN.

        Note: this test may be skipped when running as root, since root
        can execute any file regardless of the executable bit.
        """
        if os.geteuid() == 0:
            pytest.skip("Running as root -- can't test non-executable files")
        manager = EngineManager(
            store=self.store, installer=StubInstaller(),
            manifests=self.manifests,
        )
        manager.install("mcflirt")

        # Remove execute permission
        install_path = self.store.get_install_path("mcflirt")
        binary_path = os.path.join(install_path, "bin", "mcflirt")
        os.chmod(binary_path, 0o644)

        result = self.health.check("mcflirt", run_binary=False)
        assert result["status"] == BROKEN
        assert result["binary_exists"] is True
        assert result["binary_executable"] is False

    def test_check_all_returns_all_engines(self):
        results = self.health.check_all(run_binary=False)
        assert len(results) == 6  # all 6 manifests
        for eid in ("synthstrip", "mcflirt", "fsl_eddy", "hdbet",
                     "mri_deface", "pydeface"):
            assert eid in results
            assert results[eid]["status"] == NOT_INSTALLED

    def test_unknown_engine_returns_not_installed_with_issues(self):
        result = self.health.check("nonexistent", run_binary=False)
        assert result["status"] == NOT_INSTALLED
        assert any("unknown" in issue.lower() for issue in result["issues"])


# --------------------------------------------------------------------- #
# License file management
# --------------------------------------------------------------------- #

class TestLicenseManagement:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.store = EngineStore()
        self.manifests = load_all_manifests()
        self.manager = EngineManager(
            store=self.store, installer=StubInstaller(),
            manifests=self.manifests,
        )

    def test_store_license_copies_into_engine_dir(self, tmp_path):
        self.manager.install("synthstrip")
        src_license = tmp_path / "license.txt"
        src_license.write_text("fake license content")

        dest = store_license_file("synthstrip", str(src_license), self.store)
        assert os.path.isfile(dest)
        assert dest.endswith("license.txt")
        with open(dest) as f:
            assert f.read() == "fake license content"

    def test_store_license_raises_if_not_installed(self, tmp_path):
        src_license = tmp_path / "license.txt"
        src_license.write_text("fake")
        with pytest.raises(FileNotFoundError) as exc:
            store_license_file("synthstrip", str(src_license), self.store)
        assert "not installed" in str(exc.value).lower()

    def test_store_license_raises_if_source_missing(self, tmp_path):
        self.manager.install("synthstrip")
        with pytest.raises(FileNotFoundError) as exc:
            store_license_file("synthstrip", "/nonexistent/license.txt", self.store)
        assert "not found" in str(exc.value).lower()

    def test_get_license_path_from_engine_dir(self, tmp_path):
        self.manager.install("synthstrip")
        # Place a license file in the engine dir
        install_path = self.store.get_install_path("synthstrip")
        license_path = os.path.join(install_path, "license.txt")
        with open(license_path, "w") as f:
            f.write("license")

        found = get_license_path("synthstrip", self.store, self.manifests)
        assert found == license_path

    def test_get_license_path_from_freesurfer_home(self, tmp_path, monkeypatch):
        self.manager.install("synthstrip")
        # Don't put a license in the engine dir; put one in FREESURFER_HOME
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        fs_home = tmp_path / "freesurfer"
        fs_home.mkdir()
        (fs_home / "license.txt").write_text("fs license")
        monkeypatch.setenv("FREESURFER_HOME", str(fs_home))

        found = get_license_path("synthstrip", self.store, self.manifests)
        assert found is not None
        assert found.endswith("license.txt")

    def test_get_license_path_from_env_var(self, tmp_path, monkeypatch):
        self.manager.install("synthstrip")
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        env_license = tmp_path / "my_license.txt"
        env_license.write_text("env license")
        monkeypatch.setenv("MRI_PREPROCESSOR_SYNTHSTRIP_LICENSE", str(env_license))

        found = get_license_path("synthstrip", self.store, self.manifests)
        assert found == str(env_license)

    def test_get_license_path_returns_none_if_not_required(self):
        """For engines without license_file_required, returns None."""
        # mcflirt doesn't require a license
        found = get_license_path("mcflirt", self.store, self.manifests)
        assert found is None

    def test_get_license_path_returns_none_if_missing(self, tmp_path, monkeypatch):
        self.manager.install("synthstrip")
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP_LICENSE", raising=False)
        found = get_license_path("synthstrip", self.store, self.manifests)
        assert found is None


# --------------------------------------------------------------------- #
# EngineManager API
# --------------------------------------------------------------------- #

class TestEngineManagerHealthAPI:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.manager = EngineManager(
            store=EngineStore(), installer=StubInstaller(),
            manifests=load_all_manifests(),
        )

    def test_get_status_returns_not_installed(self):
        assert self.manager.get_status("synthstrip") == NOT_INSTALLED

    def test_get_status_returns_license_required(self):
        self.manager.install("synthstrip")
        assert self.manager.get_status("synthstrip") == LICENSE_REQUIRED

    def test_get_status_returns_ready_after_license(self, tmp_path):
        self.manager.install("synthstrip")
        # Provide license
        install_path = self.manager.store.get_install_path("synthstrip")
        with open(os.path.join(install_path, "license.txt"), "w") as f:
            f.write("license")
        assert self.manager.get_status("synthstrip") == READY

    def test_get_status_returns_ready_no_license_required(self):
        self.manager.install("mcflirt")
        assert self.manager.get_status("mcflirt") == READY

    def test_health_check_returns_full_dict(self):
        result = self.manager.health_check("synthstrip", run_binary=False)
        assert "status" in result
        assert "binary_exists" in result
        assert "license_required" in result
        assert "issues" in result

    def test_store_license_via_manager(self, tmp_path):
        self.manager.install("synthstrip")
        src = tmp_path / "license.txt"
        src.write_text("license")
        dest = self.manager.store_license("synthstrip", str(src))
        assert os.path.isfile(dest)

    def test_info_includes_health(self):
        info = self.manager.info("synthstrip")
        assert "health" in info
        assert info["health"]["status"] == NOT_INSTALLED

    def test_info_includes_health_after_install(self):
        self.manager.install("synthstrip")
        info = self.manager.info("synthstrip")
        assert info["health"]["status"] == LICENSE_REQUIRED
        assert info["health"]["license_required"] is True
        assert info["health"]["license_valid"] is False


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

runner = CliRunner()


class TestCLIHealthAndLicense:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()

    def test_cli_health_not_installed(self):
        result = runner.invoke(cli_app, [
            "engines", "health", "synthstrip", "--no-run-binary",
        ])
        assert result.exit_code == 0
        assert "Not installed" in result.stdout

    def test_cli_health_license_required(self):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip", "--stub",
        ])
        result = runner.invoke(cli_app, [
            "engines", "health", "synthstrip", "--no-run-binary",
        ])
        assert result.exit_code == 0
        assert "License required" in result.stdout
        assert "License:" in result.stdout
        assert "required but missing" in result.stdout

    def test_cli_health_ready_after_license(self, tmp_path):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip", "--stub",
        ])
        # Provide license
        license_file = tmp_path / "license.txt"
        license_file.write_text("license content")
        runner.invoke(cli_app, [
            "engines", "license", "synthstrip", str(license_file),
        ])
        result = runner.invoke(cli_app, [
            "engines", "health", "synthstrip", "--no-run-binary",
        ])
        assert result.exit_code == 0
        assert "Ready" in result.stdout
        assert "valid" in result.stdout

    def test_cli_license_command(self, tmp_path):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip", "--stub",
        ])
        license_file = tmp_path / "license.txt"
        license_file.write_text("license")
        result = runner.invoke(cli_app, [
            "engines", "license", "synthstrip", str(license_file),
        ])
        assert result.exit_code == 0
        assert "License stored" in result.stdout

    def test_cli_license_not_installed_raises(self, tmp_path):
        license_file = tmp_path / "license.txt"
        license_file.write_text("license")
        result = runner.invoke(cli_app, [
            "engines", "license", "synthstrip", str(license_file),
        ])
        assert result.exit_code != 0
        assert "not installed" in result.stdout.lower()

    def test_cli_license_source_missing_raises(self, tmp_path):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip", "--stub",
        ])
        result = runner.invoke(cli_app, [
            "engines", "license", "synthstrip", "/nonexistent/license.txt",
        ])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower()

    def test_cli_info_shows_health(self):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip", "--stub",
        ])
        result = runner.invoke(cli_app, [
            "engines", "info", "synthstrip",
        ])
        assert result.exit_code == 0
        assert "Health" in result.stdout
        assert "License required" in result.stdout

    def test_cli_info_shows_health_ready(self, tmp_path):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip", "--stub",
        ])
        license_file = tmp_path / "license.txt"
        license_file.write_text("license")
        runner.invoke(cli_app, [
            "engines", "license", "synthstrip", str(license_file),
        ])
        result = runner.invoke(cli_app, [
            "engines", "info", "synthstrip",
        ])
        assert result.exit_code == 0
        assert "Ready" in result.stdout
        assert "valid" in result.stdout

    def test_cli_health_unknown_engine(self):
        result = runner.invoke(cli_app, [
            "engines", "health", "nonexistent", "--no-run-binary",
        ])
        # The health check for an unknown engine returns NOT_INSTALLED
        # with issues -- the CLI command itself succeeds (exit 0) but
        # the output shows the unknown-engine issue.
        assert "Not installed" in result.stdout
        assert "unknown" in result.stdout.lower() or "Unknown" in result.stdout

    def test_cli_help_includes_health_and_license(self):
        result = runner.invoke(cli_app, ["engines", "--help"])
        assert result.exit_code == 0
        assert "health" in result.stdout
        assert "license" in result.stdout
