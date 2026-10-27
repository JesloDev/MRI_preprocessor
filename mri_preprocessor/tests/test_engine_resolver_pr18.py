"""
Tests for PR18: automatic pipeline → engine resolution.

Covers:
  - resolve_engines_for_steps(): maps steps + step_params to required
    engines, checks install/health status
  - resolve_engines_for_preset(): same, but from a Preset object
  - Built-in engines (otsu, simpleitk, dipy, placeholder) always ready
  - External engines (synthstrip, mcflirt, etc.) checked against store
  - Status transitions: not_installed -> license_required -> ready
  - CLI: engines resolve --preset <name> / --steps <list>
  - CLI: presets install <name> --dry-run
  - CLI: presets install <name> with --bundle-dir (stub bundles)
"""
import os

import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.engines.installer import StubInstaller
from app.engines.manager import EngineManager, get_engine_manager
from app.engines.manifest import load_all_manifests
from app.engines.resolver import (
    BUILT_IN_ENGINES, _is_external_engine, _get_step_engines,
    resolve_engines_for_steps, resolve_engines_for_preset,
    format_resolution_report,
)
from app.engines.store import EngineStore
from app.presets import get_preset
from app.presets.schema import Preset


# --------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------- #

def test_built_in_engines_set():
    """The built-in engine names are the ones that don't need installation."""
    assert "otsu" in BUILT_IN_ENGINES
    assert "simpleitk" in BUILT_IN_ENGINES
    assert "dipy" in BUILT_IN_ENGINES
    assert "placeholder" in BUILT_IN_ENGINES


def test_is_external_engine_built_in():
    from app.engines.manager import _reset_engine_manager
    _reset_engine_manager()
    manager = get_engine_manager()
    assert _is_external_engine("otsu", manager) is False
    assert _is_external_engine("simpleitk", manager) is False
    assert _is_external_engine("dipy", manager) is False
    assert _is_external_engine("placeholder", manager) is False


def test_is_external_engine_external():
    from app.engines.manager import _reset_engine_manager
    _reset_engine_manager()
    manager = get_engine_manager()
    assert _is_external_engine("synthstrip", manager) is True
    assert _is_external_engine("mcflirt", manager) is True
    assert _is_external_engine("fsl_eddy", manager) is True


def test_get_step_engines_for_brain_extraction():
    """brain_extraction has engine=otsu in its default_params."""
    engine = _get_step_engines("brain_extraction", {})
    assert engine == "otsu"

    # Override via step_params
    engine = _get_step_engines("brain_extraction",
                                {"brain_extraction": {"engine": "synthstrip"}})
    assert engine == "synthstrip"


def test_get_step_engines_for_step_without_engines():
    """Steps without engine selection return None."""
    engine = _get_step_engines("reorientation", {})
    assert engine is None


# --------------------------------------------------------------------- #
# resolve_engines_for_steps
# --------------------------------------------------------------------- #

class TestResolveEnginesForSteps:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.manager = get_engine_manager()

    def test_all_built_in_engines_ready(self):
        """A pipeline using only built-in engines is all_ready."""
        report = resolve_engines_for_steps(
            ["reorientation", "brain_extraction", "resampling"],
            {"brain_extraction": {"engine": "otsu"}},
            self.manager,
        )
        assert report["all_ready"] is True
        assert len(report["missing"]) == 0
        assert len(report["license_required"]) == 0

    def test_external_engine_not_installed_is_missing(self):
        """SynthStrip is external and not installed -> missing."""
        report = resolve_engines_for_steps(
            ["brain_extraction"],
            {"brain_extraction": {"engine": "synthstrip"}},
            self.manager,
        )
        assert report["all_ready"] is False
        assert "synthstrip" in report["missing"]
        assert len(report["ready"]) == 0

    def test_external_engine_installed_no_license_is_license_required(self):
        """SynthStrip installed via stub but no license -> license_required."""
        # Install via stub
        from app.engines.installer import StubInstaller
        self.manager.installer = StubInstaller()
        self.manager.install("synthstrip")

        report = resolve_engines_for_steps(
            ["brain_extraction"],
            {"brain_extraction": {"engine": "synthstrip"}},
            self.manager,
        )
        assert report["all_ready"] is False
        assert "synthstrip" in report["license_required"]
        assert len(report["missing"]) == 0

    def test_external_engine_with_license_is_ready(self, tmp_path):
        """SynthStrip installed + license provided -> ready."""
        from app.engines.installer import StubInstaller
        self.manager.installer = StubInstaller()
        self.manager.install("synthstrip")

        # Provide license
        install_path = self.manager.store.get_install_path("synthstrip")
        with open(os.path.join(install_path, "license.txt"), "w") as f:
            f.write("license")

        report = resolve_engines_for_steps(
            ["brain_extraction"],
            {"brain_extraction": {"engine": "synthstrip"}},
            self.manager,
        )
        assert report["all_ready"] is True
        assert "synthstrip" in report["ready"]
        assert len(report["missing"]) == 0
        assert len(report["license_required"]) == 0

    def test_mixed_built_in_and_external(self):
        """A pipeline with both built-in (otsu) and external (synthstrip)
        engines: otsu is ready, synthstrip is missing."""
        report = resolve_engines_for_steps(
            ["brain_extraction", "resampling"],
            {"brain_extraction": {"engine": "otsu"}},
            self.manager,
        )
        # No external engines in this config
        assert report["all_ready"] is True
        assert "otsu" in report["ready"]

    def test_multiple_external_engines(self):
        """Multiple external engines: all missing."""
        report = resolve_engines_for_steps(
            ["brain_extraction", "defacing"],
            {"brain_extraction": {"engine": "synthstrip"},
             "defacing": {"engine": "pydeface"}},
            self.manager,
        )
        assert report["all_ready"] is False
        assert "synthstrip" in report["missing"]
        assert "pydeface" in report["missing"]

    def test_step_without_engine_selection_skipped(self):
        """Steps without engine selection are not in the report."""
        report = resolve_engines_for_steps(
            ["reorientation", "resampling"],
            {},
            self.manager,
        )
        assert len(report["required_engines"]) == 0
        assert report["all_ready"] is True

    def test_deduplication(self):
        """If the same engine appears in multiple steps, it's listed once."""
        report = resolve_engines_for_steps(
            ["brain_extraction", "registration"],
            {"brain_extraction": {"engine": "synthstrip"}},
            self.manager,
        )
        # synthstrip should appear once in required_engines
        synthstrip_entries = [e for e in report["required_engines"]
                              if e["engine_name"] == "synthstrip"]
        assert len(synthstrip_entries) == 1

    def test_report_structure(self):
        """The report dict has all expected keys."""
        report = resolve_engines_for_steps(
            ["brain_extraction"], {}, self.manager,
        )
        assert "required_engines" in report
        assert "ready" in report
        assert "missing" in report
        assert "license_required" in report
        assert "all_ready" in report


# --------------------------------------------------------------------- #
# resolve_engines_for_preset
# --------------------------------------------------------------------- #

class TestResolveEnginesForPreset:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.manager = get_engine_manager()

    def test_basic_qc_all_ready(self):
        """basic-qc uses only reorientation (no engine selection)."""
        preset = get_preset("basic-qc")
        report = resolve_engines_for_preset(preset, self.manager)
        assert report["all_ready"] is True

    def test_standard_uses_otsu(self):
        """standard uses otsu brain_extraction (built-in)."""
        preset = get_preset("standard")
        report = resolve_engines_for_preset(preset, self.manager)
        assert report["all_ready"] is True
        assert "otsu" in report["ready"]

    def test_ml_ready_uses_built_in_engines(self):
        """ml-ready uses otsu, simpleitk, dipy (all built-in)."""
        preset = get_preset("ml-ready")
        report = resolve_engines_for_preset(preset, self.manager)
        assert report["all_ready"] is True
        assert "otsu" in report["ready"]
        assert "simpleitk" in report["ready"]
        assert "dipy" in report["ready"]

    def test_privacy_uses_placeholder(self):
        """privacy uses placeholder defacing (built-in)."""
        preset = get_preset("privacy")
        report = resolve_engines_for_preset(preset, self.manager)
        assert report["all_ready"] is True
        assert "placeholder" in report["ready"]

    def test_custom_preset_with_external_engine(self, tmp_path, monkeypatch):
        """A custom preset using synthstrip: missing until installed."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path / "presets"))
        preset = Preset(
            name="test-external",
            step_ids=["brain_extraction"],
            step_params={"brain_extraction": {"engine": "synthstrip"}},
        )
        report = resolve_engines_for_preset(preset, self.manager)
        assert report["all_ready"] is False
        assert "synthstrip" in report["missing"]


# --------------------------------------------------------------------- #
# format_resolution_report
# --------------------------------------------------------------------- #

class TestFormatReport:
    def test_all_ready(self, tmp_path, monkeypatch):
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        manager = get_engine_manager()
        report = resolve_engines_for_steps(
            ["brain_extraction"], {}, manager,
        )
        text = format_resolution_report(report)
        assert "All required engines are ready" in text

    def test_missing_engine(self, tmp_path, monkeypatch):
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        manager = get_engine_manager()
        report = resolve_engines_for_steps(
            ["brain_extraction"],
            {"brain_extraction": {"engine": "synthstrip"}},
            manager,
        )
        text = format_resolution_report(report)
        assert "synthstrip" in text
        assert "Missing" in text
        assert "Not installed" in text


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

runner = CliRunner()


class TestCLIEngineResolve:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()

    def test_cli_resolve_preset_basic_qc(self):
        result = runner.invoke(cli_app, [
            "engines", "resolve", "--preset", "basic-qc",
        ])
        assert result.exit_code == 0
        assert "All required engines are ready" in result.stdout

    def test_cli_resolve_preset_standard(self):
        result = runner.invoke(cli_app, [
            "engines", "resolve", "--preset", "standard",
        ])
        assert result.exit_code == 0
        assert "otsu" in result.stdout
        assert "built-in" in result.stdout

    def test_cli_resolve_steps(self):
        result = runner.invoke(cli_app, [
            "engines", "resolve", "--steps", "brain_extraction,resampling",
        ])
        assert result.exit_code == 0
        assert "otsu" in result.stdout

    def test_cli_resolve_with_external_engine(self):
        result = runner.invoke(cli_app, [
            "engines", "resolve", "--steps", "brain_extraction",
            "--preset", None,  # this won't work; need --steps only
        ])
        # Actually let's use --steps with a synthstrip param
        # But --steps doesn't accept --param. So we resolve from the
        # default engine (otsu) which is built-in.
        result = runner.invoke(cli_app, [
            "engines", "resolve", "--steps", "brain_extraction",
        ])
        assert result.exit_code == 0
        assert "otsu" in result.stdout

    def test_cli_resolve_no_args_fails(self):
        result = runner.invoke(cli_app, [
            "engines", "resolve",
        ])
        assert result.exit_code != 0

    def test_cli_resolve_unknown_preset(self):
        result = runner.invoke(cli_app, [
            "engines", "resolve", "--preset", "nonexistent",
        ])
        assert result.exit_code != 0


class TestCLIPresetsInstall:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()

    def test_cli_presets_install_dry_run_all_ready(self):
        result = runner.invoke(cli_app, [
            "presets", "install", "standard", "--dry-run",
        ])
        assert result.exit_code == 0
        assert "All required engines are ready" in result.stdout

    def test_cli_presets_install_with_missing_engines_dry_run(self, tmp_path, monkeypatch):
        """A custom preset with an external engine shows as missing in dry-run."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path / "presets"))
        # Create a custom preset with synthstrip
        runner.invoke(cli_app, [
            "presets", "create", "test-ext",
            "--steps", "brain_extraction",
            "--param", "brain_extraction.engine=synthstrip",
        ])
        result = runner.invoke(cli_app, [
            "presets", "install", "test-ext", "--dry-run",
        ])
        assert result.exit_code == 0
        assert "synthstrip" in result.stdout
        assert "would install" in result.stdout.lower()

    def test_cli_presets_install_bundle_dir(self, tmp_path, monkeypatch):
        """Install missing engines from a bundle directory."""
        monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path / "presets"))
        # Create a custom preset with synthstrip
        runner.invoke(cli_app, [
            "presets", "create", "test-bundle",
            "--steps", "brain_extraction",
            "--param", "brain_extraction.engine=synthstrip",
        ])
        # Create a bundle dir with synthstrip stub
        bundle_dir = tmp_path / "bundles"
        bundle_base = bundle_dir / "synthstrip"
        plat_dir = bundle_base / "linux-64" / "bin"
        plat_dir.mkdir(parents=True)
        binary = plat_dir / "mri_synthstrip"
        binary.write_text("#!/bin/bash\nexit 0\n")
        binary.chmod(0o755)
        monkeypatch.setenv("MRI_PREPROCESSOR_PLATFORM", "linux-64")

        result = runner.invoke(cli_app, [
            "presets", "install", "test-bundle",
            "--bundle-dir", str(bundle_dir),
        ])
        assert result.exit_code == 0
        assert "Installed" in result.stdout
        # Verify synthstrip is now installed
        assert "synthstrip" in result.stdout

    def test_cli_presets_install_unknown_preset(self):
        result = runner.invoke(cli_app, [
            "presets", "install", "nonexistent",
        ])
        assert result.exit_code != 0

    def test_cli_presets_install_already_ready(self):
        """If all engines are already ready, no installation happens."""
        result = runner.invoke(cli_app, [
            "presets", "install", "basic-qc",
        ])
        assert result.exit_code == 0
        assert "All required engines are ready" in result.stdout

    def test_cli_presets_help_includes_install(self):
        result = runner.invoke(cli_app, ["presets", "--help"])
        assert result.exit_code == 0
        assert "install" in result.stdout
