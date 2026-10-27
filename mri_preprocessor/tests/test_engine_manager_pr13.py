"""
Tests for the Engine Manager system (PR13).

Covers:
  - EngineManifest schema (fields, defaults, to_info_dict serialization)
  - Manifest loading from YAML (all 6 manifests load cleanly)
  - EngineStore filesystem ops (install marker, version dir, binary lookup)
  - EngineManager API (list known, list installed, install, uninstall, info)
  - StubInstaller (creates fake binary, marks installed)
  - Binary discovery integration (local store checked BEFORE env vars / PATH)
  - CLI: engines install / uninstall / info / list / adapters
  - End-to-end: install via CLI → binary discovery finds it → engine adapter runs
"""
import json
import os
import shutil

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.engines.installer import StubInstaller
from app.engines.manager import EngineManager, get_engine_manager
from app.engines.manifest import EngineManifest, load_all_manifests, load_manifest
from app.engines.store import EngineStore


# --------------------------------------------------------------------- #
# EngineManifest schema
# --------------------------------------------------------------------- #

def test_manifest_minimal():
    m = EngineManifest(
        id="test", name="Test", provider="X", capability="y", version="1.0",
    )
    assert m.id == "test"
    assert m.license_type == "non-commercial"
    assert m.runtime_cpu is True
    assert m.runtime_cuda is False
    assert m.binary_name == ""


def test_manifest_binary_name():
    m = EngineManifest(
        id="synthstrip", name="SynthStrip", provider="FreeSurfer",
        capability="brain_extraction", version="7.4.1",
        binary_name="mri_synthstrip",
    )
    assert m.binary_name == "mri_synthstrip"


def test_manifest_to_info_dict():
    m = EngineManifest(
        id="test", name="Test", provider="X", capability="y", version="1.0",
        binary_name="test_bin",
    )
    info = m.to_info_dict(installed=True, install_path="/path/to/test")
    assert info["installed"] is True
    assert info["install_path"] == "/path/to/test"
    assert info["license"]["type"] == "non-commercial"
    assert info["source"]["type"] == "conda"
    assert info["runtime"]["cpu"] is True


# --------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------- #

def test_all_six_manifests_load():
    manifests = load_all_manifests()
    assert len(manifests) == 6
    expected_ids = {"synthstrip", "hdbet", "mcflirt", "fsl_eddy",
                     "mri_deface", "pydeface"}
    assert set(manifests.keys()) == expected_ids


def test_synthstrip_manifest_fields():
    manifests = load_all_manifests()
    m = manifests["synthstrip"]
    assert m.name == "SynthStrip"
    assert m.provider == "FreeSurfer"
    assert m.capability == "brain_extraction"
    assert m.version == "7.4.1"
    assert m.binary_name == "mri_synthstrip"
    assert m.license_file_required is True
    assert m.source_type == "conda"
    assert m.source_package == "fsl-synthstrip"
    assert m.runtime_cuda is True


def test_eddy_manifest_requirements():
    manifests = load_all_manifests()
    m = manifests["fsl_eddy"]
    assert "dwi_4d" in m.requirements
    assert "bval" in m.requirements
    assert "bvec" in m.requirements
    assert "brain_mask" in m.requirements
    assert "acqp_file" in m.requirements
    assert "index_file" in m.requirements


def test_manifest_binary_name_matches_discovery_config():
    """Every manifest's binary_name must match its adapter's DiscoveryConfig.binary_name."""
    manifests = load_all_manifests()
    from app.engines.brain_extraction.synthstrip import SYNTHSTRIP_DISCOVERY
    from app.engines.brain_extraction.hdbet import HDBET_DISCOVERY
    from app.engines.fmri_motion.mcflirt import MCFLIRT_DISCOVERY
    from app.engines.dwi_correction.fsl_eddy import FSL_EDDY_DISCOVERY
    from app.engines.defacing.mri_deface import MRI_DEFACE_DISCOVERY
    from app.engines.defacing.pydeface import PYDEFACE_DISCOVERY

    pairs = [
        ("synthstrip", SYNTHSTRIP_DISCOVERY),
        ("hdbet", HDBET_DISCOVERY),
        ("mcflirt", MCFLIRT_DISCOVERY),
        ("fsl_eddy", FSL_EDDY_DISCOVERY),
        ("mri_deface", MRI_DEFACE_DISCOVERY),
        ("pydeface", PYDEFACE_DISCOVERY),
    ]
    for engine_id, config in pairs:
        manifest = manifests[engine_id]
        assert manifest.binary_name == config.binary_name, (
            f"Manifest {engine_id}.binary_name={manifest.binary_name!r} "
            f"doesn't match DiscoveryConfig.binary_name={config.binary_name!r}"
        )
        assert config.engine_id == engine_id, (
            f"DiscoveryConfig.engine_id={config.engine_id!r} "
            f"doesn't match manifest id={engine_id!r}"
        )


# --------------------------------------------------------------------- #
# EngineStore filesystem ops
# --------------------------------------------------------------------- #

class TestEngineStore:
    def test_store_root_created(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        store = EngineStore()
        assert os.path.isdir(store.root)

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_engines"
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(custom))
        store = EngineStore()
        assert store.root == str(custom)

    def test_is_installed_false_for_uninstalled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path))
        store = EngineStore()
        assert store.is_installed("synthstrip") is False

    def test_mark_installed_creates_structure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path))
        store = EngineStore()
        manifest = EngineManifest(
            id="test_engine", name="Test", provider="X",
            capability="y", version="1.0", binary_name="test_bin",
        )
        install_path = store.mark_installed(manifest)
        assert os.path.isdir(install_path)
        assert os.path.isdir(os.path.join(install_path, "bin"))
        assert os.path.isfile(store.installed_marker("test_engine"))
        assert os.path.isfile(store.manifest_path("test_engine"))

    def test_get_binary_path_finds_executable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path))
        store = EngineStore()
        manifest = EngineManifest(
            id="test_engine", name="Test", provider="X",
            capability="y", version="1.0", binary_name="test_bin",
        )
        store.mark_installed(manifest)
        # Create a fake binary
        bin_path = os.path.join(store.bin_dir("test_engine", "1.0"), "test_bin")
        with open(bin_path, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(bin_path, 0o755)

        found = store.get_binary_path("test_engine", "1.0", "test_bin")
        assert found == bin_path

    def test_get_binary_path_returns_none_for_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path))
        store = EngineStore()
        assert store.get_binary_path("nonexistent", "1.0", "bin") is None

    def test_remove_deletes_engine(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path))
        store = EngineStore()
        manifest = EngineManifest(
            id="to_delete", name="Test", provider="X",
            capability="y", version="1.0", binary_name="test",
        )
        store.mark_installed(manifest)
        assert store.is_installed("to_delete") is True
        assert store.remove("to_delete") is True
        assert store.is_installed("to_delete") is False
        assert store.remove("to_delete") is False  # already gone

    def test_list_installed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path))
        store = EngineStore()
        for eid in ("alpha", "beta", "gamma"):
            m = EngineManifest(
                id=eid, name=eid, provider="X",
                capability="y", version="1.0", binary_name=eid,
            )
            store.mark_installed(m)
        installed = store.list_installed()
        assert sorted(installed) == ["alpha", "beta", "gamma"]


# --------------------------------------------------------------------- #
# EngineManager
# --------------------------------------------------------------------- #

class TestEngineManager:
    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        # Use the StubInstaller so we don't need conda
        self.manager = EngineManager(
            store=EngineStore(),
            installer=StubInstaller(),
            manifests=load_all_manifests(),
        )

    def test_list_known_engines_returns_all_six(self):
        known = self.manager.list_known_engines()
        assert len(known) == 6

    def test_list_installed_empty_initially(self):
        installed = self.manager.list_installed_engines()
        assert installed == []

    def test_install_via_stub_creates_binary(self):
        install_path = self.manager.install("synthstrip")
        assert os.path.isdir(install_path)
        # The stub binary should exist at bin/mri_synthstrip
        bin_path = os.path.join(install_path, "bin", "mri_synthstrip")
        assert os.path.isfile(bin_path)
        assert os.access(bin_path, os.X_OK)

    def test_install_marks_as_installed(self):
        self.manager.install("synthstrip")
        assert self.manager.is_installed("synthstrip") is True

    def test_install_returns_existing_path_if_already_installed(self):
        path1 = self.manager.install("synthstrip")
        path2 = self.manager.install("synthstrip")  # no force
        assert path1 == path2

    def test_install_with_force_reinstalls(self):
        path1 = self.manager.install("synthstrip")
        # Corrupt the binary
        bin_path = os.path.join(path1, "bin", "mri_synthstrip")
        os.remove(bin_path)
        assert not os.path.isfile(bin_path)
        # Force reinstall
        path2 = self.manager.install("synthstrip", force=True)
        bin_path2 = os.path.join(path2, "bin", "mri_synthstrip")
        assert os.path.isfile(bin_path2)

    def test_uninstall_removes_engine(self):
        self.manager.install("synthstrip")
        assert self.manager.is_installed("synthstrip") is True
        deleted = self.manager.uninstall("synthstrip")
        assert deleted is True
        assert self.manager.is_installed("synthstrip") is False

    def test_uninstall_not_installed_returns_false(self):
        deleted = self.manager.uninstall("synthstrip")
        assert deleted is False

    def test_get_binary_path_returns_path_for_installed(self):
        self.manager.install("synthstrip")
        path = self.manager.get_binary_path("synthstrip", "mri_synthstrip")
        assert path is not None
        assert path.endswith("bin/mri_synthstrip")

    def test_get_binary_path_returns_none_for_uninstalled(self):
        path = self.manager.get_binary_path("synthstrip", "mri_synthstrip")
        assert path is None

    def test_info_returns_full_dict(self):
        info = self.manager.info("synthstrip")
        assert info["id"] == "synthstrip"
        assert info["name"] == "SynthStrip"
        assert info["provider"] == "FreeSurfer"
        assert info["installed"] is False
        assert "installer" in info
        assert info["installer"]["name"] == "stub"

    def test_info_for_installed_engine(self):
        self.manager.install("synthstrip")
        info = self.manager.info("synthstrip")
        assert info["installed"] is True
        assert info["install_path"] is not None

    def test_info_unknown_engine_raises(self):
        with pytest.raises(KeyError):
            self.manager.info("nonexistent")

    def test_install_unknown_engine_raises(self):
        with pytest.raises(KeyError):
            self.manager.install("nonexistent")


# --------------------------------------------------------------------- #
# Binary discovery integration
# --------------------------------------------------------------------- #

class TestBinaryDiscoveryIntegration:
    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path, monkeypatch):
        """Isolates the engine store + resets the singleton so env var
        changes take effect."""
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        # Use a StubInstaller by replacing the singleton AFTER the env var is set
        import app.engines.manager as mgr_module
        _reset_engine_manager()
        # The default singleton uses CondaInstaller. For tests, we need
        # StubInstaller. We replace the singleton's installer.
        mgr = mgr_module.get_engine_manager()
        from app.engines.installer import StubInstaller
        mgr.installer = StubInstaller()
        yield
        _reset_engine_manager()

    def test_discovery_finds_binary_in_local_store(self, monkeypatch):
        """After `engines install synthstrip --stub`, find_binary() finds
        the binary in the local store WITHOUT checking env vars or PATH."""
        # Install via the manager
        from app.engines.manager import get_engine_manager
        get_engine_manager().install("synthstrip")

        # Clear all env vars so only the store can provide the binary
        monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
        monkeypatch.delenv("FREESURFER_HOME", raising=False)

        from app.engines.binary_discovery import find_binary
        from app.engines.brain_extraction.synthstrip import SYNTHSTRIP_DISCOVERY

        path, reason = find_binary(SYNTHSTRIP_DISCOVERY)
        assert path is not None
        assert path.endswith("bin/mri_synthstrip")
        assert reason == ""

    def test_discovery_returns_none_when_not_installed(self, monkeypatch):
        """Without an install, discovery falls through to env vars / PATH /
        tool-suite home (all empty in test env) and returns None."""
        monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
        monkeypatch.delenv("FREESURFER_HOME", raising=False)

        from app.engines.binary_discovery import find_binary
        from app.engines.brain_extraction.synthstrip import SYNTHSTRIP_DISCOVERY

        path, reason = find_binary(SYNTHSTRIP_DISCOVERY)
        assert path is None
        # The reason should mention the local store was checked
        assert "local engine store" in reason.lower()
        # And suggest installing via the CLI
        assert "engines install" in reason.lower()

    def test_discovery_local_store_takes_precedence_over_env(self, monkeypatch, tmp_path):
        """If the engine is installed in the store AND an env var is set,
        the store wins (managed install takes priority)."""
        # Install via the manager
        from app.engines.manager import get_engine_manager
        get_engine_manager().install("synthstrip")

        # Also set an env var pointing at a DIFFERENT binary
        fake_env_bin = tmp_path / "env_synthstrip"
        fake_env_bin.write_text("#!/bin/bash\nexit 0\n")
        fake_env_bin.chmod(0o755)
        monkeypatch.setenv("MRI_PREPROCESSOR_SYNTHSTRIP", str(fake_env_bin))

        from app.engines.binary_discovery import find_binary
        from app.engines.brain_extraction.synthstrip import SYNTHSTRIP_DISCOVERY

        path, _ = find_binary(SYNTHSTRIP_DISCOVERY)
        # Should be the store path, not the env var path
        assert "synthstrip/7.4.1/bin/mri_synthstrip" in path
        assert str(fake_env_bin) not in path


# --------------------------------------------------------------------- #
# CLI: engines install / uninstall / info / list / adapters
# --------------------------------------------------------------------- #

runner = CliRunner()


@pytest.fixture(autouse=True)
def _cli_isolated_store(tmp_path, monkeypatch):
    """Isolates the engine store so CLI tests don't pollute the real
    ~/.mri-preprocessor/ or project-level engines/ directory."""
    monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "engines_store"))
    return tmp_path / "engines_store"


def test_cli_engines_list_shows_all_six():
    result = runner.invoke(cli_app, ["engines", "list"])
    assert result.exit_code == 0
    for eid in ("synthstrip", "hdbet", "mcflirt", "fsl_eddy", "mri_deface", "pydeface"):
        assert eid in result.stdout


def test_cli_engines_list_installed_only():
    result = runner.invoke(cli_app, ["engines", "list", "--installed"])
    assert result.exit_code == 0
    # Nothing installed yet
    assert "No installed engines" in result.stdout


def test_cli_engines_info_synthstrip():
    result = runner.invoke(cli_app, ["engines", "info", "synthstrip"])
    assert result.exit_code == 0
    assert "SynthStrip" in result.stdout
    assert "FreeSurfer" in result.stdout
    assert "Not installed" in result.stdout
    # The default installer is conda, which may or may not be available
    # in the test env. Either "Install with:" or "Installer not available"
    # should appear.
    assert "Install with:" in result.stdout or "Installer not available" in result.stdout


def test_cli_engines_info_unknown_engine():
    result = runner.invoke(cli_app, ["engines", "info", "nonexistent"])
    assert result.exit_code != 0


def test_cli_engines_install_with_stub():
    result = runner.invoke(cli_app, [
        "engines", "install", "synthstrip", "--stub",
    ])
    assert result.exit_code == 0
    assert "Installed" in result.stdout

    # Verify it shows as installed in `engines list`
    result = runner.invoke(cli_app, ["engines", "list"])
    assert "✓ Installed" in result.stdout


def test_cli_engines_install_then_info_shows_installed():
    runner.invoke(cli_app, ["engines", "install", "synthstrip", "--stub"])
    result = runner.invoke(cli_app, ["engines", "info", "synthstrip"])
    assert result.exit_code == 0
    assert "✓ Installed" in result.stdout


def test_cli_engines_uninstall():
    runner.invoke(cli_app, ["engines", "install", "synthstrip", "--stub"])
    result = runner.invoke(cli_app, ["engines", "uninstall", "synthstrip"])
    assert result.exit_code == 0
    assert "Removed" in result.stdout

    # Verify it's gone
    result = runner.invoke(cli_app, ["engines", "info", "synthstrip"])
    assert "Not installed" in result.stdout


def test_cli_engines_uninstall_not_installed():
    result = runner.invoke(cli_app, ["engines", "uninstall", "synthstrip"])
    assert result.exit_code != 0
    assert "not installed" in result.stdout.lower()


def test_cli_engines_adapters_brain_extraction():
    """The old `engines <step>` is now `engines adapters <step>`."""
    result = runner.invoke(cli_app, [
        "engines", "adapters", "brain_extraction",
    ])
    assert result.exit_code == 0
    assert "otsu" in result.stdout
    assert "synthstrip" in result.stdout
    assert "hdbet" in result.stdout


def test_cli_engines_adapters_unknown_step():
    result = runner.invoke(cli_app, [
        "engines", "adapters", "nonexistent_step",
    ])
    assert result.exit_code != 0


# --------------------------------------------------------------------- #
# End-to-end: install via CLI → discovery finds it → adapter runs
# --------------------------------------------------------------------- #

def test_end_to_end_install_then_run(tmp_path, monkeypatch):
    """The full PR13 flow: install synthstrip via CLI (stub), then run
    brain_extraction with engine=synthstrip. The binary discovery chain
    finds the stub binary in the local store and the step executes."""
    from app.engines.manager import _reset_engine_manager
    from app.engines.installer import StubInstaller

    # Set up isolated store + reset singleton
    _reset_engine_manager()
    monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
    _reset_engine_manager()
    import app.engines.manager as mgr_module
    mgr = mgr_module.get_engine_manager()
    mgr.installer = StubInstaller()

    # Clear env vars so only the store can provide the binary
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    # Install via CLI (using --stub so we don't need conda)
    result = runner.invoke(cli_app, [
        "engines", "install", "synthstrip", "--stub",
    ])
    assert result.exit_code == 0

    # Verify binary discovery now finds it
    from app.engines.binary_discovery import find_binary
    from app.engines.brain_extraction.synthstrip import SYNTHSTRIP_DISCOVERY
    path, reason = find_binary(SYNTHSTRIP_DISCOVERY)
    assert path is not None
    assert path.endswith("bin/mri_synthstrip")

    # Now run brain_extraction with engine=synthstrip via the CLI.
    input_dir = tmp_path / "input" / "sub-001"
    input_dir.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(input_dir / "sub-001_T1.nii.gz"),
    )
    output_dir = str(tmp_path / "output")

    result = runner.invoke(cli_app, [
        "run", str(input_dir.parent), output_dir,
        "--steps", "brain_extraction",
        "--modalities", "T1",
        "--param", "brain_extraction.engine=synthstrip",
    ])
    # The job should fail because the stub binary doesn't produce output,
    # but the failure should be at the OUTPUT VALIDATION stage, not the
    # binary discovery stage.
    assert result.exit_code == 0  # the CLI itself exits 0 even on job failure
    assert "1 failed" in result.stdout  # the acquisition failed
    # The error should NOT mention "not available" (binary WAS found)
    assert "not available" not in result.stdout.lower()
