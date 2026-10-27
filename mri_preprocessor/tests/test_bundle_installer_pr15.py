"""
Tests for the BundleInstaller (PR15).

The BundleInstaller is the offline-distribution strategy: it registers
a pre-bundled engine directory in the store WITHOUT any download. No
internet, no conda, no micromamba needed.

Covers:
  - BundleInstaller: copy mode (copies bundle dir into store)
  - BundleInstaller: symlink mode (zero disk overhead, for read-only
    application directories)
  - Bundle validation (missing binary, non-executable, missing dir)
  - EngineManager.install_bundle() API
  - CLI: engines install --bundle <path> [--symlink]
  - End-to-end: bundle install → binary discovery → adapter runs
  - Uninstall removes bundle-registered engines
  - Force reinstall replaces previous bundle install
"""
import os
import stat

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.engines.installer import BundleInstaller
from app.engines.manager import EngineManager, get_engine_manager
from app.engines.manifest import EngineManifest, load_all_manifests
from app.engines.store import EngineStore


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _make_test_bundle(bundle_path, binary_name="mri_synthstrip",
                       executable_content="#!/bin/bash\nexit 0\n"):
    """Creates a minimal bundle directory with bin/<binary_name>."""
    bin_dir = bundle_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary_path = bin_dir / binary_name
    binary_path.write_text(executable_content)
    binary_path.chmod(0o755)
    return str(bundle_path)


def _make_realistic_synthstrip_bundle(bundle_path):
    """Creates a bundle that mimics a real SynthStrip installation:
    bin/mri_synthstrip that actually does something (copies input to
    output)."""
    bin_dir = bundle_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "mri_synthstrip"
    binary.write_text("""#!/bin/bash
# Fake mri_synthstrip that mimics the real CLI:
# mri_synthstrip --i <input> --o <brain> --mask <mask>
INPUT=""
BRAIN=""
MASK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --i)    INPUT="$2"; shift 2 ;;
    --o)    BRAIN="$2"; shift 2 ;;
    --mask) MASK="$2"; shift 2 ;;
    *)      shift ;;
  esac
done
[ -n "$BRAIN" ] && cp "$INPUT" "$BRAIN"
[ -n "$MASK" ] && python3 -c "
import nibabel as nib, numpy as np
img = nib.load('$INPUT')
nib.save(nib.Nifti1Image(np.zeros(img.shape[:3], dtype='uint8'), img.affine), '$MASK')
"
exit 0
""")
    binary.chmod(0o755)
    return str(bundle_path)


# --------------------------------------------------------------------- #
# BundleInstaller unit tests
# --------------------------------------------------------------------- #

class TestBundleInstallerUnit:
    def test_is_available_always_true(self):
        installer = BundleInstaller()
        assert installer.is_available() is True
        assert installer.availability_reason() == ""

    def test_name(self):
        assert BundleInstaller().name == "bundle"

    def test_resolve_bundle_path_from_constructor(self, tmp_path):
        bundle = tmp_path / "my_bundle"
        bundle.mkdir()
        (bundle / "bin").mkdir()
        (bundle / "bin" / "mri_synthstrip").write_text("#!/bin/bash\nexit 0\n")
        (bundle / "bin" / "mri_synthstrip").chmod(0o755)

        manifest = EngineManifest(
            id="synthstrip", name="SynthStrip", provider="FreeSurfer",
            capability="brain_extraction", version="7.4.1",
            binary_name="mri_synthstrip",
        )
        installer = BundleInstaller(bundle_path=str(bundle))
        path = installer._resolve_bundle_path(manifest)
        assert path == str(bundle)

    def test_resolve_bundle_path_missing_raises(self):
        manifest = EngineManifest(
            id="synthstrip", name="SynthStrip", provider="FreeSurfer",
            capability="brain_extraction", version="7.4.1",
            binary_name="mri_synthstrip",
        )
        installer = BundleInstaller()  # no bundle_path
        with pytest.raises(FileNotFoundError) as exc_info:
            installer._resolve_bundle_path(manifest)
        assert "bundle path" in str(exc_info.value).lower()

    def test_resolve_bundle_path_nonexistent_dir_raises(self, tmp_path):
        manifest = EngineManifest(
            id="synthstrip", name="SynthStrip", provider="FreeSurfer",
            capability="brain_extraction", version="7.4.1",
            binary_name="mri_synthstrip",
        )
        installer = BundleInstaller(bundle_path=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError) as exc_info:
            installer._resolve_bundle_path(manifest)
        assert "not found" in str(exc_info.value).lower()

    def test_validate_bundle_missing_binary_raises(self, tmp_path):
        bundle = tmp_path / "bad_bundle"
        (bundle / "bin").mkdir(parents=True)
        # No binary inside bin/

        manifest = EngineManifest(
            id="synthstrip", name="SynthStrip", provider="FreeSurfer",
            capability="brain_extraction", version="7.4.1",
            binary_name="mri_synthstrip",
        )
        installer = BundleInstaller(bundle_path=str(bundle))
        with pytest.raises(FileNotFoundError) as exc_info:
            installer._validate_bundle(str(bundle), manifest)
        assert "bin/mri_synthstrip" in str(exc_info.value)

    def test_validate_bundle_non_executable_raises(self, tmp_path):
        bundle = tmp_path / "bad_bundle"
        bin_dir = bundle / "bin"
        bin_dir.mkdir(parents=True)
        binary = bin_dir / "mri_synthstrip"
        binary.write_text("#!/bin/bash\nexit 0\n")
        binary.chmod(0o644)  # not executable

        manifest = EngineManifest(
            id="synthstrip", name="SynthStrip", provider="FreeSurfer",
            capability="brain_extraction", version="7.4.1",
            binary_name="mri_synthstrip",
        )
        installer = BundleInstaller(bundle_path=str(bundle))
        with pytest.raises(FileNotFoundError) as exc_info:
            installer._validate_bundle(str(bundle), manifest)
        assert "not executable" in str(exc_info.value).lower()


# --------------------------------------------------------------------- #
# BundleInstaller: copy mode
# --------------------------------------------------------------------- #

class TestBundleInstallerCopyMode:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.store = EngineStore()
        self.manifests = load_all_manifests()
        self.manifest = self.manifests["synthstrip"]
        self.bundle_path = _make_test_bundle(tmp_path / "bundle")

    def test_copy_creates_install_dir(self):
        installer = BundleInstaller(bundle_path=self.bundle_path, symlink=False)
        install_path = installer.install(self.manifest, self.store)
        assert os.path.isdir(install_path)
        assert os.path.isdir(os.path.join(install_path, "bin"))

    def test_copy_binary_exists_and_executable(self):
        installer = BundleInstaller(bundle_path=self.bundle_path)
        install_path = installer.install(self.manifest, self.store)
        binary_path = os.path.join(install_path, "bin", "mri_synthstrip")
        assert os.path.isfile(binary_path)
        assert os.access(binary_path, os.X_OK)

    def test_copy_marks_installed(self):
        installer = BundleInstaller(bundle_path=self.bundle_path)
        installer.install(self.manifest, self.store)
        assert self.store.is_installed("synthstrip") is True

    def test_copy_creates_manifest_copy(self):
        installer = BundleInstaller(bundle_path=self.bundle_path)
        installer.install(self.manifest, self.store)
        assert os.path.isfile(self.store.manifest_path("synthstrip"))

    def test_copy_returns_version_pinned_path(self):
        installer = BundleInstaller(bundle_path=self.bundle_path)
        install_path = installer.install(self.manifest, self.store)
        assert install_path.endswith("synthstrip/7.4.1")

    def test_copy_independent_of_original(self):
        """After copy install, deleting the original bundle doesn't
        affect the store copy."""
        import shutil
        installer = BundleInstaller(bundle_path=self.bundle_path)
        install_path = installer.install(self.manifest, self.store)
        binary_path = os.path.join(install_path, "bin", "mri_synthstrip")
        # Delete the original bundle
        shutil.rmtree(self.bundle_path)
        # Store copy should still exist and be executable
        assert os.path.isfile(binary_path)
        assert os.access(binary_path, os.X_OK)

    def test_copy_overwrites_existing_install(self, tmp_path):
        """If the install dir already exists, BundleInstaller cleans it
        up and re-copies."""
        installer = BundleInstaller(bundle_path=self.bundle_path)
        # Install once
        installer.install(self.manifest, self.store)
        # Install again (simulating force-reinstall via direct installer call)
        # The manager normally handles force; here we just call install again
        installer2 = BundleInstaller(bundle_path=self.bundle_path)
        install_path = installer2.install(self.manifest, self.store)
        assert os.path.isfile(os.path.join(install_path, "bin", "mri_synthstrip"))


# --------------------------------------------------------------------- #
# BundleInstaller: symlink mode
# --------------------------------------------------------------------- #

class TestBundleInstallerSymlinkMode:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.store = EngineStore()
        self.manifests = load_all_manifests()
        self.manifest = self.manifests["synthstrip"]
        self.bundle_path = _make_test_bundle(tmp_path / "bundle")

    def test_symlink_creates_symlink(self):
        installer = BundleInstaller(bundle_path=self.bundle_path, symlink=True)
        install_path = installer.install(self.manifest, self.store)
        assert os.path.islink(install_path)
        assert os.readlink(install_path) == self.bundle_path

    def test_symlink_binary_resolvable(self):
        """The binary inside the symlinked dir is findable via normal
        path resolution (os.path follows symlinks)."""
        installer = BundleInstaller(bundle_path=self.bundle_path, symlink=True)
        install_path = installer.install(self.manifest, self.store)
        binary_path = os.path.join(install_path, "bin", "mri_synthstrip")
        assert os.path.isfile(binary_path)
        assert os.access(binary_path, os.X_OK)

    def test_symlink_marks_installed(self):
        installer = BundleInstaller(bundle_path=self.bundle_path, symlink=True)
        installer.install(self.manifest, self.store)
        assert self.store.is_installed("synthstrip") is True

    def test_symlink_zero_disk_overhead(self):
        """Symlink mode creates no copy -- the store entry is a symlink,
        not a directory tree."""
        installer = BundleInstaller(bundle_path=self.bundle_path, symlink=True)
        install_path = installer.install(self.manifest, self.store)
        # The symlink itself proves no copy was made -- the store entry
        # is just a link to the bundle, not a replicated directory tree.
        assert os.path.islink(install_path)
        # The store directory should NOT contain a copy of the bundle's
        # bin/ tree -- the binary lives only in the bundle, reached via
        # the symlink.
        store_bin = os.path.join(install_path, "bin")
        assert os.path.islink(store_bin) or os.path.isdir(store_bin)
        # The binary is accessible via the symlink (proving the link works)
        binary_in_store = os.path.join(store_bin, "mri_synthstrip")
        assert os.path.isfile(binary_in_store)
        # But it's the SAME inode as the original (no copy)
        original_inode = os.stat(
            os.path.join(self.bundle_path, "bin", "mri_synthstrip")
        ).st_ino
        store_inode = os.stat(binary_in_store).st_ino
        assert original_inode == store_inode


# --------------------------------------------------------------------- #
# EngineManager.install_bundle() API
# --------------------------------------------------------------------- #

class TestEngineManagerInstallBundle:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.manager = get_engine_manager()
        self.bundle_path = _make_test_bundle(tmp_path / "bundle")

    def test_install_bundle_returns_path(self):
        path = self.manager.install_bundle("synthstrip", self.bundle_path)
        assert os.path.isdir(path)
        assert path.endswith("synthstrip/7.4.1")

    def test_install_bundle_marks_installed(self):
        self.manager.install_bundle("synthstrip", self.bundle_path)
        assert self.manager.is_installed("synthstrip") is True

    def test_install_bundle_binary_findable(self):
        self.manager.install_bundle("synthstrip", self.bundle_path)
        binary = self.manager.get_binary_path("synthstrip", "mri_synthstrip")
        assert binary is not None
        assert binary.endswith("bin/mri_synthstrip")

    def test_install_bundle_already_installed_returns_existing(self):
        path1 = self.manager.install_bundle("synthstrip", self.bundle_path)
        path2 = self.manager.install_bundle("synthstrip", self.bundle_path)
        assert path1 == path2  # returns existing, no reinstall

    def test_install_bundle_force_reinstalls(self, tmp_path):
        path1 = self.manager.install_bundle("synthstrip", self.bundle_path)
        # Create a different bundle
        new_bundle = _make_test_bundle(tmp_path / "bundle2")
        path2 = self.manager.install_bundle(
            "synthstrip", new_bundle, force=True,
        )
        # Should be the same path (version dir) but now points at bundle2's binary
        assert path1 == path2
        # The binary should be from the new bundle
        assert os.path.isfile(os.path.join(path2, "bin", "mri_synthstrip"))

    def test_install_bundle_unknown_engine_raises(self):
        with pytest.raises(KeyError):
            self.manager.install_bundle("nonexistent", self.bundle_path)

    def test_install_bundle_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            self.manager.install_bundle(
                "synthstrip", "/nonexistent/bundle/path",
            )


# --------------------------------------------------------------------- #
# CLI: engines install --bundle
# --------------------------------------------------------------------- #

runner = CliRunner()


class TestCLIBundleInstall:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
        from app.engines.manager import _reset_engine_manager
        _reset_engine_manager()
        self.bundle_path = _make_test_bundle(tmp_path / "bundle")

    def test_cli_bundle_install_copy(self):
        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", self.bundle_path,
        ])
        assert result.exit_code == 0
        assert "Installed" in result.stdout
        assert "bundle (copy)" in result.stdout
        assert "Offline" in result.stdout

    def test_cli_bundle_install_symlink(self):
        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", self.bundle_path,
            "--symlink",
        ])
        assert result.exit_code == 0
        assert "bundle (symlink)" in result.stdout

    def test_cli_bundle_install_then_list_shows_installed(self):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", self.bundle_path,
        ])
        result = runner.invoke(cli_app, ["engines", "list"])
        assert "✓ Installed" in result.stdout

    def test_cli_bundle_install_then_info_shows_installed(self):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", self.bundle_path,
        ])
        result = runner.invoke(cli_app, ["engines", "info", "synthstrip"])
        assert "✓ Installed" in result.stdout

    def test_cli_bundle_install_then_uninstall(self):
        runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", self.bundle_path,
        ])
        result = runner.invoke(cli_app, ["engines", "uninstall", "synthstrip"])
        assert result.exit_code == 0
        assert "Removed" in result.stdout

    def test_cli_stub_and_bundle_mutually_exclusive(self):
        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--stub", "--bundle", self.bundle_path,
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.stdout

    def test_cli_bundle_install_unknown_engine(self):
        result = runner.invoke(cli_app, [
            "engines", "install", "nonexistent",
            "--bundle", self.bundle_path,
        ])
        assert result.exit_code != 0

    def test_cli_bundle_install_missing_bundle(self):
        result = runner.invoke(cli_app, [
            "engines", "install", "synthstrip",
            "--bundle", "/nonexistent/bundle",
        ])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower()


# --------------------------------------------------------------------- #
# End-to-end: bundle install → binary discovery → adapter runs
# --------------------------------------------------------------------- #

def test_end_to_end_bundle_install_then_run(tmp_path, monkeypatch):
    """The full PR15 flow:
       1. Create a realistic synthstrip bundle (fake binary that copies input)
       2. Install via CLI --bundle
       3. Binary discovery finds it in the local store
       4. Run brain_extraction with engine=synthstrip via CLI
       5. Verify the binary was actually invoked (provenance records it)
    """
    from app.engines.manager import _reset_engine_manager

    # Set up isolated store
    monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
    _reset_engine_manager()

    # Clear env vars so only the store can provide the binary
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    # Create a realistic synthstrip bundle
    bundle_path = _make_realistic_synthstrip_bundle(tmp_path / "bundle")

    # Install via CLI
    result = runner.invoke(cli_app, [
        "engines", "install", "synthstrip",
        "--bundle", str(bundle_path),
    ])
    assert result.exit_code == 0

    # Verify binary discovery finds it
    from app.engines.binary_discovery import find_binary
    from app.engines.brain_extraction.synthstrip import SYNTHSTRIP_DISCOVERY
    path, reason = find_binary(SYNTHSTRIP_DISCOVERY)
    assert path is not None
    assert path.endswith("bin/mri_synthstrip")

    # Build a tiny T1 dataset
    input_dir = tmp_path / "input" / "sub-001"
    input_dir.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(input_dir / "sub-001_T1.nii.gz"),
    )
    output_dir = str(tmp_path / "output")

    # Run brain_extraction with engine=synthstrip via CLI
    result = runner.invoke(cli_app, [
        "run", str(input_dir.parent), output_dir,
        "--steps", "brain_extraction",
        "--modalities", "T1",
        "--param", "brain_extraction.engine=synthstrip",
    ])
    assert result.exit_code == 0
    assert "1 succeeded" in result.stdout

    # Verify the provenance log records the synthstrip engine + binary path
    import json
    prov_dir = os.path.join(output_dir, "provenance")
    prov_files = [f for f in os.listdir(prov_dir) if f.endswith(".jsonl")]
    with open(os.path.join(prov_dir, prov_files[0])) as f:
        records = [json.loads(l) for l in f if l.strip()]
    be_record = next(r for r in records if r.get("step_id") == "brain_extraction")
    assert be_record["params"]["engine"] == "synthstrip"
    assert be_record["metrics"]["engine"] == "synthstrip"
    assert "binary_path" in be_record["metrics"]
    # The binary path should be inside the engine store
    assert "store" in be_record["metrics"]["binary_path"]


# --------------------------------------------------------------------- #
# Symlink bundle: end-to-end
# --------------------------------------------------------------------- #

def test_end_to_end_bundle_symlink_then_run(tmp_path, monkeypatch):
    """Same as above but with --symlink mode. The binary discovery
    chain should follow the symlink and find the binary."""
    from app.engines.manager import _reset_engine_manager

    monkeypatch.setenv("MRI_PREPROCESSOR_ENGINES_DIR", str(tmp_path / "store"))
    _reset_engine_manager()
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    bundle_path = _make_realistic_synthstrip_bundle(tmp_path / "bundle")

    # Install via CLI with --symlink
    result = runner.invoke(cli_app, [
        "engines", "install", "synthstrip",
        "--bundle", str(bundle_path),
        "--symlink",
    ])
    assert result.exit_code == 0

    # Verify the store entry is a symlink
    manager = get_engine_manager()
    install_path = manager.get_install_path("synthstrip")
    assert os.path.islink(install_path)

    # Binary discovery follows the symlink
    from app.engines.binary_discovery import find_binary
    from app.engines.brain_extraction.synthstrip import SYNTHSTRIP_DISCOVERY
    path, _ = find_binary(SYNTHSTRIP_DISCOVERY)
    assert path is not None
    assert path.endswith("bin/mri_synthstrip")

    # Run brain_extraction -- the symlinked binary should execute
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
    assert result.exit_code == 0
    assert "1 succeeded" in result.stdout
