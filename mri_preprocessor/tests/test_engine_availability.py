"""
Tests for PR7: engine availability checks.

Covers:
  - All 7 engines implement is_available() and availability_reason().
  - Pure-Python engines (otsu, simpleitk, dipy) always return True.
  - External engines (synthstrip, hdbet, mcflirt, fsl_eddy) probe via
    the binary discovery chain.
  - Binary discovery chain honors priority:
      1. params["binary_path"] (explicit override)
      2. $<env_var> (engine-specific)
      3. PATH (shutil.which)
      4. $<tool_suite_env>/bin/<binary_name> (tool-suite fallback)
  - /api/engines/<step_id> response surfaces the `available` and
    `availability_reason` fields for each engine.
"""
import os
import shutil
import stat
from unittest import mock

import pytest

from app.engines.binary_discovery import DiscoveryConfig, find_binary
from app.engines.brain_extraction.otsu import OtsuBrainExtraction
from app.engines.brain_extraction.synthstrip import (
    SYNTHSTRIP_DISCOVERY, SynthStripBrainExtraction,
)
from app.engines.brain_extraction.hdbet import (
    HDBET_DISCOVERY, HDBETBrainExtraction,
)
from app.engines.fmri_motion.simpleitk import SimpleITKMotionCorrection
from app.engines.fmri_motion.mcflirt import (
    MCFLIRT_DISCOVERY, MCFLIRTMotionCorrection,
)
from app.engines.dwi_correction.dipy import DIPYEddyCorrection
from app.engines.dwi_correction.fsl_eddy import (
    FSL_EDDY_DISCOVERY, FSLEddyCorrection,
)


# --------------------------------------------------------------------- #
# Pure-Python engines always report available
# --------------------------------------------------------------------- #

def test_otsu_is_available():
    assert OtsuBrainExtraction().is_available() is True
    assert OtsuBrainExtraction().availability_reason() == ""


def test_simpleitk_fmri_is_available():
    assert SimpleITKMotionCorrection().is_available() is True
    assert SimpleITKMotionCorrection().availability_reason() == ""


def test_dipy_eddy_is_available():
    assert DIPYEddyCorrection().is_available() is True
    assert DIPYEddyCorrection().availability_reason() == ""


# --------------------------------------------------------------------- #
# External engines: binary discovery chain
# --------------------------------------------------------------------- #

def test_synthstrip_unavailable_by_default(monkeypatch):
    """Without FreeSurfer installed, synthstrip reports unavailable with
    a helpful message listing what was checked."""
    # Make sure no env vars leak from the test environment
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    # And mri_synthstrip isn't on PATH (it shouldn't be in test envs)
    engine = SynthStripBrainExtraction()
    assert engine.is_available() is False
    reason = engine.availability_reason()
    assert "mri_synthstrip" in reason
    assert "PATH" in reason
    assert "FREESURFER_HOME" in reason
    assert "MRI_PREPROCESSOR_SYNTHSTRIP" in reason


def test_synthstrip_available_via_env_var(monkeypatch, tmp_path):
    """Setting $MRI_PREPROCESSOR_SYNTHSTRIP to an executable makes
    synthstrip report available."""
    fake_bin = tmp_path / "fake_mri_synthstrip"
    fake_bin.write_text("#!/bin/bash\nexit 0\n")
    fake_bin.chmod(0o755)

    monkeypatch.setenv("MRI_PREPROCESSOR_SYNTHSTRIP", str(fake_bin))
    # Make sure no other discovery path interferes
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    engine = SynthStripBrainExtraction()
    assert engine.is_available() is True
    assert engine.availability_reason() == ""


def test_synthstrip_available_via_freesurfer_home(monkeypatch, tmp_path):
    """Setting $FREESURFER_HOME (with bin/mri_synthstrip inside) makes
    synthstrip report available."""
    fake_fs_home = tmp_path / "freesurfer"
    fake_bin_dir = fake_fs_home / "bin"
    fake_bin_dir.mkdir(parents=True)
    fake_bin = fake_bin_dir / "mri_synthstrip"
    fake_bin.write_text("#!/bin/bash\nexit 0\n")
    fake_bin.chmod(0o755)

    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.setenv("FREESURFER_HOME", str(fake_fs_home))

    engine = SynthStripBrainExtraction()
    assert engine.is_available() is True


def test_synthstrip_available_via_path(monkeypatch, tmp_path):
    """Placing mri_synthstrip on PATH makes synthstrip report available."""
    fake_bin_dir = tmp_path
    fake_bin = fake_bin_dir / "mri_synthstrip"
    fake_bin.write_text("#!/bin/bash\nexit 0\n")
    fake_bin.chmod(0o755)

    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    # Prepend our fake dir to PATH so it's found first
    monkeypatch.setenv("PATH", f"{fake_bin_dir}:{os.environ.get('PATH', '')}")

    engine = SynthStripBrainExtraction()
    assert engine.is_available() is True


def test_synthstrip_explicit_param_overrides_env(monkeypatch, tmp_path):
    """If both params['binary_path'] and the env var are set, the
    explicit param wins."""
    fake_param = tmp_path / "param_binary"
    fake_param.write_text("#!/bin/bash\nexit 0\n")
    fake_param.chmod(0o755)

    # Set env var to a non-existent path -- should NOT be used because
    # the explicit param takes priority
    monkeypatch.setenv("MRI_PREPROCESSOR_SYNTHSTRIP",
                       str(tmp_path / "nonexistent"))

    path, reason = find_binary(SYNTHSTRIP_DISCOVERY,
                                explicit_path=str(fake_param))
    assert path == str(fake_param)


def test_synthstrip_non_executable_env_var_reports_unavailable(monkeypatch, tmp_path):
    """If $MRI_PREPROCESSOR_SYNTHSTRIP points at a non-executable file,
    availability is False and the reason mentions the env var."""
    fake_bin = tmp_path / "not_executable"
    fake_bin.write_text("not a script")  # no +x bit
    fake_bin.chmod(0o644)

    monkeypatch.setenv("MRI_PREPROCESSOR_SYNTHSTRIP", str(fake_bin))
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    engine = SynthStripBrainExtraction()
    assert engine.is_available() is False
    reason = engine.availability_reason()
    assert "not executable" in reason.lower()


def test_hdbet_unavailable_by_default(monkeypatch):
    monkeypatch.delenv("MRI_PREPROCESSOR_HDBET", raising=False)
    engine = HDBETBrainExtraction()
    assert engine.is_available() is False
    assert "hd_bet" in engine.availability_reason()


def test_hdbet_available_via_env_var(monkeypatch, tmp_path):
    fake_bin = tmp_path / "fake_hd_bet"
    fake_bin.write_text("#!/bin/bash\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("MRI_PREPROCESSOR_HDBET", str(fake_bin))
    assert HDBETBrainExtraction().is_available() is True


def test_mcflirt_unavailable_by_default(monkeypatch):
    monkeypatch.delenv("MRI_PREPROCESSOR_MCFLIRT", raising=False)
    monkeypatch.delenv("FSLDIR", raising=False)
    engine = MCFLIRTMotionCorrection()
    assert engine.is_available() is False
    reason = engine.availability_reason()
    assert "mcflirt" in reason
    assert "FSLDIR" in reason


def test_mcflirt_available_via_fsldir(monkeypatch, tmp_path):
    fake_fsl_dir = tmp_path / "fsl"
    bin_dir = fake_fsl_dir / "bin"
    bin_dir.mkdir(parents=True)
    fake_bin = bin_dir / "mcflirt"
    fake_bin.write_text("#!/bin/bash\nexit 0\n")
    fake_bin.chmod(0o755)

    monkeypatch.delenv("MRI_PREPROCESSOR_MCFLIRT", raising=False)
    monkeypatch.setenv("FSLDIR", str(fake_fsl_dir))

    assert MCFLIRTMotionCorrection().is_available() is True


def test_fsl_eddy_unavailable_by_default(monkeypatch):
    monkeypatch.delenv("MRI_PREPROCESSOR_FSL_EDDY", raising=False)
    monkeypatch.delenv("FSLDIR", raising=False)
    engine = FSLEddyCorrection()
    assert engine.is_available() is False
    assert "eddy" in engine.availability_reason()


def test_fsl_eddy_available_via_fsldir(monkeypatch, tmp_path):
    fake_fsl_dir = tmp_path / "fsl"
    bin_dir = fake_fsl_dir / "bin"
    bin_dir.mkdir(parents=True)
    fake_bin = bin_dir / "eddy"
    fake_bin.write_text("#!/bin/bash\nexit 0\n")
    fake_bin.chmod(0o755)

    monkeypatch.delenv("MRI_PREPROCESSOR_FSL_EDDY", raising=False)
    monkeypatch.setenv("FSLDIR", str(fake_fsl_dir))

    assert FSLEddyCorrection().is_available() is True


# --------------------------------------------------------------------- #
# Binary discovery: priority ordering
# --------------------------------------------------------------------- #

def test_discovery_priority_explicit_param_over_env(monkeypatch, tmp_path):
    """Explicit param wins over env var."""
    param_bin = tmp_path / "param"
    param_bin.write_text("#!/bin/bash\nexit 0\n")
    param_bin.chmod(0o755)

    env_bin = tmp_path / "env"
    env_bin.write_text("#!/bin/bash\nexit 0\n")
    env_bin.chmod(0o755)

    monkeypatch.setenv("MRI_PREPROCESSOR_TEST", str(env_bin))

    cfg = DiscoveryConfig(
        engine_name="test", binary_name="test_bin",
        env_var="MRI_PREPROCESSOR_TEST",
    )
    path, _ = find_binary(cfg, explicit_path=str(param_bin))
    assert path == str(param_bin)


def test_discovery_returns_reason_listing_what_was_checked(monkeypatch):
    """When the binary isn't found, the reason string lists every
    location that was checked."""
    monkeypatch.delenv("MRI_PREPROCESSOR_TEST", raising=False)
    monkeypatch.delenv("TOOL_SUITE_HOME", raising=False)

    cfg = DiscoveryConfig(
        engine_name="test", binary_name="test_bin",
        env_var="MRI_PREPROCESSOR_TEST",
        tool_suite_env="TOOL_SUITE_HOME",
    )
    path, reason = find_binary(cfg)
    assert path is None
    assert "PATH" in reason
    assert "MRI_PREPROCESSOR_TEST" in reason
    assert "TOOL_SUITE_HOME" in reason


# --------------------------------------------------------------------- #
# Engine registry: available_engines() surfaces availability
# --------------------------------------------------------------------- #

def test_engine_registry_available_engines_includes_availability(monkeypatch):
    """The EngineRegistry.available_engines() output now includes
    `available` and `availability_reason` per engine."""
    from app.engines import BRAIN_EXTRACTION_ENGINES

    # Ensure synthstrip is unavailable in this test
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    engines = BRAIN_EXTRACTION_ENGINES.available_engines()
    assert len(engines) == 3

    by_name = {e["name"]: e for e in engines}
    # otsu is always available
    assert by_name["otsu"]["available"] is True
    assert by_name["otsu"]["availability_reason"] == ""
    # synthstrip is unavailable in this test env
    assert by_name["synthstrip"]["available"] is False
    assert "mri_synthstrip" in by_name["synthstrip"]["availability_reason"]


# --------------------------------------------------------------------- #
# API: /api/engines/<step_id> surfaces availability
# --------------------------------------------------------------------- #

def test_api_engines_includes_availability(monkeypatch):
    """/api/engines/brain_extraction surfaces the `available` field for
    each engine in the available_engines list."""
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/brain_extraction")
        data = r.get_json()

        for engine in data["available_engines"]:
            assert "available" in engine
            assert "availability_reason" in engine

        by_name = {e["name"]: e for e in data["available_engines"]}
        assert by_name["otsu"]["available"] is True
        assert by_name["synthstrip"]["available"] is False
        # Reason should mention the binary name
        assert "mri_synthstrip" in by_name["synthstrip"]["availability_reason"]


def test_api_engines_reflects_env_changes(monkeypatch, tmp_path):
    """If $FREESURFER_HOME is set between requests, the API response
    reflects the new availability state."""
    # First request: synthstrip unavailable
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/brain_extraction")
        data = r.get_json()
        synthstrip = next(
            e for e in data["available_engines"] if e["name"] == "synthstrip"
        )
        assert synthstrip["available"] is False

        # Now set FREESURFER_HOME to a dir with bin/mri_synthstrip
        fake_fs_home = tmp_path / "freesurfer"
        bin_dir = fake_fs_home / "bin"
        bin_dir.mkdir(parents=True)
        fake_bin = bin_dir / "mri_synthstrip"
        fake_bin.write_text("#!/bin/bash\nexit 0\n")
        fake_bin.chmod(0o755)
        monkeypatch.setenv("FREESURFER_HOME", str(fake_fs_home))

        r2 = c.get("/api/engines/brain_extraction")
        data2 = r2.get_json()
        synthstrip2 = next(
            e for e in data2["available_engines"] if e["name"] == "synthstrip"
        )
        assert synthstrip2["available"] is True
        assert synthstrip2["availability_reason"] == ""
