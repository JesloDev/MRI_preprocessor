"""
Tests for PR5: selective engine adapters.

Covers:
  - ProcessingEngine Protocol contract: every engine has the required
    attributes (name, maturity, validation_notes, license_notes,
    requires_external_tool) and a .run() method.
  - EngineRegistry: register, get, get_default, available_engines,
    and the unknown-engine error message lists available engines.
  - Three engine families ship in PR5:
      brain_extraction: otsu (working), synthstrip (stub), hdbet (stub)
      fmri_motion:      simpleitk (working), mcflirt (stub)
      dwi_correction:   dipy (working), fsl_eddy (stub)
  - Step dispatcher behavior:
      * brain_extraction with no params["engine"] uses the default
        (otsu) and produces the same output as the pre-PR5 code.
      * brain_extraction with params["engine"] = "otsu" produces the
        same output as no engine specified.
      * brain_extraction with params["engine"] = "synthstrip" raises
        NotImplementedError (stub not implemented yet).
      * brain_extraction with params["engine"] = "nonexistent" raises
        KeyError with a useful message listing available engines.
  - API: /api/engines/<step_id> returns available engines + metadata.
  - Engine choice is recorded in the provenance log via params (because
    `engine` is part of default_params and gets logged by the tracker).
"""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.step_registry import registry


# --------------------------------------------------------------------- #
# Engine Protocol contract
# --------------------------------------------------------------------- #

def _check_engine_protocol(engine):
    """Helper: assert an engine satisfies the ProcessingEngine Protocol."""
    assert hasattr(engine, "name") and isinstance(engine.name, str) and engine.name
    assert hasattr(engine, "maturity") and engine.maturity in (
        "planned", "experimental", "reference",
        "production_ready", "external_engine",
    )
    assert hasattr(engine, "validation_notes")
    assert hasattr(engine, "license_notes")
    assert hasattr(engine, "requires_external_tool")
    assert isinstance(engine.requires_external_tool, bool)
    assert hasattr(engine, "run") and callable(engine.run)


def test_otsu_engine_satisfies_protocol():
    from app.engines.brain_extraction.otsu import OtsuBrainExtraction
    _check_engine_protocol(OtsuBrainExtraction())


def test_synthstrip_engine_satisfies_protocol():
    from app.engines.brain_extraction.synthstrip import SynthStripBrainExtraction
    _check_engine_protocol(SynthStripBrainExtraction())


def test_hdbet_engine_satisfies_protocol():
    from app.engines.brain_extraction.hdbet import HDBETBrainExtraction
    _check_engine_protocol(HDBETBrainExtraction())


def test_simpleitk_fmri_engine_satisfies_protocol():
    from app.engines.fmri_motion.simpleitk import SimpleITKMotionCorrection
    _check_engine_protocol(SimpleITKMotionCorrection())


def test_mcflirt_engine_satisfies_protocol():
    from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection
    _check_engine_protocol(MCFLIRTMotionCorrection())


def test_dipy_eddy_engine_satisfies_protocol():
    from app.engines.dwi_correction.dipy import DIPYEddyCorrection
    _check_engine_protocol(DIPYEddyCorrection())


def test_fsl_eddy_engine_satisfies_protocol():
    from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
    _check_engine_protocol(FSLEddyCorrection())


# --------------------------------------------------------------------- #
# Engine metadata
# --------------------------------------------------------------------- #

def test_otsu_engine_metadata():
    """The Otsu engine is the experimental default."""
    from app.engines.brain_extraction.otsu import OtsuBrainExtraction
    e = OtsuBrainExtraction()
    assert e.name == "otsu"
    assert e.maturity == "experimental"
    assert e.license_notes is None
    assert e.requires_external_tool is False
    assert "otsu" in e.validation_notes.lower() or "placeholder" in e.validation_notes.lower()


def test_synthstrip_engine_metadata():
    """SynthStrip is an external_engine stub with non-commercial license."""
    from app.engines.brain_extraction.synthstrip import SynthStripBrainExtraction
    e = SynthStripBrainExtraction()
    assert e.name == "synthstrip"
    assert e.maturity == "external_engine"
    assert e.requires_external_tool is True
    assert e.license_notes is not None
    assert "non-commercial" in e.license_notes.lower() or "freesurfer" in e.license_notes.lower()


def test_hdbet_engine_metadata():
    """HD-BET is an external_engine stub with custom non-commercial license."""
    from app.engines.brain_extraction.hdbet import HDBETBrainExtraction
    e = HDBETBrainExtraction()
    assert e.name == "hdbet"
    assert e.maturity == "external_engine"
    assert e.requires_external_tool is True
    assert e.license_notes is not None


def test_simpleitk_fmri_engine_metadata():
    """SimpleITK fMRI motion is the reference default."""
    from app.engines.fmri_motion.simpleitk import SimpleITKMotionCorrection
    e = SimpleITKMotionCorrection()
    assert e.name == "simpleitk"
    assert e.maturity == "reference"
    assert e.requires_external_tool is False
    assert "mcflirt" in e.validation_notes.lower()


def test_mcflirt_engine_metadata():
    """MCFLIRT is an external_engine stub with FSL non-commercial license."""
    from app.engines.fmri_motion.mcflirt import MCFLIRTMotionCorrection
    e = MCFLIRTMotionCorrection()
    assert e.name == "mcflirt"
    assert e.maturity == "external_engine"
    assert e.requires_external_tool is True
    assert "fsl" in e.license_notes.lower()


def test_dipy_eddy_engine_metadata():
    """DIPY eddy is the reference default."""
    from app.engines.dwi_correction.dipy import DIPYEddyCorrection
    e = DIPYEddyCorrection()
    assert e.name == "dipy"
    assert e.maturity == "reference"
    assert e.requires_external_tool is False
    assert "fsl" in e.validation_notes.lower() or "eddy" in e.validation_notes.lower()


def test_fsl_eddy_engine_metadata():
    """FSL eddy is an external_engine stub with non-commercial license."""
    from app.engines.dwi_correction.fsl_eddy import FSLEddyCorrection
    e = FSLEddyCorrection()
    assert e.name == "fsl_eddy"
    assert e.maturity == "external_engine"
    assert e.requires_external_tool is True
    assert "fsl" in e.license_notes.lower()


# --------------------------------------------------------------------- #
# EngineRegistry behavior
# --------------------------------------------------------------------- #

def test_brain_extraction_engines_registered():
    """Three engines are registered in the brain_extraction family."""
    from app.engines import BRAIN_EXTRACTION_ENGINES
    engines = BRAIN_EXTRACTION_ENGINES.available_engines()
    names = {e["name"] for e in engines}
    assert names == {"otsu", "synthstrip", "hdbet"}


def test_fmri_motion_engines_registered():
    """Two engines are registered in the fmri_motion family."""
    from app.engines import FMRI_MOTION_ENGINES
    engines = FMRI_MOTION_ENGINES.available_engines()
    names = {e["name"] for e in engines}
    assert names == {"simpleitk", "mcflirt"}


def test_dwi_correction_engines_registered():
    """Two engines are registered in the dwi_correction family."""
    from app.engines import DWI_CORRECTION_ENGINES
    engines = DWI_CORRECTION_ENGINES.available_engines()
    names = {e["name"] for e in engines}
    assert names == {"dipy", "fsl_eddy"}


def test_engine_registry_default_is_first_registered():
    """get_default() returns the name of the first-registered engine,
    which is always the built-in pure-Python one (otsu/simpleitk/dipy)."""
    from app.engines import (
        BRAIN_EXTRACTION_ENGINES,
        FMRI_MOTION_ENGINES,
        DWI_CORRECTION_ENGINES,
    )
    assert BRAIN_EXTRACTION_ENGINES.get_default() == "otsu"
    assert FMRI_MOTION_ENGINES.get_default() == "simpleitk"
    DWI_CORRECTION_ENGINES.get_default() == "dipy"


def test_engine_registry_get_unknown_raises_with_helpful_message():
    """Requesting an unregistered engine raises KeyError listing what IS available."""
    from app.engines import BRAIN_EXTRACTION_ENGINES
    with pytest.raises(KeyError) as exc_info:
        BRAIN_EXTRACTION_ENGINES.get("nonexistent_engine")
    msg = str(exc_info.value)
    # Error message should list available engines
    assert "otsu" in msg
    assert "synthstrip" in msg
    assert "hdbet" in msg


def test_engine_registry_available_engines_returns_metadata():
    """available_engines() returns a list of metadata dicts."""
    from app.engines import BRAIN_EXTRACTION_ENGINES
    engines = BRAIN_EXTRACTION_ENGINES.available_engines()
    assert isinstance(engines, list)
    assert len(engines) == 3
    for e in engines:
        assert "name" in e
        assert "maturity" in e
        assert "validation_notes" in e
        assert "license_notes" in e
        assert "requires_external_tool" in e


# --------------------------------------------------------------------- #
# Step dispatcher behavior
# --------------------------------------------------------------------- #

def test_brain_extraction_step_default_params_includes_engine():
    """The brain_extraction step has engine="otsu" in its default_params."""
    s = registry.get("brain_extraction")
    assert "engine" in s.default_params
    assert s.default_params["engine"] == "otsu"


def test_fmri_motion_correction_step_default_params_includes_engine():
    s = registry.get("fmri_motion_correction")
    assert "engine" in s.default_params
    assert s.default_params["engine"] == "simpleitk"


def test_dwi_motion_eddy_correction_step_default_params_includes_engine():
    s = registry.get("dwi_motion_eddy_correction")
    assert "engine" in s.default_params
    assert s.default_params["engine"] == "dipy"


def test_brain_extraction_default_engine_produces_output(tmp_path):
    """Running brain_extraction with no params["engine"] uses the default
    (otsu) and produces the brain + mask files."""
    # Build a small T1 volume
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(t1_path),
    )

    # Get the step function from the registry
    brain_extraction_fn = registry.get_fn("brain_extraction")
    output_dir = str(tmp_path / "brain_out")
    os.makedirs(output_dir, exist_ok=True)

    # Don't pass engine -> default (otsu) should be used
    result = brain_extraction_fn(
        input_path=str(t1_path),
        output_dir=output_dir,
        params={},  # no engine key -- default kicks in
        context={},
    )

    assert os.path.exists(result["output_path"])
    assert os.path.exists(result["outputs"]["brain_mask"])
    assert result["output_path"].endswith("brain.nii.gz")
    assert result["outputs"]["brain_mask"].endswith("brain_mask.nii.gz")


def test_brain_extraction_explicit_otsu_engine(tmp_path):
    """Passing engine="otsu" explicitly produces the same output as default."""
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(t1_path),
    )

    brain_extraction_fn = registry.get_fn("brain_extraction")
    output_dir = str(tmp_path / "brain_explicit_otsu")
    os.makedirs(output_dir, exist_ok=True)

    result = brain_extraction_fn(
        input_path=str(t1_path),
        output_dir=output_dir,
        params={"engine": "otsu"},  # explicit
        context={},
    )

    assert os.path.exists(result["output_path"])
    assert os.path.exists(result["outputs"]["brain_mask"])


def test_brain_extraction_synthstrip_engine_raises_when_binary_missing(tmp_path, monkeypatch):
    """The synthstrip engine raises FileNotFoundError when the binary
    isn't available, with a helpful message about what was checked
    and how to fix it.

    PR9 made the SynthStrip engine a real subprocess wrapper. When
    the binary isn't found (the normal case in test environments
    without FreeSurfer installed), it raises FileNotFoundError
    (NOT NotImplementedError) with the discovery-failure reason.
    """
    # Make sure no env vars leak from the test environment
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    t1_path = tmp_path / "t1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(t1_path),
    )

    brain_extraction_fn = registry.get_fn("brain_extraction")
    output_dir = str(tmp_path / "brain_synthstrip_missing")
    os.makedirs(output_dir, exist_ok=True)

    with pytest.raises(FileNotFoundError) as exc_info:
        brain_extraction_fn(
            input_path=str(t1_path),
            output_dir=output_dir,
            params={"engine": "synthstrip"},
            context={},
        )
    # Error message should explain the engine is not available and
    # point the user at the otsu fallback.
    msg = str(exc_info.value).lower()
    assert "not available" in msg or "mri_synthstrip" in msg
    assert "otsu" in msg  # suggest the fallback


def test_brain_extraction_hdbet_engine_raises_not_implemented(tmp_path):
    """The hdbet engine is a stub; same pattern as synthstrip."""
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(t1_path),
    )

    brain_extraction_fn = registry.get_fn("brain_extraction")
    output_dir = str(tmp_path / "brain_hdbet")
    os.makedirs(output_dir, exist_ok=True)

    with pytest.raises(NotImplementedError):
        brain_extraction_fn(
            input_path=str(t1_path),
            output_dir=output_dir,
            params={"engine": "hdbet"},
            context={},
        )


def test_brain_extraction_unknown_engine_raises_keyerror(tmp_path):
    """An unknown engine name raises KeyError with a useful message."""
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(t1_path),
    )

    brain_extraction_fn = registry.get_fn("brain_extraction")
    output_dir = str(tmp_path / "brain_unknown")
    os.makedirs(output_dir, exist_ok=True)

    with pytest.raises(KeyError) as exc_info:
        brain_extraction_fn(
            input_path=str(t1_path),
            output_dir=output_dir,
            params={"engine": "nonexistent_engine"},
            context={},
        )
    msg = str(exc_info.value)
    # Should list available engines
    assert "otsu" in msg
    assert "synthstrip" in msg


def test_fmri_motion_correction_default_engine_produces_output(tmp_path):
    """The simpleitk fMRI motion correction engine produces output."""
    fmri_path = tmp_path / "bold.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8, 4).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(fmri_path),
    )

    fn = registry.get_fn("fmri_motion_correction")
    output_dir = str(tmp_path / "fmri_out")
    os.makedirs(output_dir, exist_ok=True)

    result = fn(
        input_path=str(fmri_path),
        output_dir=output_dir,
        params={},  # default engine
        context={},
    )
    assert os.path.exists(result["output_path"])
    assert "motion_params" in result["outputs"]


def test_fmri_motion_correction_mcflirt_raises_not_implemented(tmp_path):
    """The mcflirt engine is a stub."""
    fmri_path = tmp_path / "bold.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8, 4).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(fmri_path),
    )

    fn = registry.get_fn("fmri_motion_correction")
    output_dir = str(tmp_path / "fmri_mcflirt")
    os.makedirs(output_dir, exist_ok=True)

    with pytest.raises(NotImplementedError):
        fn(
            input_path=str(fmri_path),
            output_dir=output_dir,
            params={"engine": "mcflirt"},
            context={},
        )


# --------------------------------------------------------------------- #
# Engine choice is recorded in provenance
# --------------------------------------------------------------------- #

def test_engine_choice_logged_in_provenance(tmp_path):
    """When a step runs with a specific engine, the engine name appears
    in the provenance log because `engine` is part of the step's params."""
    from app.provenance.tracker import ProvenanceTracker
    from app.pipeline.dag import run_pipeline_for_file
    from app.pipeline.dependency_resolver import resolve_pipeline

    t1_path = tmp_path / "T1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(t1_path),
    )

    # We need bias_field_correction to run first (brain_extraction depends on it)
    resolution = resolve_pipeline(["brain_extraction"])
    provenance = ProvenanceTracker(str(tmp_path / "prov.jsonl"))

    run_pipeline_for_file(
        subject_id="001",
        modality="T1",
        input_path=str(t1_path),
        ordered_step_ids=resolution["ordered_pipeline"],
        step_params={"brain_extraction": {"engine": "otsu"}},  # explicit engine
        work_dir=str(tmp_path / "work"),
        context={},
        provenance=provenance,
    )

    history = provenance.read_all()
    be_record = next(r for r in history if r.get("step_id") == "brain_extraction")
    # The engine param should be in the logged params
    assert "engine" in be_record["params"]
    assert be_record["params"]["engine"] == "otsu"


# --------------------------------------------------------------------- #
# API: /api/engines/<step_id>
# --------------------------------------------------------------------- #

def test_api_engines_for_brain_extraction_step():
    """/api/engines/brain_extraction returns 3 engines + metadata."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/brain_extraction")
        assert r.status_code == 200
        data = r.get_json()
        assert data["step_id"] == "brain_extraction"
        assert data["supports_engine_selection"] is True
        assert data["default_engine"] == "otsu"
        assert data["current_engine"] == "otsu"  # matches default_params
        assert len(data["available_engines"]) == 3
        names = {e["name"] for e in data["available_engines"]}
        assert names == {"otsu", "synthstrip", "hdbet"}


def test_api_engines_for_fmri_motion_correction_step():
    """/api/engines/fmri_motion_correction returns 2 engines."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/fmri_motion_correction")
        assert r.status_code == 200
        data = r.get_json()
        assert data["supports_engine_selection"] is True
        assert data["default_engine"] == "simpleitk"
        assert len(data["available_engines"]) == 2
        names = {e["name"] for e in data["available_engines"]}
        assert names == {"simpleitk", "mcflirt"}


def test_api_engines_for_dwi_motion_eddy_correction_step():
    """/api/engines/dwi_motion_eddy_correction returns 2 engines."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/dwi_motion_eddy_correction")
        assert r.status_code == 200
        data = r.get_json()
        assert data["supports_engine_selection"] is True
        assert data["default_engine"] == "dipy"
        assert len(data["available_engines"]) == 2


def test_api_engines_for_step_without_engine_support():
    """A step that doesn't support engine selection (e.g. reorientation)
    returns supports_engine_selection=False with a helpful message."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/reorientation")
        assert r.status_code == 200
        data = r.get_json()
        assert data["supports_engine_selection"] is False
        assert "step_id" in data
        assert "message" in data


def test_api_engines_for_unknown_step_returns_404():
    """Unknown step id returns 404."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/nonexistent_step")
        assert r.status_code == 404
        data = r.get_json()
        assert "error" in data


def test_api_engines_includes_maturity_and_license_metadata():
    """The /api/engines response surfaces maturity, validation_notes, and
    license_notes for each engine -- so the UI can render badges and
    license warnings before the user selects an engine."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/brain_extraction")
        data = r.get_json()
        for engine in data["available_engines"]:
            assert "maturity" in engine
            assert "validation_notes" in engine
            assert "license_notes" in engine
            assert "requires_external_tool" in engine

        # The otsu engine should be marked experimental with no license notes
        otsu = next(e for e in data["available_engines"] if e["name"] == "otsu")
        assert otsu["maturity"] == "experimental"
        assert otsu["license_notes"] is None
        assert otsu["requires_external_tool"] is False

        # The synthstrip engine should be marked external_engine with license notes
        synthstrip = next(e for e in data["available_engines"] if e["name"] == "synthstrip")
        assert synthstrip["maturity"] == "external_engine"
        assert synthstrip["license_notes"] is not None
        assert synthstrip["requires_external_tool"] is True


# --------------------------------------------------------------------- #
# Backward compat: end-to-end pipeline still runs successfully
# --------------------------------------------------------------------- #

def test_end_to_end_pipeline_runs_with_engine_dispatcher(tmp_path):
    """The end-to-end pipeline from test_subject_aware_processing still
    works after the dispatcher refactor -- the existing test_structural_
    pipeline_runs_end_to_end is the canary, but this adds an explicit
    check that dispatcher steps (brain_extraction) run within the full
    DAG flow."""
    from app.pipeline.dag import run_pipeline_for_file
    from app.pipeline.dependency_resolver import resolve_pipeline
    from app.provenance.tracker import ProvenanceTracker

    t1_path = tmp_path / "T1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(t1_path),
    )

    resolution = resolve_pipeline(["brain_extraction"])
    provenance = ProvenanceTracker(str(tmp_path / "prov.jsonl"))

    result = run_pipeline_for_file(
        subject_id="001",
        modality="T1",
        input_path=str(t1_path),
        ordered_step_ids=resolution["ordered_pipeline"],
        step_params={"brain_extraction": {"engine": "otsu"}},
        work_dir=str(tmp_path / "work"),
        context={},
        provenance=provenance,
    )

    # The final output of brain_extraction should be the brain.nii.gz file
    assert os.path.exists(result["final_output"])
    assert result["final_output"].endswith("brain.nii.gz")
