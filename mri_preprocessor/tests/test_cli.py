"""
Tests for the CLI (PR8).

Uses Typer's CliRunner to invoke commands and check stdout/stderr +
exit codes. The tests don't actually render the rich output (CliRunner
captures stdout as plain text); they verify the commands work
end-to-end with the existing backend.
"""
import os

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from app.cli import app


runner = CliRunner()


# --------------------------------------------------------------------- #
# Fixtures: tiny synthetic datasets
# --------------------------------------------------------------------- #

@pytest.fixture
def tiny_dataset(tmp_path):
    """A 2-subject T1+fMRI dataset."""
    input_dir = tmp_path / "input"
    for s in range(2):
        sub = input_dir / f"sub-{s:03d}"
        sub.mkdir(parents=True)
        nib.save(
            nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                            affine=np.eye(4)),
            str(sub / f"sub-{s:03d}_T1.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(np.random.rand(8, 8, 8, 5).astype(np.float32) * 100,
                            affine=np.eye(4)),
            str(sub / f"sub-{s:03d}_task-rest_bold.nii.gz"),
        )
    return {
        "input_dir": str(input_dir),
        "output_dir": str(tmp_path / "output"),
    }


# --------------------------------------------------------------------- #
# Top-level help
# --------------------------------------------------------------------- #

def test_cli_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # All 5 commands should be listed
    for cmd in ("scan", "capabilities", "engines", "resolve", "run"):
        assert cmd in result.stdout


# --------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------- #

def test_scan_command_outputs_subject_count(tiny_dataset):
    result = runner.invoke(app, ["scan", tiny_dataset["input_dir"]])
    assert result.exit_code == 0
    assert "Subjects: 2" in result.stdout
    # Modality counts should include T1 and fMRI
    assert "T1" in result.stdout
    assert "fMRI" in result.stdout


def test_scan_command_outputs_qc_summary(tiny_dataset):
    result = runner.invoke(app, ["scan", tiny_dataset["input_dir"]])
    assert result.exit_code == 0
    assert "Valid files:" in result.stdout
    assert "4" in result.stdout  # 4 valid files


def test_scan_command_supports_depth_strategy(tmp_path):
    """--subject-strategy depth --subject-depth 2 handles the
    Root/ScanFolder/Subject/T1.nii.gz layout."""
    nested = tmp_path / "ScanFolder" / "SubA"
    nested.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(nested / "T1.nii.gz"),
    )

    result = runner.invoke(app, [
        "scan", str(tmp_path),
        "--subject-strategy", "depth",
        "--subject-depth", "2",
    ])
    assert result.exit_code == 0
    # Should report 1 subject (SubA), not "ScanFolder"
    assert "Subjects: 1" in result.stdout


def test_scan_command_missing_directory():
    result = runner.invoke(app, ["scan", "/nonexistent/path/abc/123"])
    assert result.exit_code != 0


# --------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------- #

def test_capabilities_command_lists_all_steps():
    result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == 0
    # Built-in steps should appear
    for step_id in ("reorientation", "bias_field_correction",
                    "brain_extraction", "registration"):
        assert step_id in result.stdout


def test_capabilities_command_modality_filter():
    """--modality T1 should only show T1-applicable steps."""
    result = runner.invoke(app, ["capabilities", "--modality", "T1"])
    assert result.exit_code == 0
    # T1-applicable step should be there
    assert "reorientation" in result.stdout
    # DWI-only step should NOT be there
    assert "dwi_denoise" not in result.stdout


def test_capabilities_command_maturity_filter():
    """--maturity experimental should only show experimental steps."""
    result = runner.invoke(app, ["capabilities", "--maturity", "experimental"])
    assert result.exit_code == 0
    # brain_extraction is the only experimental step
    assert "brain_extraction" in result.stdout
    # production_ready steps should NOT be there
    assert "reorientation" not in result.stdout
    assert "bias_field_correction" not in result.stdout


def test_capabilities_command_no_match_returns_nonzero():
    """If no steps match the filter, exit non-zero."""
    result = runner.invoke(app, [
        "capabilities", "--maturity", "planned",
    ])
    assert result.exit_code != 0


# --------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------- #

def test_engines_command_brain_extraction():
    result = runner.invoke(app, ["engines", "brain_extraction"])
    assert result.exit_code == 0
    # All 3 engine names should appear
    assert "otsu" in result.stdout
    assert "synthstrip" in result.stdout
    assert "hdbet" in result.stdout


def test_engines_command_fmri_motion_correction():
    result = runner.invoke(app, ["engines", "fmri_motion_correction"])
    assert result.exit_code == 0
    assert "simpleitk" in result.stdout
    assert "mcflirt" in result.stdout


def test_engines_command_dwi_motion_eddy_correction():
    result = runner.invoke(app, ["engines", "dwi_motion_eddy_correction"])
    assert result.exit_code == 0
    assert "dipy" in result.stdout
    assert "fsl_eddy" in result.stdout


def test_engines_command_unknown_step():
    result = runner.invoke(app, ["engines", "nonexistent_step"])
    assert result.exit_code != 0


def test_engines_command_step_without_engine_support():
    """A step like 'reorientation' has no engine selection -- the CLI
    should report this rather than crash."""
    result = runner.invoke(app, ["engines", "reorientation"])
    assert result.exit_code == 0
    assert "does not support engine selection" in result.stdout


# --------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------- #

def test_resolve_command_simple():
    result = runner.invoke(app, ["resolve", "reorientation"])
    assert result.exit_code == 0
    assert "reorientation" in result.stdout
    assert "Final execution order" in result.stdout


def test_resolve_command_with_dependencies():
    """resolve registration should auto-insert reorientation +
    bias_field_correction + brain_extraction as hard deps."""
    result = runner.invoke(app, ["resolve", "registration"])
    assert result.exit_code == 0
    # Auto-inserted deps should appear
    assert "reorientation" in result.stdout
    assert "bias_field_correction" in result.stdout
    assert "brain_extraction" in result.stdout
    assert "Final execution order" in result.stdout


def test_resolve_command_skip_recommended():
    """--skip-recommended should appear in the output."""
    # brain_extraction has only hard deps; using --skip-recommended
    # for one of them is a no-op (because it's hard, not recommended).
    # The CLI should still echo what was requested to skip.
    result = runner.invoke(app, [
        "resolve", "brain_extraction",
        "--skip-recommended", "bias_field_correction",
    ])
    assert result.exit_code == 0
    assert "Skipped recommended" in result.stdout
    assert "bias_field_correction" in result.stdout


def test_resolve_command_unknown_step():
    result = runner.invoke(app, ["resolve", "nonexistent_step"])
    assert result.exit_code != 0


def test_resolve_command_with_param():
    """--param should be accepted (and parsed into step_params)."""
    result = runner.invoke(app, [
        "resolve", "brain_extraction",
        "--param", "brain_extraction.engine=otsu",
    ])
    assert result.exit_code == 0


# --------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------- #

def test_run_command_executes_simple_pipeline(tiny_dataset):
    """`run` with one step should produce output files + provenance."""
    result = runner.invoke(app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--steps", "reorientation",
    ])
    assert result.exit_code == 0
    assert "Job completed" in result.stdout
    assert "succeeded" in result.stdout

    # Output tree should exist
    assert os.path.exists(tiny_dataset["output_dir"])
    # Provenance log should exist
    prov_dir = os.path.join(tiny_dataset["output_dir"], "provenance")
    assert os.path.isdir(prov_dir)
    prov_files = [f for f in os.listdir(prov_dir) if f.endswith(".jsonl")]
    assert len(prov_files) == 1


def test_run_command_with_modality_filter(tiny_dataset):
    """--modalities T1 should process only T1 acquisitions."""
    result = runner.invoke(app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--steps", "reorientation",
        "--modalities", "T1",
    ])
    assert result.exit_code == 0
    assert "Modalities filter: T1" in result.stdout
    # 2 subjects * 1 modality = 2 successes
    assert "2 succeeded" in result.stdout


def test_run_command_with_engine_param(tiny_dataset):
    """--param brain_extraction.engine=otsu should flow through to the
    brain_extraction step. End-to-end engine-selection test."""
    result = runner.invoke(app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--steps", "brain_extraction",
        "--modalities", "T1",
        "--param", "brain_extraction.engine=otsu",
    ])
    assert result.exit_code == 0
    # 2 subjects * 1 T1 acquisition = 2 successes
    assert "2 succeeded" in result.stdout


def test_run_command_skip_recommended(tiny_dataset):
    """--skip-recommended should be accepted and propagated."""
    result = runner.invoke(app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--steps", "reorientation",
        "--skip-recommended", "bias_field_correction",  # irrelevant; reorientation has no recommended deps
    ])
    assert result.exit_code == 0


def test_run_command_unknown_step(tiny_dataset):
    """Unknown step should fail early (during resolve)."""
    result = runner.invoke(app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--steps", "nonexistent_step",
    ])
    assert result.exit_code != 0


# --------------------------------------------------------------------- #
# Param parsing helper
# --------------------------------------------------------------------- #

def test_parse_params_simple_key_value():
    from app.cli import parse_params
    result = parse_params(["brain_extraction.engine=otsu"])
    assert result == {"brain_extraction": {"engine": "otsu"}}


def test_parse_params_multiple():
    from app.cli import parse_params
    result = parse_params([
        "brain_extraction.engine=otsu",
        "resampling.voxel_size=[1.0,1.0,1.0]",
    ])
    assert result["brain_extraction"]["engine"] == "otsu"
    # JSON value should be parsed as a list, not a string
    assert result["resampling"]["voxel_size"] == [1.0, 1.0, 1.0]


def test_parse_params_empty():
    from app.cli import parse_params
    assert parse_params([]) == {}


def test_parse_params_invalid_format():
    from app.cli import parse_params
    import typer
    with pytest.raises(typer.BadParameter):
        parse_params(["no_equals_sign"])
    with pytest.raises(typer.BadParameter):
        parse_params(["no_dot_key=value"])  # missing the dot separator


def test_parse_params_json_value_parsed():
    from app.cli import parse_params
    result = parse_params([
        "resampling.voxel_size=[2.0, 2.0, 2.0]",
        "remove_initial_volumes.n_volumes=5",
        "spatial_smoothing.fwhm=4.5",
    ])
    assert result["resampling"]["voxel_size"] == [2.0, 2.0, 2.0]
    assert result["remove_initial_volumes"]["n_volumes"] == 5
    assert result["spatial_smoothing"]["fwhm"] == 4.5


def test_parse_params_string_value_when_not_json():
    from app.cli import parse_params
    # A value that's not valid JSON stays as a string
    result = parse_params(["some_step.some_param=not_a_json_value"])
    assert result == {"some_step": {"some_param": "not_a_json_value"}}
