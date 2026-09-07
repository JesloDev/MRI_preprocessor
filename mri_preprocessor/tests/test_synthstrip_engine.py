"""
Tests for the SynthStrip engine (PR9).

Real SynthStrip isn't installed in the test environment -- and we
wouldn't want to depend on a 5GB FreeSurfer download just to run unit
tests anyway. So we use a tiny fake `mri_synthstrip` shell script that
mimics the interface:

    fake_mri_synthstrip --i <input> --o <brain> --mask <mask>

It just copies the input NIfTI to --o and writes a zero-filled mask
to --mask. The SynthStrip wrapper's logic (binary discovery, command
construction, subprocess execution, output validation, result-dict
shape) is fully exercised without needing the real tool.

Covers:
  - subprocess execution against the fake binary
  - output files exist after a successful run
  - result dict shape matches the otsu engine's contract (output_path,
    outputs["brain_mask"], metrics["engine"])
  - subprocess failure produces a clear RuntimeError with stderr in
    the message
  - missing output file produces a clear RuntimeError
  - brain_mask flows into SubjectContext via context.update -- this
    is the load-bearing integration that proves the engine adapter
    pattern works end-to-end: a step running on the T1 acquisition
    makes its brain mask available to subsequent steps operating on
    sibling acquisitions of the same subject.
  - extra_args flow through to the subprocess command
  - binary discovery chain works for SynthStrip via env vars and PATH
"""
import os
import stat
import subprocess
from unittest import mock

import nibabel as nib
import numpy as np
import pytest

from app.engines.brain_extraction.synthstrip import (
    SYNTHSTRIP_DISCOVERY, SynthStripBrainExtraction,
)
from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.step_registry import registry


# --------------------------------------------------------------------- #
# Fake mri_synthstrip helper
# --------------------------------------------------------------------- #

def _make_fake_synthstrip_binary(binary_path, *, fail_with=None,
                                  skip_brain=False, skip_mask=False):
    """Creates a fake mri_synthstrip shell script.

    Args:
        binary_path: where to write the fake script (must be writable).
        fail_with: if set, the script exits with this code and writes
            an error to stderr.
        skip_brain: if True, the script doesn't produce the brain output.
        skip_mask: if True, the script doesn't produce the mask output.
    """
    lines = ["#!/bin/bash"]
    lines.append('echo "fake mri_synthstrip called with: $@" >&2')

    if fail_with is not None:
        lines.append(f'echo "fake_synthstrip failing on purpose" >&2')
        lines.append(f"exit {fail_with}")
        binary_path.write_text("\n".join(lines))
        binary_path.chmod(0o755)
        return

    # Parse --i, --o, --mask arguments
    # We need a tiny bit of bash logic to extract the values
    lines.append("""
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
""")
    if not skip_brain:
        # Copy input to brain output
        lines.append('[ -n "$BRAIN" ] && cp "$INPUT" "$BRAIN"')
    if not skip_mask:
        # Write a tiny zero-filled NIfTI mask. We use python here because
        # we can't depend on FreeSurfer tools. Using python at runtime is
        # fine in tests since the test env has nibabel.
        lines.append("""[ -n "$MASK" ] && python3 -c "
import nibabel as nib
import numpy as np
img = nib.load('$INPUT')
shape = img.shape[:3]
mask = np.zeros(shape, dtype='uint8')
nib.save(nib.Nifti1Image(mask, img.affine), '$MASK')
" """)
    lines.append("exit 0")
    binary_path.write_text("\n".join(lines))
    binary_path.chmod(0o755)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

@pytest.fixture
def synthstrip_env(monkeypatch, tmp_path):
    """Sets up a fake mri_synthstrip binary via $MRI_PREPROCESSOR_SYNTHSTRIP.

    Returns a dict with the binary path so tests can customize behavior
    by re-creating the script with different fail_with / skip_brain /
    skip_mask flags.
    """
    fake_bin = tmp_path / "fake_mri_synthstrip"
    _make_fake_synthstrip_binary(fake_bin)
    monkeypatch.setenv("MRI_PREPROCESSOR_SYNTHSTRIP", str(fake_bin))
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    return {"binary": fake_bin}


@pytest.fixture
def t1_image(tmp_path):
    """A small T1 NIfTI file."""
    img_path = tmp_path / "t1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(img_path),
    )
    return str(img_path)


# --------------------------------------------------------------------- #
# Availability check (re-tested here because SynthStrip now has real run())
# --------------------------------------------------------------------- #

def test_synthstrip_is_available_when_binary_set(synthstrip_env):
    """When $MRI_PREPROCESSOR_SYNTHSTRIP points at a valid binary,
    is_available() returns True."""
    engine = SynthStripBrainExtraction()
    assert engine.is_available() is True
    assert engine.availability_reason() == ""


def test_synthstrip_unavailable_when_no_binary(monkeypatch):
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    engine = SynthStripBrainExtraction()
    assert engine.is_available() is False
    assert "mri_synthstrip" in engine.availability_reason()


# --------------------------------------------------------------------- #
# Successful execution
# --------------------------------------------------------------------- #

def test_synthstrip_run_produces_brain_and_mask(synthstrip_env, t1_image, tmp_path):
    """A successful SynthStrip run produces brain.nii.gz and brain_mask.nii.gz."""
    output_dir = str(tmp_path / "synthstrip_out")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    result = engine.run(
        input_path=t1_image,
        output_dir=output_dir,
        params={},
        context={},
    )

    # Result dict shape matches the otsu engine's contract
    assert result["output_path"].endswith("brain.nii.gz")
    assert os.path.isfile(result["output_path"])

    assert "brain_mask" in result["outputs"]
    assert result["outputs"]["brain_mask"].endswith("brain_mask.nii.gz")
    assert os.path.isfile(result["outputs"]["brain_mask"])

    # Metrics include the engine name and binary path
    assert result["metrics"]["engine"] == "synthstrip"
    assert "binary_path" in result["metrics"]
    assert result["metrics"]["returncode"] == 0


def test_synthstrip_mask_is_loadable_nifti(synthstrip_env, t1_image, tmp_path):
    """The produced brain_mask is a loadable NIfTI with the same shape
    as the input."""
    output_dir = str(tmp_path / "synthstrip_out2")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    result = engine.run(
        input_path=t1_image,
        output_dir=output_dir,
        params={},
        context={},
    )

    mask_img = nib.load(result["outputs"]["brain_mask"])
    input_img = nib.load(t1_image)
    assert mask_img.shape[:3] == input_img.shape[:3]


# --------------------------------------------------------------------- #
# Subprocess failure
# --------------------------------------------------------------------- #

def test_synthstrip_subprocess_failure_raises_runtime_error(synthstrip_env, t1_image, tmp_path, monkeypatch):
    """When mri_synthstrip exits non-zero, the engine raises a clear
    RuntimeError that includes stderr so the user can diagnose."""
    # Replace the fake binary with one that fails
    _make_fake_synthstrip_binary(synthstrip_env["binary"], fail_with=42)

    output_dir = str(tmp_path / "synthstrip_fail")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    with pytest.raises(RuntimeError) as exc_info:
        engine.run(
            input_path=t1_image,
            output_dir=output_dir,
            params={},
            context={},
        )

    msg = str(exc_info.value)
    # Should include exit code and stderr output
    assert "42" in msg  # exit code
    assert "fake_synthstrip failing on purpose" in msg  # stderr


# --------------------------------------------------------------------- #
# Missing output validation
# --------------------------------------------------------------------- #

def test_synthstrip_missing_brain_output_raises(synthstrip_env, t1_image, tmp_path):
    """If the binary doesn't produce a brain output file, the engine
    raises a clear RuntimeError instead of returning a half-success."""
    _make_fake_synthstrip_binary(synthstrip_env["binary"], skip_brain=True)

    output_dir = str(tmp_path / "synthstrip_no_brain")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    with pytest.raises(RuntimeError) as exc_info:
        engine.run(
            input_path=t1_image,
            output_dir=output_dir,
            params={},
            context={},
        )
    assert "brain output is missing" in str(exc_info.value)


def test_synthstrip_missing_mask_output_raises(synthstrip_env, t1_image, tmp_path):
    """If the binary doesn't produce a mask file, the engine raises."""
    _make_fake_synthstrip_binary(synthstrip_env["binary"], skip_mask=True)

    output_dir = str(tmp_path / "synthstrip_no_mask")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    with pytest.raises(RuntimeError) as exc_info:
        engine.run(
            input_path=t1_image,
            output_dir=output_dir,
            params={},
            context={},
        )
    assert "brain mask output is missing" in str(exc_info.value)


# --------------------------------------------------------------------- #
# Binary not found
# --------------------------------------------------------------------- #

def test_synthstrip_run_raises_when_binary_missing(t1_image, tmp_path, monkeypatch):
    """Without a discoverable binary, run() raises FileNotFoundError
    with a helpful message about what was checked."""
    monkeypatch.delenv("MRI_PREPROCESSOR_SYNTHSTRIP", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    output_dir = str(tmp_path / "synthstrip_no_binary")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    with pytest.raises(FileNotFoundError) as exc_info:
        engine.run(
            input_path=t1_image,
            output_dir=output_dir,
            params={},
            context={},
        )
    msg = str(exc_info.value)
    assert "mri_synthstrip" in msg
    assert "otsu" in msg  # suggests the fallback


# --------------------------------------------------------------------- #
# Explicit binary_path param overrides discovery
# --------------------------------------------------------------------- #

def test_synthstrip_explicit_binary_path_param(synthstrip_env, t1_image, tmp_path):
    """params["binary_path"] overrides the discovery chain."""
    # Point at a different fake binary
    alt_bin = tmp_path / "alt_mri_synthstrip"
    _make_fake_synthstrip_binary(alt_bin)

    output_dir = str(tmp_path / "synthstrip_explicit")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    result = engine.run(
        input_path=t1_image,
        output_dir=output_dir,
        params={"binary_path": str(alt_bin)},
        context={},
    )
    assert os.path.isfile(result["output_path"])
    assert os.path.isfile(result["outputs"]["brain_mask"])
    # The metrics should record which binary was used
    assert result["metrics"]["binary_path"] == str(alt_bin)


# --------------------------------------------------------------------- #
# extra_args pass-through
# --------------------------------------------------------------------- #

def test_synthstrip_extra_args_passed_to_subprocess(synthstrip_env, t1_image, tmp_path, monkeypatch):
    """params["extra_args"] (a list) is appended to the command line.
    This is the escape hatch for SynthStrip options we don't model
    explicitly (e.g. --no-cuda)."""
    # Use a fake binary that logs its argv to a file so we can inspect
    log_path = tmp_path / "argv.log"
    fake_bin = synthstrip_env["binary"]
    fake_bin.write_text(f"""#!/bin/bash
echo "$@" > {log_path}
# Parse and write outputs as usual
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
    fake_bin.chmod(0o755)

    output_dir = str(tmp_path / "synthstrip_extra_args")
    os.makedirs(output_dir, exist_ok=True)

    engine = SynthStripBrainExtraction()
    result = engine.run(
        input_path=t1_image,
        output_dir=output_dir,
        params={"extra_args": ["--no-cuda", "--sd", "/tmp/subjects"]},
        context={},
    )

    # Outputs should exist (run succeeded)
    assert os.path.isfile(result["output_path"])
    # Check the logged argv contains the extra args
    logged_argv = log_path.read_text()
    assert "--no-cuda" in logged_argv
    assert "--sd" in logged_argv
    assert "/tmp/subjects" in logged_argv


# --------------------------------------------------------------------- #
# Integration with the dispatcher + SubjectContext
# --------------------------------------------------------------------- #

def test_synthstrip_via_dispatcher_writes_brain_mask_to_context(
    synthstrip_env, t1_image, tmp_path,
):
    """End-to-end: brain_extraction step with engine=synthstrip runs
    via the dispatcher, and the brain_mask it produces is written
    into the SubjectContext by dag.py.

    This is the load-bearing test that proves the engine adapter
    pattern works: a step running on the T1 acquisition makes its
    brain mask available to subsequent steps operating on sibling
    acquisitions of the same subject.
    """
    # Build a SubjectContext that we can inspect after the step runs
    from app.data_model import Acquisition, Subject, SubjectContext
    subject = Subject(subject_id="001", acquisitions=[
        Acquisition(
            acquisition_id="T1_1", path=t1_image,
            filename=os.path.basename(t1_image),
            modality="T1",
        ),
    ])
    ctx = SubjectContext(subject)

    # Run brain_extraction via the dispatcher
    brain_extraction_fn = registry.get_fn("brain_extraction")
    output_dir = str(tmp_path / "dispatcher_out")
    os.makedirs(output_dir, exist_ok=True)

    result = brain_extraction_fn(
        input_path=t1_image,
        output_dir=output_dir,
        params={"engine": "synthstrip"},
        context=ctx,
    )

    # The result shape matches the otsu engine
    assert result["output_path"].endswith("brain.nii.gz")
    assert os.path.isfile(result["output_path"])
    assert "brain_mask" in result["outputs"]
    assert os.path.isfile(result["outputs"]["brain_mask"])

    # When dag.py calls `context.update(result["outputs"])`, the brain_mask
    # is written into the SubjectContext -- making it available to
    # subsequent steps on sibling acquisitions. Simulate that update:
    ctx.update(result["outputs"])

    # Now the brain_mask should be accessible via the flat dict-like
    # interface AND via the structured .masks dict
    assert ctx.get("brain_mask") == result["outputs"]["brain_mask"]
    assert ctx.masks["brain_mask"] == result["outputs"]["brain_mask"]
    assert "brain_mask" in ctx  # __contains__


def test_synthstrip_via_cli_run_command(monkeypatch, tmp_path):
    """End-to-end via the CLI: `python -m app.cli run ... --param
    brain_extraction.engine=synthstrip` works against a fake SynthStrip
    binary."""
    from typer.testing import CliRunner
    from app.cli import app

    # Build a 2-subject T1 dataset
    input_dir = tmp_path / "input"
    for s in range(2):
        sub = input_dir / f"sub-{s:03d}"
        sub.mkdir(parents=True)
        nib.save(
            nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                            affine=np.eye(4)),
            str(sub / f"sub-{s:03d}_T1.nii.gz"),
        )

    # Set up a fake SynthStrip binary via env var
    fake_bin = tmp_path / "fake_mri_synthstrip"
    _make_fake_synthstrip_binary(fake_bin)
    monkeypatch.setenv("MRI_PREPROCESSOR_SYNTHSTRIP", str(fake_bin))
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    runner = CliRunner()
    result = runner.invoke(app, [
        "run",
        str(input_dir),
        str(tmp_path / "cli_synthstrip_out"),
        "--steps", "brain_extraction",
        "--modalities", "T1",
        "--param", "brain_extraction.engine=synthstrip",
    ])
    assert result.exit_code == 0
    assert "Job completed" in result.stdout
    # 2 subjects * 1 T1 = 2 successes
    assert "2 succeeded" in result.stdout


# --------------------------------------------------------------------- #
# Re-tested availability: PR9 keeps the PR7 contract
# --------------------------------------------------------------------- #

def test_synthstrip_engine_metadata_after_real_implementation():
    """After PR9 (real subprocess wrapper), the engine's metadata is
    still correctly reported."""
    engine = SynthStripBrainExtraction()
    assert engine.name == "synthstrip"
    assert engine.maturity == "external_engine"
    assert engine.requires_external_tool is True
    assert engine.license_notes is not None
    assert "FreeSurfer" in engine.license_notes
