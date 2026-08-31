"""
End-to-end test: the engine builds a real SubjectContext per subject and
passes it through run_pipeline_for_file. Steps can read sibling-acquisition
references from the context -- which is what unblocks multimodal processing.

This test is the canary for PR1: if the existing
test_structural_pipeline_runs_end_to_end still passes AND this test passes,
the refactor is safe to roll forward.
"""
import os

import nibabel as nib
import numpy as np
import pytest

from app.data_model import SubjectContext, build_dataset
from app.pipeline.bootstrap import *  # noqa: F401,F403 -- registers all steps
from app.pipeline.dag import run_pipeline_for_file
from app.pipeline.dependency_resolver import resolve_pipeline
from app.provenance.tracker import ProvenanceTracker


def test_subject_context_flows_through_structural_pipeline(tmp_path):
    """A Subject built from a real dataset can be passed to
    run_pipeline_for_file as the context. Step code that does
    `context.get(...)` / `context.update({...})` keeps working."""
    # Build a minimal dataset: one subject, one T1
    sub_dir = tmp_path / "sub-001" / "anat"
    sub_dir.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub_dir / "sub-001_T1.nii.gz"),
    )

    ds = build_dataset(str(tmp_path))
    subject = ds.get_subject("001")
    assert subject is not None
    t1_acq = subject.get_primary_acquisition("T1")
    assert t1_acq is not None

    ctx = SubjectContext(subject)
    # Flat keys should be populated for step code to read
    assert ctx.get("t1_path").endswith("sub-001_T1.nii.gz")
    assert ctx.get("reference_image") == ctx.get("t1_path")

    resolution = resolve_pipeline(["bias_field_correction", "resampling"])
    provenance = ProvenanceTracker(str(tmp_path / "prov.jsonl"))

    result = run_pipeline_for_file(
        subject_id="001",
        modality="T1",
        input_path=t1_acq.path,
        ordered_step_ids=resolution["ordered_pipeline"],
        step_params={"resampling": {"voxel_size": [2.0, 2.0, 2.0]}},
        work_dir=str(tmp_path / "work"),
        context=ctx,  # <-- SubjectContext, not {} -- the whole point of PR1
        provenance=provenance,
    )

    assert os.path.exists(result["final_output"])

    # The provenance log should record all 3 steps that ran (reorientation,
    # bias_field_correction, resampling -- the auto-inserted prereqs).
    history = provenance.read_all()
    step_ids_logged = [r["step_id"] for r in history if "step_id" in r]
    assert "reorientation" in step_ids_logged
    assert "bias_field_correction" in step_ids_logged
    assert "resampling" in step_ids_logged


def test_dwi_acquisition_sees_bval_bvec_in_context(tmp_path):
    """When the engine builds a SubjectContext for a DWI acquisition that
    has bval/bvec sidecars, the DWI step code can read them via
    `context.get("bval_path")` -- without the step changing at all.
    """
    sub_dir = tmp_path / "sub-001" / "dwi"
    sub_dir.mkdir(parents=True)
    dwi_data = np.random.rand(8, 8, 8, 4).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(dwi_data, affine=np.eye(4)),
             str(sub_dir / "sub-001_dwi.nii.gz"))
    (sub_dir / "sub-001_dwi.bval").write_text("0 1000 1000 1000\n")
    (sub_dir / "sub-001_dwi.bvec").write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    ds = build_dataset(str(tmp_path))
    subject = ds.get_subject("001")
    ctx = SubjectContext(subject)

    # The whole point: existing step code does `context.get("bval_path")`
    # and that now works without changing the step.
    assert ctx.get("bval_path").endswith("sub-001_dwi.bval")
    assert ctx.get("bvec_path").endswith("sub-001_dwi.bvec")


def test_subject_context_persists_across_multiple_acquisitions(tmp_path):
    """The critical capability: when processing one subject's T1 and FLAIR
    sequentially, the brain_mask written by brain_extraction on the T1
    is visible to subsequent steps on the FLAIR acquisition.

    This is exactly the scenario that the old `context={}` per-file flow
    couldn't support -- each file started fresh, so cross-acquisition
    state was impossible.
    """
    sub_dir = tmp_path / "sub-001"
    sub_dir.mkdir()
    nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                             affine=np.eye(4)),
             str(sub_dir / "T1.nii.gz"))
    nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                             affine=np.eye(4)),
             str(sub_dir / "FLAIR.nii.gz"))

    ds = build_dataset(str(tmp_path))
    subject = ds.get_subject("001")
    ctx = SubjectContext(subject)

    # Simulate brain_extraction on T1 writing brain_mask into the context
    ctx.update({"brain_mask": "/path/to/brain_mask.nii.gz"})

    # When we now process FLAIR, the brain_mask is visible -- which is
    # exactly what a FLAIR-to-T1 registration step needs.
    assert ctx.get("brain_mask") == "/path/to/brain_mask.nii.gz"
    assert ctx.masks["brain_mask"] == "/path/to/brain_mask.nii.gz"

    # The structured acquisitions are also still queryable
    assert len(ctx.get_acquisitions("T1")) == 1
    assert len(ctx.get_acquisitions("FLAIR")) == 1


def test_engine_runs_full_job_with_subject_context(synthetic_dataset):
    """The full engine flow: start_job -> _run_job -> Dataset built ->
    SubjectContext per subject -> run_pipeline_for_file with the context.
    """
    from app.execution.engine import start_job, get_job, list_jobs
    import time

    job_id = start_job(
        input_dir=synthetic_dataset["input_dir"],
        output_dir=synthetic_dataset["output_dir"],
        selected_step_ids=["reorientation"],
        step_params={},
        target_modalities=None,
    )

    # Poll until done (with a timeout)
    deadline = time.time() + 30
    job = get_job(job_id)
    while job["status"] in ("pending", "running") and time.time() < deadline:
        time.sleep(0.1)
        job = get_job(job_id)

    assert job["status"] == "completed", f"Job did not complete: {job}"
    assert job["error"] is None

    result = job["result"]
    # 2 subjects x (T1 + fMRI) = 4 acquisitions processed
    successful = [r for r in result["per_file_results"] if r["status"] == "success"]
    assert len(successful) == 4

    # Each result should have an acquisition_id -- the new field added in PR1
    for r in successful:
        assert "acquisition_id" in r
        assert r["acquisition_id"] in ("T1_1", "fMRI_1")

    # The provenance log should record 4 steps (one per acquisition, since
    # we only ran "reorientation")
    prov = get_job(job_id)  # not used; below we'd fetch provenance via the API
    provenance_path = os.path.join(synthetic_dataset["output_dir"],
                                    "provenance", f"{job_id}.jsonl")
    assert os.path.exists(provenance_path)

    from app.provenance.tracker import ProvenanceTracker
    tracker = ProvenanceTracker(provenance_path)
    history = tracker.read_all()
    step_records = [r for r in history if "step_id" in r]
    assert len(step_records) == 4
    for r in step_records:
        assert r["step_id"] == "reorientation"
