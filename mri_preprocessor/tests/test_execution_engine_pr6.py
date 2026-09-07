"""
Tests for PR6: skip_recommended propagation + synchronous run_job core.

Covers:
  - `skip_recommended` flows from API -> start_job -> run_job ->
    resolve_pipeline, and the EXECUTED pipeline omits the skipped
    recommended step. Previously the API accepted skip_recommended
    for preview only; the actual job ignored it.
  - `run_job()` is the synchronous core: callable directly, returns
    the result dict, raises on hard failure.
  - `start_job()` wraps run_job in a thread for Flask; behavior
    unchanged for existing callers that don't pass skip_recommended.
  - The `progress_callback` arg lets the CLI receive live updates
    without polling the in-memory job store.
"""
import os
import threading
import time

import nibabel as nib
import numpy as np
import pytest

from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.step_registry import Dependency, ProcessingStep, registry


# --------------------------------------------------------------------- #
# Test fixtures: register throwaway steps that use tiered dependencies
# --------------------------------------------------------------------- #

# We reuse test steps from test_dependency_tiers.py if they're already
# registered (pytest imports modules once per session).
# These registrations are idempotent -- the registry raises on duplicates,
# so we catch and ignore.

def _safe_register(step_id, depends_on=None, dependencies=None,
                    default_params=None, modalities=("T1",)):
    try:
        registry.get(step_id)
        return
    except KeyError:
        pass
    registry.register(
        ProcessingStep(
            id=step_id,
            name=f"Test step {step_id}",
            modalities=list(modalities),
            engine="test",
            depends_on=depends_on or [],
            dependencies=dependencies or [],
            default_params=default_params or {},
        ),
        # No-op step implementation that just returns the input path
        # (we're testing the engine, not the step).
        lambda input_path, output_dir, params, context: {
            "output_path": input_path, "outputs": {}, "metrics": {},
        },
    )


# Layout:
#   pr6_root                -- no deps
#   pr6_recommended_user    -- recommends pr6_root
#   pr6_conditional_user    -- conditional on pr6_root (when foo == "bar")
_safe_register("pr6_root")
_safe_register("pr6_recommended_user",
                dependencies=[Dependency(step_id="pr6_root", tier="recommended")])
_safe_register("pr6_conditional_user",
                dependencies=[Dependency(
                    step_id="pr6_root", tier="conditional", when={"foo": "bar"},
                )])


# --------------------------------------------------------------------- #
# skip_recommended propagation through run_job
# --------------------------------------------------------------------- #

def test_run_job_omits_recommended_step_when_skipped(tmp_path):
    """When skip_recommended is passed, the recommended step is NOT in
    the executed pipeline."""
    from app.execution.engine import run_job

    # Build a minimal input
    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    # Run WITHOUT skip_recommended: pr6_root IS in the pipeline (recommended)
    result_with = run_job(
        job_id="test-1",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-with"),
        selected_step_ids=["pr6_recommended_user"],
        step_params={},
    )
    pipeline_with = result_with["pipeline"]["ordered_pipeline"]
    assert "pr6_root" in pipeline_with
    assert "pr6_recommended_user" in pipeline_with

    # Run WITH skip_recommended=["pr6_root"]: pr6_root is OMITTED
    result_without = run_job(
        job_id="test-2",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-without"),
        selected_step_ids=["pr6_recommended_user"],
        step_params={},
        skip_recommended=["pr6_root"],
    )
    pipeline_without = result_without["pipeline"]["ordered_pipeline"]
    assert "pr6_recommended_user" in pipeline_without
    assert "pr6_root" not in pipeline_without


def test_run_job_conditional_dep_fires_when_when_matches(tmp_path):
    """Conditional dep with matching when= fires during execution."""
    from app.execution.engine import run_job

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    result = run_job(
        job_id="test-cond-fire",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-cond-fire"),
        selected_step_ids=["pr6_conditional_user"],
        step_params={"pr6_conditional_user": {"foo": "bar"}},
    )
    pipeline = result["pipeline"]["ordered_pipeline"]
    assert "pr6_root" in pipeline
    assert "pr6_conditional_user" in pipeline


def test_run_job_conditional_dep_does_not_fire_when_when_doesnt_match(tmp_path):
    """Conditional dep without matching when= does NOT fire."""
    from app.execution.engine import run_job

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    result = run_job(
        job_id="test-cond-nofire",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-cond-nofire"),
        selected_step_ids=["pr6_conditional_user"],
        step_params={"pr6_conditional_user": {"foo": "baz"}},
    )
    pipeline = result["pipeline"]["ordered_pipeline"]
    assert "pr6_conditional_user" in pipeline
    assert "pr6_root" not in pipeline


# --------------------------------------------------------------------- #
# Synchronous run_job behavior
# --------------------------------------------------------------------- #

def test_run_job_returns_result_dict(tmp_path):
    """run_job returns a result dict (not a job_id) -- the synchronous
    core's contract."""
    from app.execution.engine import run_job

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    result = run_job(
        job_id="test-sync",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-sync"),
        selected_step_ids=["reorientation"],
        step_params={},
    )

    assert isinstance(result, dict)
    assert "dataset_summary" in result
    assert "qc_report" in result
    assert "pipeline" in result
    assert "per_file_results" in result
    # Pipeline should have just reorientation (no deps)
    assert result["pipeline"]["ordered_pipeline"] == ["reorientation"]
    # One file processed successfully
    assert len(result["per_file_results"]) == 1
    assert result["per_file_results"][0]["status"] == "success"


def test_run_job_progress_callback_invoked(tmp_path):
    """The progress_callback is invoked at least once with the running
    status and final progress count."""
    from app.execution.engine import run_job

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    statuses_seen = []

    def callback(snapshot):
        statuses_seen.append(snapshot.get("status"))

    run_job(
        job_id="test-progress",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-progress"),
        selected_step_ids=["reorientation"],
        step_params={},
        progress_callback=callback,
    )

    # We should see at least "running" (during execution) and "completed"
    assert "running" in statuses_seen
    assert "completed" in statuses_seen


def test_run_job_writes_provenance_log(tmp_path):
    """run_job writes a provenance JSONL log under output_dir/provenance/."""
    from app.execution.engine import run_job

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    job_id = "test-prov"
    output_dir = str(tmp_path / "out-prov")
    run_job(
        job_id=job_id,
        input_dir=str(tmp_path),
        output_dir=output_dir,
        selected_step_ids=["reorientation"],
        step_params={},
    )

    prov_path = os.path.join(output_dir, "provenance", f"{job_id}.jsonl")
    assert os.path.exists(prov_path)

    from app.provenance.tracker import ProvenanceTracker
    tracker = ProvenanceTracker(prov_path)
    history = tracker.read_all()
    events = [r.get("event") for r in history]
    assert "job_started" in events
    assert "job_completed" in events


# --------------------------------------------------------------------- #
# Backward compatibility: start_job (Flask path) still works
# --------------------------------------------------------------------- #

def test_start_job_accepts_skip_recommended(tmp_path):
    """start_job now accepts skip_recommended and propagates it to the
    background thread -> run_job -> resolve_pipeline."""
    from app.execution.engine import start_job, get_job

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    job_id = start_job(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-async"),
        selected_step_ids=["pr6_recommended_user"],
        step_params={},
        skip_recommended=["pr6_root"],
    )

    # Poll until done
    deadline = time.time() + 30
    job = get_job(job_id)
    while job["status"] in ("pending", "running") and time.time() < deadline:
        time.sleep(0.05)
        job = get_job(job_id)

    assert job["status"] == "completed"
    # pr6_root should NOT be in the executed pipeline
    pipeline = job["result"]["pipeline"]["ordered_pipeline"]
    assert "pr6_root" not in pipeline
    assert "pr6_recommended_user" in pipeline


def test_start_job_backward_compat_without_skip_recommended(tmp_path):
    """Existing callers that don't pass skip_recommended still work."""
    from app.execution.engine import start_job, get_job

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    job_id = start_job(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-compat"),
        selected_step_ids=["reorientation"],
        step_params={},
        # No skip_recommended -- backward compat
    )

    deadline = time.time() + 30
    job = get_job(job_id)
    while job["status"] in ("pending", "running") and time.time() < deadline:
        time.sleep(0.05)
        job = get_job(job_id)

    assert job["status"] == "completed"
    assert job["result"]["pipeline"]["ordered_pipeline"] == ["reorientation"]


# --------------------------------------------------------------------- #
# API: /api/jobs propagates skip_recommended
# --------------------------------------------------------------------- #

def test_api_jobs_endpoint_accepts_skip_recommended(tmp_path):
    """The /api/jobs endpoint accepts skip_recommended in the JSON body
    and propagates it through to the executed pipeline."""
    from app.main import app

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "T1.nii.gz"),
    )

    with app.test_client() as c:
        # Start the job with skip_recommended
        r = c.post("/api/jobs", json={
            "input_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out-api"),
            "step_ids": ["pr6_recommended_user"],
            "skip_recommended": ["pr6_root"],
        })
        assert r.status_code == 202
        job_id = r.get_json()["job_id"]

        # Poll until done
        deadline = time.time() + 30
        while True:
            jr = c.get(f"/api/jobs/{job_id}")
            data = jr.get_json()
            if data["status"] in ("completed", "failed"):
                break
            if time.time() > deadline:
                pytest.fail("Job did not complete in time")
            time.sleep(0.05)

        assert data["status"] == "completed"
        pipeline = data["result"]["pipeline"]["ordered_pipeline"]
        assert "pr6_root" not in pipeline
        assert "pr6_recommended_user" in pipeline


# --------------------------------------------------------------------- #
# run_job handles target_modalities filter
# --------------------------------------------------------------------- #

def test_run_job_filters_target_modalities(tmp_path):
    """The target_modalities filter is honored by run_job (same as the
    old _run_job)."""
    from app.execution.engine import run_job

    # Build two subjects: one T1, one fMRI
    for sub_name in ("sub-001", "sub-002"):
        sub = tmp_path / sub_name
        sub.mkdir()

    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(tmp_path / "sub-001" / "sub-001_T1.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8, 10).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(tmp_path / "sub-002" / "sub-002_task-rest_bold.nii.gz"),
    )

    # Filter to T1 only
    result = run_job(
        job_id="test-filter",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out-filter"),
        selected_step_ids=["reorientation"],
        step_params={},
        target_modalities=["T1"],
    )

    per_file = result["per_file_results"]
    # Only the T1 file should be processed
    assert len(per_file) == 1
    assert per_file[0]["modality"] == "T1"
