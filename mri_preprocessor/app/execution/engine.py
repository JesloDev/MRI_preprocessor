"""
Job execution engine: orchestrates a full run (scan -> classify ->
validate -> resolve pipeline -> execute per subject/modality -> report).

Two execution entry points:
  - `start_job(...)` -- spawns a background thread, updates the in-memory
                        job store, returns immediately with a job_id.
                        Used by the Flask API.
  - `run_job(...)`   -- synchronous core that does the actual work.
                        Used by the CLI directly, and by `start_job`
                        as the thread target.

Both entry points share the same core, so the CLI and the API always
execute pipelines via the SAME code path. There is one source of truth
for pipeline resolution and execution semantics.

The in-memory job store is a simple single-process dict -- fine for a
desktop/local-server tool. Swap for Redis + Celery/RQ if you ever need
multi-process or multi-machine execution.

Execution model (PR1 refactor):
  Build a Dataset tree, then for each Subject build ONE SubjectContext
  and reuse it across all of that subject's acquisitions. Steps that
  produce side outputs (brain_mask, transform, motion_params) make
  them available to later steps operating on sibling acquisitions of
  the same subject -- which is what unblocks multimodal registration
  and DWI tensor fitting with proper sidecar wiring.

  The per-step function contract is unchanged:
      fn(input_path, output_dir, params, context) -> dict

PR6 changes:
  - `skip_recommended` is now a first-class parameter that flows all
    the way from API/CLI -> start_job/run_job -> resolve_pipeline.
    Previously the API accepted it but never propagated it to
    execution.
  - `run_job()` is the synchronous core; `start_job()` wraps it in a
    thread for the Flask API. The CLI calls `run_job()` directly.
  - An optional `progress_callback` lets callers (CLI, future UI)
    receive live progress updates without polling the job store.
"""
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from app.data_model import (
    Dataset,
    Subject,
    SubjectContext,
    build_dataset,
)
from app.detection.classifier import classify_manifest, modality_counts
from app.ingestion.scanner import scan_directory, summarize_manifest
from app.output.structure import initialize_output_tree, subject_processed_dir
from app.pipeline.dag import run_pipeline_for_file
from app.pipeline.dependency_resolver import resolve_pipeline
from app.provenance.tracker import ProvenanceTracker
from app.qc.validators import validate_manifest

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


# --------------------------------------------------------------------- #
# Progress callback signature
# --------------------------------------------------------------------- #

# A progress callback receives the full job status dict on every update.
# This lets the CLI render a live progress bar, the API to update its
# in-memory store, and future UIs to subscribe to updates via websockets.
ProgressCallback = Callable[[Dict[str, Any]], None]


# --------------------------------------------------------------------- #
# In-memory job store (Flask API only)
# --------------------------------------------------------------------- #

def _new_job_record(job_id: str, input_dir: str, output_dir: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "pending",  # pending -> running -> completed | failed
        "input_dir": input_dir,
        "output_dir": output_dir,
        "created_at": time.time(),
        "updated_at": time.time(),
        "progress": {"total_files": 0, "processed_files": 0},
        "result": None,
        "error": None,
    }


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs() -> List[Dict[str, Any]]:
    with _jobs_lock:
        return list(_jobs.values())


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields, updated_at=time.time())


# --------------------------------------------------------------------- #
# Synchronous core -- the single source of truth for pipeline execution.
# Used by BOTH the Flask API (via start_job's background thread) and
# the CLI (directly).
# --------------------------------------------------------------------- #

def run_job(
    job_id: str,
    input_dir: str,
    output_dir: str,
    selected_step_ids: List[str],
    step_params: Dict[str, Dict[str, Any]],
    target_modalities: Optional[List[str]] = None,
    skip_recommended: Optional[List[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Synchronous job execution core.

    Runs the full pipeline:
        1. Scan + classify
        2. QC validate
        3. Build Dataset tree
        4. Resolve pipeline (with skip_recommended applied)
        5. Execute per subject -> SubjectContext -> per acquisition

    Returns the final result dict (also returned via the in-memory job
    store when called via `start_job`).

    Args:
        job_id: unique id for this run (used in provenance log path).
        input_dir: directory to scan.
        output_dir: where to write processed outputs.
        selected_step_ids: the user's chosen steps (pre-resolution).
        step_params: per-step param overrides (e.g. {"brain_extraction":
            {"engine": "synthstrip"}}).
        target_modalities: optional list of modalities to process
            (others are filtered out).
        skip_recommended: list of recommended-dependency step ids to
            suppress. This is now propagated through to resolve_pipeline,
            so the EXECUTED pipeline matches the preview from
            /api/pipeline/resolve. (Previously the API accepted this but
            the engine ignored it.)
        progress_callback: optional callback that receives the current
            job status dict on every progress update. Used by the CLI
            to render live progress; ignored by Flask (which polls the
            in-memory store instead).

    Returns:
        The result dict containing dataset_summary, qc_report, pipeline,
        per_file_results. Raises on hard failure (so the CLI can catch
        and exit non-zero).
    """
    # If a callback is provided, we mirror status into it (and into the
    # in-memory store if a job_id is registered). The store update is
    # a no-op for CLI calls that didn't pre-register a job record.
    def _emit_status(status: str, **extra: Any) -> None:
        if job_id in _jobs:
            _update_job(job_id, status=status, **extra)
        if progress_callback is not None:
            snapshot = _jobs.get(job_id, {}).copy() if job_id in _jobs else {}
            snapshot.update(status=status, **extra)
            progress_callback(snapshot)

    initialize_output_tree(output_dir)
    provenance = ProvenanceTracker(f"{output_dir}/provenance/{job_id}.jsonl")
    provenance.log_event("job_started", job_id=job_id, input_dir=input_dir)

    _emit_status("running")

    # 1. Ingest (flat manifest -- kept for backward-compat with the
    #    QC layer and the dataset_summary shape returned by the API).
    manifest = scan_directory(input_dir)
    manifest = classify_manifest(manifest)
    dataset_summary = summarize_manifest(manifest)
    dataset_summary["modality_counts"] = modality_counts(manifest)

    # 2. QC / validate (operates on the flat manifest -- unchanged).
    qc_report = validate_manifest(manifest)
    failed_filepaths = {
        r["filepath"] for r in qc_report["failed_files"]
    }
    manifest = [m for m in manifest if m["filepath"] not in failed_filepaths]

    if target_modalities:
        manifest = [m for m in manifest if m.get("modality") in target_modalities]

    # 3. Build the Dataset tree from the (filtered) manifest.
    #    This is the layer that lets the SubjectContext be shared across
    #    a subject's acquisitions.
    dataset: Dataset = build_dataset(
        input_dir=input_dir,
        manifest=manifest,
    )

    # Build a flat list of (subject, acquisition) tuples for progress
    # reporting, in deterministic order.
    execution_plan: List[tuple] = []
    for subject in dataset.subjects:
        for acq in subject.acquisitions:
            if not acq.modality:
                continue  # unclassified -- skipped, flagged in report
            execution_plan.append((subject, acq))

    total_files = len(execution_plan)
    _emit_status("running", progress={"total_files": total_files,
                                       "processed_files": 0})

    # 4. Resolve pipeline once (dependency resolution is modality-agnostic
    #    at this stage; per-acquisition execution filters steps by modality).
    #    PR6: skip_recommended is now propagated all the way through.
    #    This is the load-bearing fix -- previously the API accepted
    #    skip_recommended in /api/pipeline/resolve but never passed it
    #    to the executing job.
    resolution = resolve_pipeline(
        selected_step_ids,
        step_params=step_params,
        skip_recommended=skip_recommended,
    )
    ordered_steps = resolution["ordered_pipeline"]

    # 5. Execute per subject (not per file). Within a subject, all
    #    acquisitions share the same SubjectContext, so a step that
    #    produces `brain_mask` on the T1 acquisition makes it available
    #    to a subsequent registration step on the FLAIR acquisition.
    per_file_results: List[Dict[str, Any]] = []
    processed_count = 0

    for subject in dataset.subjects:
        # Build the per-subject context once. This is the load-bearing
        # change: previously each file got `context={}`; now a subject's
        # acquisitions share a context that knows about sibling files,
        # sidecars, and any masks/transforms produced so far.
        subject_context = SubjectContext(subject)

        subject_acquisitions = [
            acq for acq in subject.acquisitions
            if acq.modality  # skip unclassified
        ]

        for acq in subject_acquisitions:
            work_dir = subject_processed_dir(
                output_dir, subject.subject_id, acq.modality,
            )
            try:
                result = run_pipeline_for_file(
                    subject_id=subject.subject_id,
                    modality=acq.modality,
                    input_path=acq.path,
                    ordered_step_ids=ordered_steps,
                    step_params=step_params,
                    work_dir=work_dir,
                    context=subject_context,
                    provenance=provenance,
                )
                per_file_results.append({
                    "subject": subject.subject_id,
                    "modality": acq.modality,
                    "acquisition_id": acq.acquisition_id,
                    "input": acq.path,
                    "final_output": result["final_output"],
                    "status": "success",
                })
            except Exception as exc:  # noqa: BLE001
                per_file_results.append({
                    "subject": subject.subject_id,
                    "modality": acq.modality,
                    "acquisition_id": acq.acquisition_id,
                    "input": acq.path,
                    "status": "failed",
                    "error": str(exc),
                })
                provenance.log_event(
                    "step_error", subject_id=subject.subject_id,
                    modality=acq.modality, acquisition_id=acq.acquisition_id,
                    error=str(exc),
                )

            processed_count += 1
            _emit_status("running", progress={
                "total_files": total_files,
                "processed_files": processed_count,
            })

    provenance.log_event("job_completed", job_id=job_id)

    final_result = {
        "dataset_summary": dataset_summary,
        "qc_report": qc_report,
        "pipeline": resolution,
        "per_file_results": per_file_results,
    }

    _emit_status("completed", result=final_result)
    return final_result


# --------------------------------------------------------------------- #
# Asynchronous wrapper for Flask API.
# --------------------------------------------------------------------- #

def start_job(
    input_dir: str,
    output_dir: str,
    selected_step_ids: List[str],
    step_params: Dict[str, Dict[str, Any]],
    target_modalities: Optional[List[str]] = None,
    skip_recommended: Optional[List[str]] = None,
) -> str:
    """Spawns a background thread that runs `run_job`, and returns
    immediately with a job_id.

    The CLI does NOT use this; it calls `run_job` directly for synchronous
    execution with live progress. The Flask API uses this so the
    /api/jobs endpoint can return 202 Accepted immediately and the
    client polls /api/jobs/<id> for status.

    PR6: skip_recommended is now propagated through to run_job ->
    resolve_pipeline. This was the load-bearing fix the user requested
    -- previously the API accepted skip_recommended in /api/pipeline/
    resolve for PREVIEW but never propagated it to actual EXECUTION.
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = _new_job_record(job_id, input_dir, output_dir)

    thread = threading.Thread(
        target=_run_job_in_thread,
        args=(job_id, input_dir, output_dir, selected_step_ids,
              step_params, target_modalities, skip_recommended),
        daemon=True,
    )
    thread.start()
    return job_id


def _run_job_in_thread(
    job_id: str,
    input_dir: str,
    output_dir: str,
    selected_step_ids: List[str],
    step_params: Dict[str, Dict[str, Any]],
    target_modalities: Optional[List[str]],
    skip_recommended: Optional[List[str]],
) -> None:
    """Thread-target wrapper around run_job that catches and records
    hard failures (instead of letting them propagate into the thread
    and disappear silently)."""
    try:
        run_job(
            job_id=job_id,
            input_dir=input_dir,
            output_dir=output_dir,
            selected_step_ids=selected_step_ids,
            step_params=step_params,
            target_modalities=target_modalities,
            skip_recommended=skip_recommended,
            # No progress_callback needed: run_job updates the
            # in-memory job store directly when job_id is registered.
        )
    except Exception as exc:  # noqa: BLE001
        if job_id in _jobs:
            _update_job(job_id, status="failed",
                        error=f"{exc}\n{traceback.format_exc()}")
