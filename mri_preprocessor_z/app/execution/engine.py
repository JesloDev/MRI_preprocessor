"""
Job execution engine: orchestrates a full run (scan -> classify ->
validate -> resolve pipeline -> execute per subject/modality -> report)
in a background thread, and exposes job status for the REST API to poll.

This is a simple in-memory, single-process job store -- fine for a
desktop/local-server tool. Swap for Redis + Celery/RQ if you ever need
multi-process or multi-machine execution.

Execution model (PR1 refactor):
  Previously each file flowed through the pipeline in isolation, with
  `context={}` passed in. Now we build a Dataset tree first, then for
  each Subject we build ONE SubjectContext and reuse it across all of
  that subject's acquisitions. This means steps that produce side
  outputs (brain_mask, transform, motion_params) make them available
  to later steps operating on sibling acquisitions of the same subject
  -- which is what unblocks multimodal registration and DWI tensor
  fitting with proper sidecar wiring.

  The per-step function contract is unchanged:
      fn(input_path, output_dir, params, context) -> dict

  SubjectContext quacks like the dict that `context` used to be, so
  existing step code (which does `context.get("bval_path")`,
  `context.update({...})`, etc.) keeps working without modification.
"""
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

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


def start_job(
    input_dir: str,
    output_dir: str,
    selected_step_ids: List[str],
    step_params: Dict[str, Dict[str, Any]],
    target_modalities: Optional[List[str]] = None,
) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = _new_job_record(job_id, input_dir, output_dir)

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, input_dir, output_dir, selected_step_ids, step_params, target_modalities),
        daemon=True,
    )
    thread.start()
    return job_id


def _run_job(
    job_id: str,
    input_dir: str,
    output_dir: str,
    selected_step_ids: List[str],
    step_params: Dict[str, Dict[str, Any]],
    target_modalities: Optional[List[str]],
) -> None:
    try:
        _update_job(job_id, status="running")
        initialize_output_tree(output_dir)
        provenance = ProvenanceTracker(f"{output_dir}/provenance/{job_id}.jsonl")
        provenance.log_event("job_started", job_id=job_id, input_dir=input_dir)

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
        #    This is the new layer: instead of iterating the flat manifest,
        #    we group acquisitions by subject so the SubjectContext can
        #    be shared across a subject's acquisitions.
        dataset: Dataset = build_dataset(
            input_dir=input_dir,
            manifest=manifest,
            # SubjectContext reads sidecars from the primary acquisition;
            # if a modality was filtered out above we want its acquisitions
            # to disappear from the Dataset entirely.
        )

        # Build a flat list of (subject, acquisition) tuples for progress
        # reporting, in deterministic order.
        execution_plan: List[tuple] = []
        for subject in dataset.subjects:
            for acq in subject.acquisitions:
                if not acq.modality:
                    continue  # unclassified -- skipped, flagged in report
                execution_plan.append((subject, acq))

        _update_job(job_id, progress={
            "total_files": len(execution_plan),
            "processed_files": 0,
        })

        # 4. Resolve pipeline once (dependency resolution is modality-agnostic
        #    at this stage; per-acquisition execution filters steps by modality).
        #    Pass step_params so conditional dependencies can be evaluated
        #    against the requesting step's effective configuration.
        resolution = resolve_pipeline(selected_step_ids, step_params=step_params)
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
                _update_job(job_id, progress={
                    "total_files": len(execution_plan),
                    "processed_files": processed_count,
                })

        provenance.log_event("job_completed", job_id=job_id)

        _update_job(
            job_id,
            status="completed",
            result={
                "dataset_summary": dataset_summary,
                "qc_report": qc_report,
                "pipeline": resolution,
                "per_file_results": per_file_results,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _update_job(job_id, status="failed", error=f"{exc}\n{traceback.format_exc()}")
