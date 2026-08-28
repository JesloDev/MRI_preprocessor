"""
Job execution engine: orchestrates a full run (scan -> classify ->
validate -> resolve pipeline -> execute per subject/modality -> report)
in a background thread, and exposes job status for the REST API to poll.

This is a simple in-memory, single-process job store -- fine for a
desktop/local-server tool. Swap for Redis + Celery/RQ if you ever need
multi-process or multi-machine execution.
"""
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

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

        # 1. Ingest
        manifest = scan_directory(input_dir)
        manifest = classify_manifest(manifest)
        dataset_summary = summarize_manifest(manifest)
        dataset_summary["modality_counts"] = modality_counts(manifest)

        # 2. QC / validate
        qc_report = validate_manifest(manifest)
        valid_filepaths = {
            r["filepath"] for r in qc_report["failed_files"]
        }
        manifest = [m for m in manifest if m["filepath"] not in valid_filepaths]

        if target_modalities:
            manifest = [m for m in manifest if m.get("modality") in target_modalities]

        _update_job(job_id, progress={"total_files": len(manifest), "processed_files": 0})

        # 3. Resolve pipeline once (dependency resolution is modality-agnostic
        #    at this stage; per-file execution filters steps by modality)
        resolution = resolve_pipeline(selected_step_ids)
        ordered_steps = resolution["ordered_pipeline"]

        # 4. Execute per file
        per_file_results = []
        for i, entry in enumerate(manifest):
            modality = entry.get("modality")
            if not modality:
                continue  # unclassified files are skipped, flagged in report

            work_dir = subject_processed_dir(output_dir, entry["subject"], modality)
            try:
                result = run_pipeline_for_file(
                    subject_id=entry["subject"],
                    modality=modality,
                    input_path=entry["filepath"],
                    ordered_step_ids=ordered_steps,
                    step_params=step_params,
                    work_dir=work_dir,
                    context={},
                    provenance=provenance,
                )
                per_file_results.append({
                    "subject": entry["subject"],
                    "modality": modality,
                    "input": entry["filepath"],
                    "final_output": result["final_output"],
                    "status": "success",
                })
            except Exception as exc:  # noqa: BLE001
                per_file_results.append({
                    "subject": entry["subject"],
                    "modality": modality,
                    "input": entry["filepath"],
                    "status": "failed",
                    "error": str(exc),
                })
                provenance.log_event(
                    "step_error", subject_id=entry["subject"],
                    modality=modality, error=str(exc),
                )

            _update_job(job_id, progress={
                "total_files": len(manifest), "processed_files": i + 1,
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
