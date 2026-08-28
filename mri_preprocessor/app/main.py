"""
REST API for the MRI/fMRI Preprocessing Platform.

Endpoints:
  GET  /api/health
  POST /api/scan                 {input_dir}
  GET  /api/capabilities         -> all registered steps
  GET  /api/capabilities/<modality>
  POST /api/pipeline/resolve     {step_ids: [...]}
  POST /api/jobs                 {input_dir, output_dir, step_ids, step_params?, modalities?}
  GET  /api/jobs                 -> list all jobs
  GET  /api/jobs/<job_id>
  GET  /api/jobs/<job_id>/provenance

Run with: python -m app.main
"""
from flask import Flask, jsonify, request

from app.config import HOST, PORT
from app.detection.classifier import classify_manifest, modality_counts
from app.execution.engine import get_job, list_jobs, start_job
from app.ingestion.scanner import scan_directory, summarize_manifest
from app.pipeline.bootstrap import *  # noqa: F401,F403 -- registers all steps
from app.pipeline.dependency_resolver import DependencyError, resolve_pipeline
from app.pipeline.step_registry import registry
from app.provenance.tracker import ProvenanceTracker
from app.qc.validators import validate_manifest


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "registered_steps": len(registry.all_steps())})

    @app.post("/api/scan")
    def scan():
        payload = request.get_json(force=True) or {}
        input_dir = payload.get("input_dir")
        if not input_dir:
            return jsonify({"error": "input_dir is required"}), 400

        try:
            manifest = scan_directory(input_dir)
        except NotADirectoryError as exc:
            return jsonify({"error": str(exc)}), 404

        manifest = classify_manifest(manifest)
        qc_report = validate_manifest(manifest)
        summary = summarize_manifest(manifest)
        summary["modality_counts"] = modality_counts(manifest)

        return jsonify({
            "dataset_summary": summary,
            "qc_report": qc_report,
            "manifest_preview": manifest[:50],  # avoid huge payloads
        })

    @app.get("/api/capabilities")
    def capabilities():
        return jsonify(registry.as_dict())

    @app.get("/api/capabilities/<modality>")
    def capabilities_for_modality(modality: str):
        steps = registry.for_modality(modality)
        return jsonify([s.model_dump() for s in steps])

    @app.post("/api/pipeline/resolve")
    def pipeline_resolve():
        payload = request.get_json(force=True) or {}
        step_ids = payload.get("step_ids", [])
        try:
            resolution = resolve_pipeline(step_ids)
        except DependencyError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(resolution)

    @app.post("/api/jobs")
    def create_job():
        payload = request.get_json(force=True) or {}
        input_dir = payload.get("input_dir")
        output_dir = payload.get("output_dir")
        step_ids = payload.get("step_ids", [])
        step_params = payload.get("step_params", {})
        modalities = payload.get("modalities")  # optional filter

        if not input_dir or not output_dir:
            return jsonify({"error": "input_dir and output_dir are required"}), 400
        if not step_ids:
            return jsonify({"error": "step_ids must be a non-empty list"}), 400

        try:
            resolve_pipeline(step_ids)  # validate before starting the job
        except DependencyError as exc:
            return jsonify({"error": str(exc)}), 400

        job_id = start_job(input_dir, output_dir, step_ids, step_params, modalities)
        return jsonify({"job_id": job_id, "status": "started"}), 202

    @app.get("/api/jobs")
    def jobs_list():
        return jsonify(list_jobs())

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = get_job(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)

    @app.get("/api/jobs/<job_id>/provenance")
    def job_provenance(job_id: str):
        job = get_job(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        tracker = ProvenanceTracker(f"{job['output_dir']}/provenance/{job_id}.jsonl")
        return jsonify(tracker.read_all())

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
