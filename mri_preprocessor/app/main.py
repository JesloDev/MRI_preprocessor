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
        return jsonify({
            "status": "ok",
            "registered_steps": len(registry.all_steps()),
            "maturity_counts": registry.maturity_counts(),
        })

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
        # Optional query param: ?maturity=experimental to filter.
        # Multiple values supported: ?maturity=experimental&maturity=reference
        maturity_filter = request.args.getlist("maturity")
        all_steps = registry.all_steps()
        if maturity_filter:
            # Filter steps by the requested maturity levels. Steps matching
            # ANY of the requested maturity levels are included.
            maturity_set = set(maturity_filter)
            filtered = {
                s.id: s.model_dump()
                for s in all_steps
                if s.maturity in maturity_set
            }
            return jsonify(filtered)
        return jsonify({s.id: s.model_dump() for s in all_steps})

    @app.get("/api/capabilities/summary")
    def capabilities_summary():
        """High-level capability summary: counts by maturity level.

        Useful for dashboards that want to render "X steps total,
        Y production-ready, Z experimental" without fetching the full
        step list.
        """
        all_steps = registry.all_steps()
        return jsonify({
            "total_steps": len(all_steps),
            "maturity_counts": registry.maturity_counts(),
            "modalities_covered": sorted({
                m for s in all_steps for m in s.modalities
            }),
        })

    @app.get("/api/capabilities/<modality>")
    def capabilities_for_modality(modality: str):
        steps = registry.for_modality(modality)
        return jsonify([s.model_dump() for s in steps])

    @app.get("/api/engines/<step_id>")
    def engines_for_step(step_id: str):
        """Lists the available engine adapters for a given step.

        Returns 404 if the step doesn't exist, or 400 if the step doesn't
        support engine selection (i.e. it has no 'engine' param in its
        default_params -- meaning it's a single-implementation step like
        reorientation or resampling).

        PR5 supports engines for three step families:
            brain_extraction   -> otsu, synthstrip (stub), hdbet (stub)
            fmri_motion_correction -> simpleitk, mcflirt (stub)
            dwi_motion_eddy_correction -> dipy, fsl_eddy (stub)

        The UI uses this endpoint to render the engine-selection dropdown
        on step configuration panels.
        """
        try:
            step = registry.get(step_id)
        except KeyError:
            return jsonify({"error": f"Unknown step: {step_id}"}), 404

        # Engine family lookup. Each dispatcher step has a known
        # engine registry; this mapping is the single source of truth
        # for "which step supports engine selection."
        from app.engines import (
            BRAIN_EXTRACTION_ENGINES,
            FMRI_MOTION_ENGINES,
            DWI_CORRECTION_ENGINES,
        )
        engine_registries = {
            "brain_extraction":            BRAIN_EXTRACTION_ENGINES,
            "fmri_motion_correction":     FMRI_MOTION_ENGINES,
            "dwi_motion_eddy_correction": DWI_CORRECTION_ENGINES,
        }

        if step_id not in engine_registries:
            return jsonify({
                "step_id": step_id,
                "supports_engine_selection": False,
                "message": (
                    f"Step '{step_id}' does not support engine selection. "
                    "It has a single implementation."
                ),
            }), 200

        engine_registry = engine_registries[step_id]
        default_engine = engine_registry.get_default()

        return jsonify({
            "step_id": step_id,
            "supports_engine_selection": True,
            "default_engine": default_engine,
            "current_engine": step.default_params.get("engine", default_engine),
            "available_engines": engine_registry.available_engines(),
        })

    @app.post("/api/pipeline/resolve")
    def pipeline_resolve():
        payload = request.get_json(force=True) or {}
        step_ids = payload.get("step_ids", [])
        step_params = payload.get("step_params", {})  # for conditional dep eval
        skip_recommended = payload.get("skip_recommended", [])  # list of step ids
        try:
            resolution = resolve_pipeline(
                step_ids,
                step_params=step_params,
                skip_recommended=skip_recommended,
            )
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
        skip_recommended = payload.get("skip_recommended", [])

        if not input_dir or not output_dir:
            return jsonify({"error": "input_dir and output_dir are required"}), 400
        if not step_ids:
            return jsonify({"error": "step_ids must be a non-empty list"}), 400

        try:
            # Validate the resolved pipeline before starting the job.
            # step_params is required so conditional dependencies can be
            # evaluated against the requesting step's effective config.
            resolve_pipeline(
                step_ids,
                step_params=step_params,
                skip_recommended=skip_recommended,
            )
        except DependencyError as exc:
            return jsonify({"error": str(exc)}), 400

        # Note: skip_recommended is now propagated through start_job ->
        # run_job -> resolve_pipeline. PR6 closes the loop so the
        # EXECUTED pipeline matches the PREVIEW from
        # /api/pipeline/resolve.
        job_id = start_job(
            input_dir=input_dir,
            output_dir=output_dir,
            selected_step_ids=step_ids,
            step_params=step_params,
            target_modalities=modalities,
            skip_recommended=skip_recommended,
        )
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
