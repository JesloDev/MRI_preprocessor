"""
End-to-end smoke test: runs a real (small) structural pipeline through
the DAG executor on synthetic data, using the actual SimpleITK/nibabel
implementations -- not mocks.
"""
import os

from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.dag import run_pipeline_for_file
from app.pipeline.dependency_resolver import resolve_pipeline
from app.provenance.tracker import ProvenanceTracker


def test_structural_pipeline_runs_end_to_end(synthetic_dataset, tmp_path):
    from app.ingestion.scanner import scan_directory
    from app.detection.classifier import classify_manifest

    manifest = classify_manifest(scan_directory(synthetic_dataset["input_dir"]))
    t1_entry = next(m for m in manifest if m["modality"] == "T1")

    resolution = resolve_pipeline(["bias_field_correction", "resampling"])

    work_dir = str(tmp_path / "work")
    provenance = ProvenanceTracker(str(tmp_path / "prov.jsonl"))

    result = run_pipeline_for_file(
        subject_id=t1_entry["subject"],
        modality="T1",
        input_path=t1_entry["filepath"],
        ordered_step_ids=resolution["ordered_pipeline"],
        step_params={"resampling": {"voxel_size": [2.0, 2.0, 2.0]}},
        work_dir=work_dir,
        context={},
        provenance=provenance,
    )

    assert os.path.exists(result["final_output"])

    history = provenance.read_all()
    step_ids_logged = [r["step_id"] for r in history if "step_id" in r]
    assert "reorientation" in step_ids_logged
    assert "bias_field_correction" in step_ids_logged
    assert "resampling" in step_ids_logged
