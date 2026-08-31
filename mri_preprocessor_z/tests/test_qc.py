from app.ingestion.scanner import scan_directory
from app.qc.validators import validate_manifest


def test_valid_synthetic_files_pass_qc(synthetic_dataset):
    manifest = scan_directory(synthetic_dataset["input_dir"])
    report = validate_manifest(manifest)
    assert report["valid_count"] == 4
    assert report["failed_count"] == 0
