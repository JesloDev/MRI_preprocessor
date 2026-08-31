from app.detection.classifier import classify_manifest, modality_counts
from app.ingestion.scanner import scan_directory, summarize_manifest


def test_scan_finds_all_files(synthetic_dataset):
    manifest = scan_directory(synthetic_dataset["input_dir"])
    assert len(manifest) == 4  # 2 subjects x (T1 + fMRI)


def test_subject_inference(synthetic_dataset):
    manifest = scan_directory(synthetic_dataset["input_dir"])
    subjects = {m["subject"] for m in manifest}
    # BIDS convention: the subject *label* excludes the "sub-" prefix.
    assert subjects == {"001", "002"}


def test_modality_classification(synthetic_dataset):
    manifest = scan_directory(synthetic_dataset["input_dir"])
    manifest = classify_manifest(manifest)
    counts = modality_counts(manifest)
    assert counts.get("T1") == 2
    assert counts.get("fMRI") == 2


def test_summary(synthetic_dataset):
    manifest = scan_directory(synthetic_dataset["input_dir"])
    summary = summarize_manifest(manifest)
    assert summary["num_subjects"] == 2
    assert summary["num_files"] == 4
