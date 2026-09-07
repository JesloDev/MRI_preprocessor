"""
Tests for the new data_model layer (Dataset / Subject / Acquisition /
ClassificationResult / SubjectContext).

These tests exercise the data model in isolation -- no pipeline steps
are run here. The integration test that confirms SubjectContext flows
correctly through the existing pipeline lives in
test_subject_aware_processing.py.
"""
import os

import nibabel as nib
import numpy as np
import pytest

from app.data_model import (
    Acquisition,
    AutoSubjectStrategy,
    BIDSSubjectStrategy,
    ClassificationResult,
    ClassificationSignal,
    Dataset,
    DirectoryDepthStrategy,
    RegexStrategy,
    Subject,
    SubjectContext,
    build_dataset,
    discover_sidecars,
)


# --------------------------------------------------------------------- #
# Sidecar discovery
# --------------------------------------------------------------------- #

def test_discover_sidecars_finds_bval_and_bvec(tmp_path):
    """bval + bvec files next to a dwi.nii.gz are discovered."""
    nii = tmp_path / "sub-001_dwi.nii.gz"
    nii.write_bytes(b"")  # empty file is enough; we never load it here
    (tmp_path / "sub-001_dwi.bval").write_text("0 1000\n")
    (tmp_path / "sub-001_dwi.bvec").write_text("0 1 0\n0 0 0\n0 0 0\n")
    (tmp_path / "sub-001_dwi.json").write_text("{}")

    sidecars = discover_sidecars(str(nii))
    assert sidecars["bval"].endswith("sub-001_dwi.bval")
    assert sidecars["bvec"].endswith("sub-001_dwi.bvec")
    assert sidecars["json"].endswith("sub-001_dwi.json")


def test_discover_sidecars_empty_for_structural(tmp_path):
    """A T1 with no sidecars returns an empty dict (no exception)."""
    nii = tmp_path / "sub-001_T1.nii.gz"
    nii.write_bytes(b"")
    assert discover_sidecars(str(nii)) == {}


def test_discover_sidecars_finds_confounds(tmp_path):
    """Confounds tsv files following common naming are discovered.

    discover_sidecars uses (NIfTI stem) + suffix, so for a NIfTI named
    'sub-001_task-rest_bold.nii.gz' the confounds file must be
    'sub-001_task-rest_bold_confounds.tsv' (matching BIDS convention).
    """
    nii = tmp_path / "sub-001_task-rest_bold.nii.gz"
    nii.write_bytes(b"")
    (tmp_path / "sub-001_task-rest_bold_confounds.tsv").write_text("x,y\n1,2\n")

    sidecars = discover_sidecars(str(nii))
    assert "confounds" in sidecars
    assert sidecars["confounds"].endswith("_bold_confounds.tsv")


# --------------------------------------------------------------------- #
# Subject inference strategies
# --------------------------------------------------------------------- #

def test_bids_subject_strategy_extracts_label():
    s = BIDSSubjectStrategy()
    assert s.infer("/root", "/root/sub-001/anat/sub-001_T1.nii.gz") == "001"
    assert s.infer("/root", "/root/sub-ABC123/dwi/sub-ABC123_dwi.nii.gz") == "ABC123"


def test_directory_depth_strategy_handles_nested_layout():
    """For Root/ScanFolder/Subject/T1.nii.gz, depth=2 picks 'Subject'."""
    s = DirectoryDepthStrategy(depth=2)
    assert s.infer("/root", "/root/ScanFolder/SubA/T1.nii.gz") == "SubA"
    # depth=1 (default) picks the first dir under root
    s1 = DirectoryDepthStrategy(depth=1)
    assert s1.infer("/root", "/root/ScanFolder/SubA/T1.nii.gz") == "ScanFolder"


def test_regex_strategy_with_named_group():
    s = RegexStrategy(r"Scan_(?P<subject>\w+)/")
    assert s.infer("/root", "/root/Scan_XYZ123/dwi.nii.gz") == "XYZ123"


def test_auto_strategy_preserves_existing_scanner_behaviour(synthetic_dataset):
    """AutoSubjectStrategy matches the old _infer_subject_id heuristic,
    so existing tests that assert specific subject ids keep passing."""
    s = AutoSubjectStrategy()
    # The conftest creates sub-001/ and sub-002/ -- Auto should produce '001'/'002'.
    assert s.infer(synthetic_dataset["input_dir"],
                   os.path.join(synthetic_dataset["input_dir"], "sub-001", "sub-001_T1.nii.gz")) == "001"


# --------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------- #

def test_build_dataset_groups_files_by_subject(synthetic_dataset):
    """2 subjects x (T1 + fMRI) -> 2 subjects, each with 2 acquisitions."""
    ds = build_dataset(synthetic_dataset["input_dir"])

    assert isinstance(ds, Dataset)
    assert len(ds.subjects) == 2
    assert set(ds.subject_ids) == {"001", "002"}
    # 2 acquisitions per subject (T1 + fMRI)
    assert all(len(s.acquisitions) == 2 for s in ds.subjects)
    # Total of 4 acquisitions across the dataset
    assert ds.acquisition_count() == 4


def test_build_dataset_assigns_per_modality_acquisition_ids(synthetic_dataset):
    """acquisition_id is per-modality: T1_1, fMRI_1 (not T1_1, fMRI_2)."""
    ds = build_dataset(synthetic_dataset["input_dir"])
    subject = ds.get_subject("001")
    assert subject is not None

    ids = sorted(a.acquisition_id for a in subject.acquisitions)
    # One T1 and one fMRI -> T1_1 and fMRI_1
    assert "T1_1" in ids
    assert "fMRI_1" in ids


def test_build_dataset_populates_modality_via_classifier(synthetic_dataset):
    """The builder runs the existing filename classifier and stores
    ClassificationResult on each acquisition."""
    ds = build_dataset(synthetic_dataset["input_dir"])
    subject = ds.get_subject("001")

    modalities = sorted(a.modality for a in subject.acquisitions)
    assert modalities == ["T1", "fMRI"]

    t1_acq = subject.get_primary_acquisition("T1")
    assert t1_acq is not None
    assert t1_acq.classification is not None
    assert t1_acq.classification.modality == "T1"
    assert t1_acq.classification.confidence == 1.0  # PR1: filename match -> 1.0
    assert len(t1_acq.classification.signals) == 1
    assert t1_acq.classification.signals[0].source == "filename"


def test_build_dataset_reads_header_metadata(synthetic_dataset):
    """Lightweight header metadata (shape, ndim, voxel_size) is populated."""
    ds = build_dataset(synthetic_dataset["input_dir"])
    t1_acq = ds.get_subject("001").get_primary_acquisition("T1")

    assert "shape" in t1_acq.metadata
    assert t1_acq.metadata["shape"] == [16, 16, 16]  # matches conftest fixture
    assert t1_acq.metadata["ndim"] == 3

    fmri_acq = ds.get_subject("001").get_primary_acquisition("fMRI")
    assert fmri_acq.metadata["ndim"] == 4
    assert fmri_acq.metadata["shape"] == [16, 16, 16, 10]


def test_build_dataset_discovers_sidecars_for_dwi(tmp_path):
    """A DWI acquisition with bval/bvec sidecars has them populated."""
    # Create a minimal DWI dataset
    sub = tmp_path / "sub-001" / "dwi"
    sub.mkdir(parents=True)
    dwi_data = np.random.rand(8, 8, 8, 4).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(dwi_data, affine=np.eye(4)),
             str(sub / "sub-001_dwi.nii.gz"))
    (sub / "sub-001_dwi.bval").write_text("0 1000 1000 1000\n")
    (sub / "sub-001_dwi.bvec").write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    ds = build_dataset(str(tmp_path))
    dwi_acq = ds.get_subject("001").get_primary_acquisition("DWI")

    assert dwi_acq is not None
    assert "bval" in dwi_acq.associated_files
    assert "bvec" in dwi_acq.associated_files
    assert dwi_acq.associated_files["bval"].endswith("sub-001_dwi.bval")


def test_build_dataset_supports_multiple_acquisitions_of_same_modality(tmp_path):
    """A subject with two T1s gets T1_1 and T1_2 acquisition ids."""
    sub = tmp_path / "sub-001"
    sub.mkdir()
    for n in (1, 2):
        nib.save(
            nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                            affine=np.eye(4)),
            str(sub / f"sub-001_run-{n}_T1.nii.gz"),
        )

    ds = build_dataset(str(tmp_path))
    subject = ds.get_subject("001")
    t1_acqs = subject.get_acquisitions("T1")

    assert len(t1_acqs) == 2
    ids = sorted(a.acquisition_id for a in t1_acqs)
    assert ids == ["T1_1", "T1_2"]
    # primary = first
    assert subject.get_primary_acquisition("T1").acquisition_id == "T1_1"


def test_build_dataset_with_directory_depth_strategy(tmp_path):
    """The builder accepts a subject_strategy that overrides the default
    Auto behaviour -- this is what handles the
    Root/ScanFolder/Subject/T1.nii.gz layout that the old heuristic
    misclassified."""
    # Build Root/ScanFolder/SubA/...
    nested = tmp_path / "ScanFolder" / "SubA"
    nested.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(nested / "T1.nii.gz"),
    )

    # With Auto (default), the subject would be 'ScanFolder' (wrong).
    ds_auto = build_dataset(str(tmp_path))
    assert ds_auto.subject_ids == ["ScanFolder"]

    # With DirectoryDepth(depth=2), the subject is correctly 'SubA'.
    ds_nested = build_dataset(str(tmp_path),
                              subject_strategy=DirectoryDepthStrategy(depth=2))
    assert ds_nested.subject_ids == ["SubA"]


# --------------------------------------------------------------------- #
# Subject helpers
# --------------------------------------------------------------------- #

def test_subject_helpers_return_empty_for_missing_modality(synthetic_dataset):
    ds = build_dataset(synthetic_dataset["input_dir"])
    subject = ds.get_subject("001")
    assert subject.get_acquisitions("DWI") == []
    assert subject.get_primary_acquisition("DWI") is None
    assert "DWI" not in subject.modalities()


def test_subject_modalities_lists_distinct_modalities(synthetic_dataset):
    ds = build_dataset(synthetic_dataset["input_dir"])
    subject = ds.get_subject("001")
    assert set(subject.modalities()) == {"T1", "fMRI"}
