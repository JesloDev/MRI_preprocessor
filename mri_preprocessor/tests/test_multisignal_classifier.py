"""
Tests for the multi-signal classifier (PR2).

Each signal source is tested in isolation, then the MultiSignalClassifier
aggregator is tested end-to-end covering:
  - single-signal agreement (high confidence)
  - multi-signal agreement (still high confidence, more evidence)
  - signal conflict (lower confidence)
  - no signals fired (modality=None, confidence=0.0)
  - manual-review threshold (below 0.5 -> modality=None, signals preserved)
"""
import os

import nibabel as nib
import numpy as np
import pytest

from app.data_model.schema import ClassificationResult, ClassificationSignal
from app.detection.multisignal_classifier import (
    MultiSignalClassifier,
    classify_acquisition,
)
from app.detection.signals import (
    DimensionSignal,
    FilenameSignal,
    HeaderSignal,
    PathSignal,
    SidecarSignal,
    default_signals,
)


# --------------------------------------------------------------------- #
# FilenameSignal
# --------------------------------------------------------------------- #

def test_filename_signal_detects_t1():
    s = FilenameSignal()
    sig = s.evaluate("/data/sub-001_T1.nii.gz", "sub-001_T1.nii.gz", {}, {})
    assert sig is not None
    assert sig.source == "filename"
    assert sig.modality == "T1"
    assert sig.weight == 1.0
    assert "T1" in sig.evidence


def test_filename_signal_detects_fmri_via_bold_keyword():
    s = FilenameSignal()
    sig = s.evaluate("/data/task-rest_bold.nii.gz",
                     "task-rest_bold.nii.gz", {}, {})
    assert sig is not None
    assert sig.modality == "fMRI"


def test_filename_signal_detects_dwi():
    s = FilenameSignal()
    sig = s.evaluate("/data/sub-001_dwi.nii.gz", "sub-001_dwi.nii.gz", {}, {})
    assert sig is not None
    assert sig.modality == "DWI"


def test_filename_signal_returns_none_for_unknown_modality():
    s = FilenameSignal()
    sig = s.evaluate("/data/random_file.nii.gz", "random_file.nii.gz", {}, {})
    assert sig is None


def test_filename_signal_weight_is_configurable():
    s = FilenameSignal(weight=2.5)
    sig = s.evaluate("/data/T1.nii.gz", "T1.nii.gz", {}, {})
    assert sig.weight == 2.5


# --------------------------------------------------------------------- #
# PathSignal
# --------------------------------------------------------------------- #

def test_path_signal_detects_dwi_from_bids_directory():
    s = PathSignal()
    # BIDS-style: subject/dwi/sub-001_dwi.nii.gz
    sig = s.evaluate("/data/sub-001/dwi/sub-001_dwi.nii.gz",
                     "sub-001_dwi.nii.gz", {}, {})
    assert sig is not None
    assert sig.source == "path"
    assert sig.modality == "DWI"


def test_path_signal_detects_fmri_from_func_directory():
    s = PathSignal()
    sig = s.evaluate("/data/sub-001/func/bold.nii.gz",
                     "bold.nii.gz", {}, {})
    assert sig is not None
    assert sig.modality == "fMRI"


def test_path_signal_abstains_for_anat_directory():
    """anat/ could be T1/T2/FLAIR/PD -- path signal abstains."""
    s = PathSignal()
    sig = s.evaluate("/data/sub-001/anat/sub-001_T1.nii.gz",
                     "sub-001_T1.nii.gz", {}, {})
    assert sig is None


def test_path_signal_returns_none_when_no_bids_dirs():
    s = PathSignal()
    sig = s.evaluate("/data/sub-001_T1.nii.gz", "sub-001_T1.nii.gz", {}, {})
    assert sig is None


# --------------------------------------------------------------------- #
# SidecarSignal
# --------------------------------------------------------------------- #

def test_sidecar_signal_detects_dwi_from_bval_bvec():
    s = SidecarSignal()
    sidecars = {
        "bval": "/data/sub-001_dwi.bval",
        "bvec": "/data/sub-001_dwi.bvec",
    }
    sig = s.evaluate("/data/sub-001_dwi.nii.gz",
                     "sub-001_dwi.nii.gz", sidecars, {})
    assert sig is not None
    assert sig.source == "sidecar"
    assert sig.modality == "DWI"
    assert sig.weight == 2.0  # default weight


def test_sidecar_signal_detects_fmri_from_confounds():
    s = SidecarSignal()
    sidecars = {"confounds": "/data/sub-001_bold_confounds.tsv"}
    sig = s.evaluate("/data/sub-001_bold.nii.gz",
                     "sub-001_bold.nii.gz", sidecars, {})
    assert sig is not None
    assert sig.modality == "fMRI"


def test_sidecar_signal_returns_none_for_structural():
    """T1 with no sidecars -> no sidecar signal."""
    s = SidecarSignal()
    sig = s.evaluate("/data/T1.nii.gz", "T1.nii.gz", {}, {})
    assert sig is None


def test_sidecar_signal_only_bval_no_bvec_abstains():
    """A bval without a bvec is suspicious -- we abstain rather than guess."""
    s = SidecarSignal()
    sidecars = {"bval": "/data/sub-001_dwi.bval"}
    sig = s.evaluate("/data/sub-001_dwi.nii.gz",
                     "sub-001_dwi.nii.gz", sidecars, {})
    assert sig is None


def test_sidecar_signal_weight_is_configurable():
    s = SidecarSignal(weight=3.0)
    sidecars = {"bval": "/data/dwi.bval", "bvec": "/data/dwi.bvec"}
    sig = s.evaluate("/data/dwi.nii.gz", "dwi.nii.gz", sidecars, {})
    assert sig.weight == 3.0


# --------------------------------------------------------------------- #
# DimensionSignal / HeaderSignal (PR2 stubs)
# --------------------------------------------------------------------- #

def test_dimension_signal_returns_none_in_pr2():
    """PR2 leaves DimensionSignal as a no-op stub. Future PRs will add
    rules; this test pins the current contract."""
    s = DimensionSignal()
    # 4D volume
    assert s.evaluate("/data/x.nii.gz", "x.nii.gz", {},
                      {"ndim": 4, "shape": [16, 16, 16, 50]}) is None
    # 3D volume
    assert s.evaluate("/data/x.nii.gz", "x.nii.gz", {},
                      {"ndim": 3, "shape": [16, 16, 16]}) is None


def test_header_signal_returns_none_in_pr2():
    s = HeaderSignal()
    assert s.evaluate("/data/x.nii.gz", "x.nii.gz", {},
                      {"shape": [16, 16, 16], "voxel_size": [1.0, 1.0, 1.0]}) is None


def test_default_signals_includes_all_five_sources():
    """Default signal set must include all 5 signal sources."""
    signals = default_signals()
    names = [s.name for s in signals]
    assert "filename" in names
    assert "path" in names
    assert "sidecar" in names
    assert "dimension" in names
    assert "header" in names


# --------------------------------------------------------------------- #
# MultiSignalClassifier -- confidence scoring
# --------------------------------------------------------------------- #

def test_classifier_single_filename_signal_gives_full_confidence():
    """T1 with no sidecars, no BIDS dir -> only FilenameSignal fires ->
    confidence = 1.0."""
    clf = MultiSignalClassifier()
    result = clf.classify(
        filepath="/data/sub-001_T1.nii.gz",
        filename="sub-001_T1.nii.gz",
        sidecars={},
        metadata={},
    )
    assert result.modality == "T1"
    assert result.confidence == 1.0
    assert len(result.signals) == 1
    assert result.signals[0].source == "filename"
    assert result.alternatives == []


def test_classifier_multiple_agreeing_signals_increase_evidence():
    """DWI with bval+bvec sidecars AND filename match -> both signals agree
    on DWI. Confidence still 1.0 (single modality voted), but signals list
    has 2 entries."""
    clf = MultiSignalClassifier()
    result = clf.classify(
        filepath="/data/sub-001_dwi.nii.gz",
        filename="sub-001_dwi.nii.gz",
        sidecars={"bval": "/data/sub-001_dwi.bval",
                  "bvec": "/data/sub-001_dwi.bvec"},
        metadata={},
    )
    assert result.modality == "DWI"
    assert result.confidence == 1.0
    sources = {s.source for s in result.signals}
    assert sources == {"filename", "sidecar"}


def test_classifier_conflicting_signals_lower_confidence():
    """A file named 'T1' but with a bval+bvec sidecar pair: filename says
    T1, sidecar says DWI. Top modality is DWI (sidecar weight 2.0 vs
    filename weight 1.0). Confidence = 2.0/(2.0+1.0) = 0.667. Above
    threshold so modality = DWI, with T1 as an alternative."""
    clf = MultiSignalClassifier()
    result = clf.classify(
        filepath="/data/T1.nii.gz",
        filename="T1.nii.gz",
        sidecars={"bval": "/data/T1.bval", "bvec": "/data/T1.bvec"},
        metadata={},
    )
    assert result.modality == "DWI"
    assert result.confidence == pytest.approx(2.0 / 3.0, abs=0.01)
    assert len(result.alternatives) == 1
    assert result.alternatives[0][0] == "T1"
    assert result.alternatives[0][1] == pytest.approx(1.0)


def test_classifier_no_signals_returns_none_modality():
    """A file with no recognizable patterns and no sidecars -> modality=None,
    confidence=0.0."""
    clf = MultiSignalClassifier()
    result = clf.classify(
        filepath="/data/random_random.nii.gz",
        filename="random_random.nii.gz",
        sidecars={},
        metadata={},
    )
    assert result.modality is None
    assert result.confidence == 0.0
    assert result.signals == []
    assert result.alternatives == []


def test_classifier_below_threshold_returns_none_modality_but_keeps_signals():
    """When confidence is below threshold, modality=None (flagged for manual
    review) but signals + alternatives are preserved so the UI can show
    why. Construct a 50/50 split between two modalities to force this."""
    # Custom signals that vote equally for two different modalities.
    # This produces confidence = 0.5 which is AT the threshold, so we need
    # a tiebreaker scenario. Easier: use a stricter threshold (0.7) and
    # the conflicting T1/DWI case (confidence ~0.667).
    clf = MultiSignalClassifier(confidence_threshold=0.7)
    result = clf.classify(
        filepath="/data/T1.nii.gz",
        filename="T1.nii.gz",
        sidecars={"bval": "/data/T1.bval", "bvec": "/data/T1.bvec"},
        metadata={},
    )
    # DWI got 2.0, T1 got 1.0. Top=DWI, confidence = 2.0/3.0 ~ 0.667
    # which is below the 0.7 threshold -> flagged for manual review.
    assert result.confidence == pytest.approx(2.0 / 3.0, abs=0.01)
    assert result.modality is None  # below threshold
    assert len(result.signals) == 2
    assert len(result.alternatives) == 1  # T1 is recorded as an alternative
    # Even though modality is None, the underlying votes are preserved
    modalities_voted = {s.modality for s in result.signals}
    assert modalities_voted == {"T1", "DWI"}


def test_classifier_default_threshold_is_0_5():
    clf = MultiSignalClassifier()
    assert clf.confidence_threshold == 0.5


def test_classifier_custom_signal_set():
    """The classifier can be configured with a custom signal set, e.g.
    only filename + sidecar (skipping path/dimension/header)."""
    clf = MultiSignalClassifier(
        signals=[FilenameSignal(weight=2.0), SidecarSignal(weight=3.0)],
    )
    # DWI with sidecars: filename 2.0 + sidecar 3.0 = 5.0 for DWI
    result = clf.classify(
        filepath="/data/sub-001_dwi.nii.gz",
        filename="sub-001_dwi.nii.gz",
        sidecars={"bval": "/data/sub-001_dwi.bval",
                  "bvec": "/data/sub-001_dwi.bvec"},
        metadata={},
    )
    assert result.modality == "DWI"
    assert result.confidence == 1.0
    # Only 2 signal sources configured -> only 2 signals fired
    sources = {s.source for s in result.signals}
    assert sources == {"filename", "sidecar"}


# --------------------------------------------------------------------- #
# Convenience function
# --------------------------------------------------------------------- #

def test_classify_acquisition_convenience_function():
    """The module-level classify_acquisition uses the default singleton."""
    result = classify_acquisition(
        filepath="/data/T1.nii.gz",
        filename="T1.nii.gz",
        sidecars={},
        metadata={},
    )
    assert isinstance(result, ClassificationResult)
    assert result.modality == "T1"


# --------------------------------------------------------------------- #
# Integration with the data model builder
# --------------------------------------------------------------------- #

def test_builder_uses_multisignal_classifier_for_dwi_with_sidecars(tmp_path):
    """build_dataset should now produce a ClassificationResult with multiple
    signals when DWI sidecars are present (filename + sidecar both vote)."""
    from app.data_model import build_dataset

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
    assert dwi_acq.classification is not None
    assert dwi_acq.classification.modality == "DWI"
    assert dwi_acq.classification.confidence == 1.0
    # Both filename AND sidecar signals fired
    sources = {s.source for s in dwi_acq.classification.signals}
    assert "filename" in sources
    assert "sidecar" in sources


def test_builder_synthetic_dataset_still_classifies_correctly(synthetic_dataset):
    """The synthetic dataset (T1 + fMRI, no sidecars, no BIDS dirs) should
    still classify with confidence 1.0 via filename-only signal -- same as
    PR1, just with the multi-signal classifier doing the work."""
    from app.data_model import build_dataset

    ds = build_dataset(synthetic_dataset["input_dir"])
    subject = ds.get_subject("001")

    t1_acq = subject.get_primary_acquisition("T1")
    assert t1_acq.classification.modality == "T1"
    assert t1_acq.classification.confidence == 1.0
    # Only filename fires (no BIDS dirs, no sidecars)
    sources = {s.source for s in t1_acq.classification.signals}
    assert sources == {"filename"}

    fmri_acq = subject.get_primary_acquisition("fMRI")
    assert fmri_acq.classification.modality == "fMRI"
    assert fmri_acq.classification.confidence == 1.0


def test_builder_path_signal_fires_for_bids_layout(tmp_path):
    """A BIDS-layout DWI in /dwi/ directory gets BOTH path and filename
    signals voting for DWI -- more evidence, still confidence 1.0."""
    from app.data_model import build_dataset

    sub = tmp_path / "sub-001" / "dwi"
    sub.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8, 4).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "sub-001_dwi.nii.gz"),
    )

    ds = build_dataset(str(tmp_path))
    dwi_acq = ds.get_subject("001").get_primary_acquisition("DWI")
    assert dwi_acq is not None
    sources = {s.source for s in dwi_acq.classification.signals}
    # No sidecar here (bval/bvec not created), but path AND filename both fire
    assert "filename" in sources
    assert "path" in sources
    assert "sidecar" not in sources  # no bval/bvec


def test_unclassified_file_marked_for_manual_review(tmp_path):
    """A file with no recognizable patterns should be classified as None
    (flagged for manual review), not crash."""
    from app.data_model import build_dataset

    sub = tmp_path / "sub-001"
    sub.mkdir()
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "random_unknown_scan.nii.gz"),
    )

    ds = build_dataset(str(tmp_path))
    subject = ds.get_subject("001")
    # Acquisition should exist with modality=None
    assert len(subject.acquisitions) == 1
    acq = subject.acquisitions[0]
    assert acq.modality is None
    assert acq.classification is not None
    assert acq.classification.modality is None
    assert acq.classification.confidence == 0.0
    # acquisition_id should be UNK_1 since modality is unknown
    assert acq.acquisition_id == "UNK_1"
