"""
Tests for SubjectContext -- the seam between the dataset-aware engine
and the existing step contract (fn(input_path, output_dir, params, context)).

These tests confirm two things:

1. SubjectContext quacks like a dict. Existing step code that does
   `context.get("bval_path")`, `context.update({...})`, `context["x"] = y`,
   `if "key" in context: ...` works unchanged.

2. SubjectContext exposes structured data. New code that wants to be
   explicit can use `context.get_acquisitions("DWI")`,
   `context.get_primary_acquisition("T1")`, `context.get_reference_image("T1")`.
"""
import os

import nibabel as nib
import numpy as np

from app.data_model import (
    Acquisition,
    Subject,
    SubjectContext,
    build_dataset,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _build_subject_with(t1=True, dwi=True, fmri=True):
    """Builds a Subject object directly without going through the scanner."""
    acqs = []
    if t1:
        acqs.append(Acquisition(
            acquisition_id="T1_1", path="/data/sub-001_T1.nii.gz",
            filename="sub-001_T1.nii.gz", modality="T1",
            associated_files={},
            metadata={"shape": [16, 16, 16], "ndim": 3},
        ))
    if dwi:
        acqs.append(Acquisition(
            acquisition_id="DWI_1", path="/data/sub-001_dwi.nii.gz",
            filename="sub-001_dwi.nii.gz", modality="DWI",
            associated_files={
                "bval": "/data/sub-001_dwi.bval",
                "bvec": "/data/sub-001_dwi.bvec",
            },
            metadata={"shape": [16, 16, 16, 4], "ndim": 4},
        ))
    if fmri:
        acqs.append(Acquisition(
            acquisition_id="fMRI_1", path="/data/sub-001_bold.nii.gz",
            filename="sub-001_bold.nii.gz", modality="fMRI",
            associated_files={
                "confounds": "/data/sub-001_confounds.tsv",
            },
            metadata={"shape": [16, 16, 16, 50], "ndim": 4},
        ))
    return Subject(subject_id="001", acquisitions=acqs)


# --------------------------------------------------------------------- #
# Dict compatibility (the load-bearing backward-compat surface)
# --------------------------------------------------------------------- #

def test_context_get_returns_subject_id():
    s = _build_subject_with()
    ctx = SubjectContext(s)
    assert ctx.get("subject_id") == "001"


def test_context_get_returns_none_for_missing_key():
    ctx = SubjectContext(_build_subject_with())
    assert ctx.get("nonexistent_key") is None
    assert ctx.get("nonexistent_key", "default") == "default"


def test_context_getitem_setitem_works():
    ctx = SubjectContext(_build_subject_with())
    ctx["custom_key"] = "custom_value"
    assert ctx["custom_key"] == "custom_value"


def test_context_contains_works():
    ctx = SubjectContext(_build_subject_with())
    assert "subject_id" in ctx
    assert "bval_path" in ctx
    assert "nonexistent" not in ctx


def test_context_update_merges_dict():
    """dag.py calls `context.update(result.get("outputs", {}))` after each step.
    SubjectContext must accept this without breaking."""
    ctx = SubjectContext(_build_subject_with())
    ctx.update({"brain_mask": "/path/to/mask.nii.gz",
                "transform": "/path/to/transform.tfm"})
    assert ctx.get("brain_mask") == "/path/to/mask.nii.gz"
    assert ctx.get("transform") == "/path/to/transform.tfm"


def test_context_update_kwargs_form():
    """SubjectContext.update supports kwargs form (rarely used but Pythonic)."""
    ctx = SubjectContext(_build_subject_with())
    ctx.update(custom_key="custom_value")
    assert ctx.get("custom_key") == "custom_value"


def test_context_keys_items_values_iterate():
    """Standard dict iteration methods work."""
    ctx = SubjectContext(_build_subject_with())
    keys = list(ctx.keys())
    assert "subject_id" in keys
    assert "bval_path" in keys

    items = dict(ctx.items())
    assert items["subject_id"] == "001"


# --------------------------------------------------------------------- #
# Flat key surfacing (the load-bearing forward-compat surface)
# --------------------------------------------------------------------- #

def test_context_surfaces_sidecar_flat_keys():
    """The primary DWI's bval/bvec are surfaced as flat keys, so existing
    step code that does `context.get("bval_path")` keeps working without
    modification."""
    ctx = SubjectContext(_build_subject_with())
    assert ctx.get("bval_path") == "/data/sub-001_dwi.bval"
    assert ctx.get("bvec_path") == "/data/sub-001_dwi.bvec"
    assert ctx.get("confounds_path") == "/data/sub-001_confounds.tsv"


def test_context_surfaces_modality_path_keys():
    """Convenience flat keys for primary acquisition paths."""
    ctx = SubjectContext(_build_subject_with())
    assert ctx.get("t1_path") == "/data/sub-001_T1.nii.gz"
    assert ctx.get("dwi_path") == "/data/sub-001_dwi.nii.gz"
    assert ctx.get("fmri_path") == "/data/sub-001_bold.nii.gz"


def test_context_reference_image_defaults_to_t1():
    """`reference_image` is the most common flat key for cross-modality
    registration. Defaults to the primary T1 if present."""
    ctx = SubjectContext(_build_subject_with())
    assert ctx.get("reference_image") == "/data/sub-001_T1.nii.gz"


def test_context_reference_image_falls_back_to_other_structural():
    """If no T1, fall back to T2 / FLAIR."""
    s = Subject(subject_id="001", acquisitions=[
        Acquisition(acquisition_id="FLAIR_1", path="/data/flair.nii.gz",
                    filename="flair.nii.gz", modality="FLAIR"),
    ])
    ctx = SubjectContext(s)
    assert ctx.get("reference_image") == "/data/flair.nii.gz"


def test_context_reference_image_absent_when_no_structural():
    """If only DWI/fMRI are present, reference_image is not surfaced."""
    s = Subject(subject_id="001", acquisitions=[
        Acquisition(acquisition_id="DWI_1", path="/data/dwi.nii.gz",
                    filename="dwi.nii.gz", modality="DWI",
                    associated_files={"bval": "/data/dwi.bval", "bvec": "/data/dwi.bvec"}),
    ])
    ctx = SubjectContext(s)
    assert "reference_image" not in ctx
    # But bval/bvec should still be there
    assert ctx.get("bval_path") == "/data/dwi.bval"


# --------------------------------------------------------------------- #
# Structured API (the new, explicit way for future step code)
# --------------------------------------------------------------------- #

def test_context_get_acquisitions_returns_list():
    ctx = SubjectContext(_build_subject_with())
    t1_acqs = ctx.get_acquisitions("T1")
    assert len(t1_acqs) == 1
    assert t1_acqs[0].acquisition_id == "T1_1"


def test_context_get_acquisitions_returns_empty_for_missing_modality():
    ctx = SubjectContext(_build_subject_with())
    assert ctx.get_acquisitions("ASL") == []


def test_context_get_primary_acquisition():
    ctx = SubjectContext(_build_subject_with())
    t1 = ctx.get_primary_acquisition("T1")
    assert t1 is not None
    assert t1.path == "/data/sub-001_T1.nii.gz"


def test_context_get_reference_image():
    ctx = SubjectContext(_build_subject_with())
    assert ctx.get_reference_image("T1") == "/data/sub-001_T1.nii.gz"
    assert ctx.get_reference_image("DWI") == "/data/sub-001_dwi.nii.gz"
    assert ctx.get_reference_image("ASL") is None


def test_context_modalities():
    ctx = SubjectContext(_build_subject_with())
    assert set(ctx.modalities()) == {"T1", "DWI", "fMRI"}


# --------------------------------------------------------------------- #
# Multiple acquisitions of the same modality
# --------------------------------------------------------------------- #

def test_context_handles_multiple_acquisitions_of_same_modality():
    """The critical capability: a subject with two T1s doesn't lose either."""
    s = Subject(subject_id="001", acquisitions=[
        Acquisition(acquisition_id="T1_1", path="/data/t1_run1.nii.gz",
                    filename="t1_run1.nii.gz", modality="T1"),
        Acquisition(acquisition_id="T1_2", path="/data/t1_run2.nii.gz",
                    filename="t1_run2.nii.gz", modality="T1"),
    ])
    ctx = SubjectContext(s)

    # Structured API returns both
    t1_acqs = ctx.get_acquisitions("T1")
    assert len(t1_acqs) == 2

    # Primary is the first
    primary = ctx.get_primary_acquisition("T1")
    assert primary is not None
    assert primary.acquisition_id == "T1_1"

    # Flat key surfaces the primary (first) only; secondary is via structured API
    assert ctx.get("t1_path") == "/data/t1_run1.nii.gz"


# --------------------------------------------------------------------- #
# Structured routing of step-produced outputs
# --------------------------------------------------------------------- #

def test_context_routes_mask_writes_to_masks_dict():
    """When a step does `context.update({"brain_mask": path})`, the mask is
    also surfaced via the structured .masks dict."""
    ctx = SubjectContext(_build_subject_with())
    ctx.update({"brain_mask": "/path/to/mask.nii.gz"})

    assert ctx.get("brain_mask") == "/path/to/mask.nii.gz"
    assert ctx.masks["brain_mask"] == "/path/to/mask.nii.gz"


def test_context_routes_transform_writes_to_transforms_dict():
    ctx = SubjectContext(_build_subject_with())
    ctx.update({"transform": "/path/to/transform.tfm"})

    assert ctx.get("transform") == "/path/to/transform.tfm"
    assert ctx.transforms["transform"] == "/path/to/transform.tfm"


# --------------------------------------------------------------------- #
# Integration with the existing conftest fixture
# --------------------------------------------------------------------- #

def test_context_built_from_synthetic_dataset(synthetic_dataset):
    """End-to-end: build_dataset produces a Subject that SubjectContext
    can consume. This is the same path engine.py takes."""
    ds = build_dataset(synthetic_dataset["input_dir"])
    subject = ds.get_subject("001")

    ctx = SubjectContext(subject)
    assert ctx.subject_id == "001"
    assert set(ctx.modalities()) == {"T1", "fMRI"}
    assert ctx.get("t1_path").endswith("sub-001_T1.nii.gz")
    assert ctx.get("fmri_path").endswith("sub-001_task-rest_bold.nii.gz")
    # reference_image defaults to T1
    assert ctx.get("reference_image") == ctx.get("t1_path")


def test_context_repr_is_useful():
    """__repr__ shouldn't blow up on a normal context."""
    ctx = SubjectContext(_build_subject_with())
    s = repr(ctx)
    assert "001" in s
    assert "T1" in s
