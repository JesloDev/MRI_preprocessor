"""
Relationship discovery: for each NIfTI file, find sibling sidecar files
that belong to the same acquisition (bval, bvec, json, confounds tsv, ...).

This is what gives SubjectContext its real value: instead of asking the
user to wire up "this DWI uses these bvals/bvecs", we discover the
relationship from the filesystem layout (BIDS-style co-location: a DWI
acquisition's bval file has the same stem as the .nii.gz, with .bval
extension).
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

# Sidecar discovery rules. The order within each modality's list matters:
# earlier patterns are tried first.
#
# Each rule is (key, [filename_suffix_candidates]). The builder looks for
# a sibling file with the same stem + suffix.
SIDECAR_RULES: List[Tuple[str, List[str]]] = [
    ("bval",      [".bval"]),
    ("bvec",      [".bvec"]),
    ("json",      [".json"]),
    ("confounds", ["_confounds.tsv", "_desc-confounds.tsv",
                   "confounds.tsv", "_desc-confounds_regressors.tsv"]),
]


def _strip_nifti_ext(filename: str) -> str:
    """Returns the filename without its .nii / .nii.gz extension."""
    for ext in (".nii.gz", ".nii"):
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename


def discover_sidecars(nifti_path: str) -> Dict[str, str]:
    """For one NIfTI file, return the sibling sidecar files that exist.

    Returns a dict like {"bval": "/path/to/sub-001_dwi.bval", ...}.
    Empty dict if no sidecars are present (which is the common case for
    structural T1/T2/FLAIR acquisitions).
    """
    directory = os.path.dirname(nifti_path)
    stem = _strip_nifti_ext(os.path.basename(nifti_path))

    sidecars: Dict[str, str] = {}
    for key, suffixes in SIDECAR_RULES:
        for suffix in suffixes:
            candidate = os.path.join(directory, stem + suffix)
            if os.path.isfile(candidate):
                sidecars[key] = candidate
                break
    return sidecars
