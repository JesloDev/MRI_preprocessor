"""
Dataset ingestion: recursively scans an input directory for NIfTI files
and groups them by inferred subject, with zero manual file location
required from the user.

Subject-id inference strategy (in priority order):
  1. BIDS-style: a path component matching 'sub-<id>'
  2. First directory component under the input root (common for
     non-BIDS clinical/research exports, e.g. INPUT/CSVD-001/T1.nii.gz)
  3. Falls back to the filename stem if no directory structure exists.

This is intentionally a heuristic, swappable module -- replace with a
full pybids.BIDSLayout-based implementation if your datasets are BIDS
compliant.
"""
import os
import re
from typing import Dict, List

from app.config import NIFTI_EXTENSIONS

BIDS_SUBJECT_RE = re.compile(r"sub-([a-zA-Z0-9]+)")


def _strip_nifti_ext(filename: str) -> str:
    for ext in NIFTI_EXTENSIONS:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename


def _infer_subject_id(root_dir: str, filepath: str) -> str:
    rel_path = os.path.relpath(filepath, root_dir)
    parts = rel_path.split(os.sep)

    for part in parts:
        m = BIDS_SUBJECT_RE.match(part)
        if m:
            return m.group(1)

    if len(parts) > 1:
        return parts[0]

    return _strip_nifti_ext(parts[-1])


def scan_directory(input_dir: str) -> List[Dict[str, str]]:
    """
    Returns a flat manifest:
      [{"subject": "CSVD-001", "filepath": "...", "filename": "..."}, ...]
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    manifest = []
    for dirpath, _dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(NIFTI_EXTENSIONS):
                filepath = os.path.join(dirpath, filename)
                manifest.append({
                    "subject": _infer_subject_id(input_dir, filepath),
                    "filepath": filepath,
                    "filename": filename,
                })
    return manifest


def summarize_manifest(manifest: List[Dict[str, str]]) -> Dict[str, object]:
    subjects = sorted({m["subject"] for m in manifest})
    return {
        "num_subjects": len(subjects),
        "subjects": subjects,
        "num_files": len(manifest),
    }
