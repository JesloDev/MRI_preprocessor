"""
Subject-id inference strategies.

A SubjectInferenceStrategy takes (input_dir, filepath) and returns a
subject id. Different datasets use different conventions; the strategy
pattern lets the builder stay agnostic.

Strategies:
  - BIDSSubjectStrategy  : use the 'sub-XXX' component anywhere in the path
                           (this is what the existing scanner does today).
  - DirectoryDepthStrategy(depth=N): take the Nth directory component
                           under the input root. Useful for non-BIDS layouts
                           like Root/ScanFolder/Subject/T1.nii.gz where the
                           subject is at depth 2 (not depth 1).
  - RegexStrategy(pattern): a user-supplied regex with a named group `subject`
                           applied to the relative path.
  - AutoSubjectStrategy   : the default; tries BIDS first, falls back to
                           depth-1. Equivalent to the existing scanner
                           heuristic, so existing tests keep passing.

Pluggability: strategies are picked per-dataset, configured via the API
or app/config.py. Future work may add a registry of named strategies that
the API can expose for the UI to render as a dropdown.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

BIDS_SUBJECT_RE = re.compile(r"sub-([a-zA-Z0-9]+)")


class SubjectInferenceStrategy(Protocol):
    """Returns the subject id for a given NIfTI file path."""

    def infer(self, input_dir: str, filepath: str) -> str:
        ...


class BIDSSubjectStrategy:
    """Uses the first 'sub-XXX' component anywhere in the relative path."""

    def infer(self, input_dir: str, filepath: str) -> str:
        rel_path = os.path.relpath(filepath, input_dir)
        for part in rel_path.split(os.sep):
            m = BIDS_SUBJECT_RE.match(part)
            if m:
                return m.group(1)
        # Fall back to top-level dir if no BIDS marker
        parts = rel_path.split(os.sep)
        return parts[0] if len(parts) > 1 else os.path.basename(filepath)


class DirectoryDepthStrategy:
    """Uses the Nth directory component under the input root as the subject.

    For Root/ScanFolder/Subject/T1.nii.gz with depth=2, this returns
    'Subject' (whereas the existing heuristic would wrongly return
    'ScanFolder').
    """

    def __init__(self, depth: int = 1):
        if depth < 1:
            raise ValueError("DirectoryDepthStrategy depth must be >= 1")
        self.depth = depth

    def infer(self, input_dir: str, filepath: str) -> str:
        rel_path = os.path.relpath(filepath, input_dir)
        parts = rel_path.split(os.sep)
        # parts[0] is the first dir under input root; parts[depth-1] is what we want
        if len(parts) > self.depth:
            return parts[self.depth - 1]
        # Path is shallower than expected; fall back to whatever is available
        return parts[0] if len(parts) > 1 else os.path.basename(filepath)


class RegexStrategy:
    """Applies a user-supplied regex with a `subject` named group to the
    relative path. Useful for bespoke layouts.

    Example: RegexStrategy(r"Scan_(?P<subject>\\w+)/") matches
    'Scan_ABC123/...' and returns 'ABC123'.
    """

    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def infer(self, input_dir: str, filepath: str) -> str:
        rel_path = os.path.relpath(filepath, input_dir)
        m = self.pattern.search(rel_path)
        if m and "subject" in m.groupdict():
            return m.group("subject")
        if m:
            return m.group(0)
        # No match: fall back to filename stem
        return os.path.splitext(os.path.basename(filepath))[0]


class AutoSubjectStrategy:
    """The default strategy. Tries BIDS first, falls back to depth-1.

    This preserves the exact behaviour of the existing scanner's
    `_infer_subject_id`, so existing tests that assert subject-id values
    keep passing.
    """

    def infer(self, input_dir: str, filepath: str) -> str:
        rel_path = os.path.relpath(filepath, input_dir)
        parts = rel_path.split(os.sep)

        for part in parts:
            m = BIDS_SUBJECT_RE.match(part)
            if m:
                return m.group(1)

        if len(parts) > 1:
            return parts[0]

        # No directory structure at all: fall back to filename stem
        return _strip_nifti_ext(os.path.basename(filepath))


def _strip_nifti_ext(filename: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename
