"""
Builds a Dataset tree from a flat scanner manifest.

Pipeline:
    scanner.scan_directory(input_dir)            # flat list of files
        |
        v
    builder.build_dataset(input_dir, strategy)   # applies subject inference
        |                                          # + sidecar discovery
        v
    Dataset                                       # tree of Subjects -> Acquisitions

For now, modality classification is delegated to the existing
`app.detection.classifier.classify_modality`, producing a single-signal
ClassificationResult. PR3 will replace this with a multi-signal classifier
that combines filename + header + sidecar evidence; the schema already
supports that, so this is the only file that needs to change for the upgrade.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

from app.config import NIFTI_EXTENSIONS
from app.data_model.relationships import discover_sidecars
from app.data_model.schema import (
    Acquisition,
    ClassificationResult,
    ClassificationSignal,
    Dataset,
    Subject,
)
from app.data_model.subject_inference import (
    AutoSubjectStrategy,
    SubjectInferenceStrategy,
)
from app.detection.classifier import classify_modality
from app.ingestion.scanner import scan_directory


def _strip_nifti_ext(filename: str) -> str:
    for ext in NIFTI_EXTENSIONS:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename


def _generate_acquisition_id(modality: Optional[str], index: int) -> str:
    """Generates a unique-within-subject acquisition id like 'T1_1', 'DWI_2'.

    The index is 1-based and per-modality, so a subject with two T1s gets
    T1_1 and T1_2. Unclassified acquisitions use 'UNK_1', 'UNK_2', ...
    """
    label = modality if modality else "UNK"
    return f"{label}_{index}"


def _read_header_metadata(filepath: str) -> Dict[str, object]:
    """Reads lightweight metadata from the NIfTI header (no full volume load).

    Used to populate Acquisition.metadata -- shape, ndim, voxel size. Kept
    intentionally cheap so building a Dataset over thousands of files
    doesn't become I/O-bound.
    """
    try:
        import nibabel as nib
        img = nib.load(filepath)
        return {
            "shape": list(img.shape),
            "ndim": img.ndim,
            "voxel_size": list(img.header.get_zooms()),
            "dtype": str(img.get_data_dtype()),
        }
    except Exception:  # noqa: BLE001 -- bad files are reported by QC layer, not here
        return {"load_error": True}


def _classify_acquisition(
    filename: str,
    filepath: str,
    sidecars: Dict[str, str],
    metadata: Optional[Dict[str, object]] = None,
) -> ClassificationResult:
    """PR2 implementation: multi-signal classifier.

    Combines filename + path + sidecar evidence with weighted confidence
    scoring. Falls back to manual review (modality=None) when confidence
    is below the default threshold (0.5).

    Each signal source (FilenameSignal, PathSignal, SidecarSignal, ...)
    votes for a specific modality with a configurable weight. The
    aggregator sums weights per modality and computes a confidence score
    that reflects agreement across signals. See
    `app/detection/multisignal_classifier.py` for the full algorithm.
    """
    from app.detection.multisignal_classifier import classify_acquisition
    return classify_acquisition(filepath, filename, sidecars, metadata or {})


def build_dataset(
    input_dir: str,
    subject_strategy: Optional[SubjectInferenceStrategy] = None,
    read_headers: bool = True,
    classify: bool = True,
    manifest: Optional[List[dict]] = None,
) -> Dataset:
    """Builds a Dataset tree from an input directory.

    Args:
        input_dir: directory to scan (passed to scanner.scan_directory if
            `manifest` is None).
        subject_strategy: how to infer subject ids. Defaults to
            AutoSubjectStrategy, which preserves the existing scanner
            behaviour (BIDS-first, depth-1 fallback).
        read_headers: if True, reads lightweight NIfTI header metadata
            (shape, ndim, voxel_size) for each acquisition. Set to False
            to skip header reads (useful for dry-run / scan-only flows).
        classify: if True, runs the classifier on each acquisition and
            populates `modality` and `classification`. If False, leaves
            them as None -- useful for testing the data model in
            isolation from the classifier.
        manifest: optional pre-built scanner manifest. If provided, skips
            the internal scan_directory call (lets callers reuse a
            manifest that's already been QC-filtered).

    Returns:
        A Dataset with one Subject per inferred subject id, each containing
        its Acquisitions (with sidecars and metadata populated).
    """
    if manifest is None:
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
        manifest = scan_directory(input_dir)
    elif not os.path.isdir(input_dir):
        # Caller passed a manifest but the input_dir is just a label.
        # That's fine for the manifest-driven path; we just skip the
        # is-dir precondition check.
        pass

    strategy = subject_strategy or AutoSubjectStrategy()

    # Group files by inferred subject id. Files of the same modality get
    # numbered within a subject (T1_1, T1_2, ...).
    by_subject: Dict[str, List[dict]] = defaultdict(list)
    for entry in manifest:
        subject_id = strategy.infer(input_dir, entry["filepath"])
        by_subject[subject_id].append(entry)

    # Deterministic subject ordering (sorted by subject id)
    subjects: List[Subject] = []
    for subject_id in sorted(by_subject.keys()):
        entries = by_subject[subject_id]
        # Sort within subject for deterministic acquisition_id assignment
        entries.sort(key=lambda e: e["filepath"])

        # First pass: classify each entry, build Acquisition objects
        # We need to know the modality before assigning acquisition_id,
        # because the id is per-modality (T1_1, T1_2, ...).
        acquisitions: List[Acquisition] = []
        per_modality_counter: Dict[str, int] = defaultdict(int)

        for entry in entries:
            filepath = entry["filepath"]
            filename = entry["filename"]
            sidecars = discover_sidecars(filepath)
            metadata = _read_header_metadata(filepath) if read_headers else {}

            if classify:
                # Pass metadata to the classifier so it can use header-derived
                # signals (shape, ndim, voxel_size) in addition to filename
                # and sidecars.
                classification = _classify_acquisition(
                    filename, filepath, sidecars, metadata,
                )
                modality = classification.modality
            else:
                classification = None
                modality = None

            label = modality if modality else "UNK"
            per_modality_counter[label] += 1
            acq_id = _generate_acquisition_id(modality, per_modality_counter[label])

            acquisitions.append(Acquisition(
                acquisition_id=acq_id,
                path=filepath,
                filename=filename,
                modality=modality,
                associated_files=sidecars,
                metadata=metadata,
                classification=classification,
            ))

        subjects.append(Subject(
            subject_id=subject_id,
            acquisitions=acquisitions,
        ))

    return Dataset(
        path=input_dir,
        subjects=subjects,
        metadata={
            "num_files": len(manifest),
            "num_subjects": len(subjects),
            "subject_inference_strategy": type(strategy).__name__,
        },
    )
