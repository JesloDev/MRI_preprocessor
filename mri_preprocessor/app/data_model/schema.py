"""
Core data model entities.

    Dataset
      └── Subject
            └── Acquisition (image + sidecars + classification + metadata)

Design notes:
  - A Subject may have multiple acquisitions of the same modality (e.g.
    two T1 scans, two DWI series). That is why `Subject.get_acquisitions`
    returns a list, not a single Acquisition. Helper
    `get_primary_acquisition(modality)` returns the first one (a
    deterministic default; later we may add a "primary" flag on
    Acquisition for an explicit choice).
  - ClassificationResult is multi-signal: a single acquisition can be
    classified using filename + header + sidecar evidence, with a
    confidence score and a list of alternatives. This is forward-looking
    -- for PR1 the classifier is still filename-only and produces a
    ClassificationResult with a single filename signal. The shape is
    here so PR3 (multi-signal classification) can extend it without
    touching the schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClassificationSignal:
    """One piece of evidence contributing to a modality classification.

    PR2 addition: each signal votes for a specific modality. This lets
    the MultiSignalClassifier aggregate votes per modality and compute a
    confidence score that reflects agreement (multiple signals voting
    for the same modality) vs. disagreement (signals split across
    modalities).
    """
    source: str   # "filename" | "header" | "sidecar" | "path" | "dimension"
    modality: str  # which modality this signal votes for
    weight: float
    evidence: str  # human-readable description of what was matched


@dataclass
class ClassificationResult:
    modality: Optional[str]  # None = unclassified / below threshold
    confidence: float = 0.0  # 0.0 .. 1.0
    signals: List[ClassificationSignal] = field(default_factory=list)
    alternatives: List[tuple] = field(default_factory=list)  # [(modality, confidence), ...]

    def is_confident(self, threshold: float = 0.5) -> bool:
        return self.confidence >= threshold


@dataclass
class Acquisition:
    """
    One image file plus everything directly associated with it
    (sidecars, derived metadata, classification).
    """
    acquisition_id: str  # e.g. "T1_1", "DWI_2" -- unique within a subject
    path: str             # absolute path to the .nii / .nii.gz
    filename: str         # basename
    modality: Optional[str] = None  # filled in by classifier
    associated_files: Dict[str, str] = field(default_factory=dict)
        # "bval" -> path, "bvec" -> path, "json" -> path, "confounds" -> path
    metadata: Dict[str, Any] = field(default_factory=dict)
        # header-derived: shape, ndim, voxel_size, etc.
    classification: Optional[ClassificationResult] = None

    @property
    def sidecars(self) -> Dict[str, str]:
        """Alias for associated_files -- a common name in the BIDS world."""
        return self.associated_files


@dataclass
class Subject:
    subject_id: str
    acquisitions: List[Acquisition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_acquisitions(self, modality: str) -> List[Acquisition]:
        """All acquisitions of a given modality. May be empty."""
        return [a for a in self.acquisitions if a.modality == modality]

    def get_primary_acquisition(self, modality: str) -> Optional[Acquisition]:
        """The first acquisition of a given modality, if any.

        This is a deterministic default; future work may add an explicit
        `primary: bool` flag on Acquisition, or a session-aware primary
        selector. For now "first by scan order" is good enough and the
        step code never has to assume there's only one.
        """
        acqs = self.get_acquisitions(modality)
        return acqs[0] if acqs else None

    def modalities(self) -> List[str]:
        """Distinct modalities present for this subject."""
        seen: List[str] = []
        for acq in self.acquisitions:
            if acq.modality and acq.modality not in seen:
                seen.append(acq.modality)
        return seen


@dataclass
class Dataset:
    path: str  # input_dir
    subjects: List[Subject] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        for s in self.subjects:
            if s.subject_id == subject_id:
                return s
        return None

    @property
    def subject_ids(self) -> List[str]:
        return [s.subject_id for s in self.subjects]

    def acquisition_count(self) -> int:
        return sum(len(s.acquisitions) for s in self.subjects)
