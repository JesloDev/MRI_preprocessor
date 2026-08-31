"""
Signal sources for multi-signal modality classification.

Each signal source examines ONE source of evidence (filename, path,
NIfTI header, sidecar files, dimensionality) and produces a
ClassificationSignal voting for a specific modality, or None ("no
opinion"). The MultiSignalClassifier aggregates these into a final
ClassificationResult with confidence scoring.

Design notes:
  - The SignalSource interface is intentionally minimal: just `name` +
    `evaluate(...) -> Optional[ClassificationSignal]`. This makes
    adding a new signal source (e.g. a DICOM-header signal, or a
    machine-learning-based signal) trivial.
  - Weights are configurable per signal source but have sensible
    defaults. The defaults are NOT magic numbers -- they reflect a
    reasonable prior about each signal's reliability:
        sidecar    = 2.0  (bval+bvec presence is near-definitive for DWI)
        filename   = 1.0  (the existing heuristic, treated as one vote)
        path       = 0.7  (BIDS dir hints are useful but not authoritative)
        dimension  = 0.6  (4D vs 3D narrows the family but doesn't pin a member)
        header     = 0.5  (low-level header hints -- reserved for future use)
  - FilenameSignal reuses the regex patterns from the existing
    `app.detection.classifier` module so we don't duplicate them. PR3+
    may consolidate classifier.py into this module.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

from app.data_model.schema import ClassificationSignal
from app.detection.classifier import MODALITY_PATTERNS  # reuse regex set


class SignalSource(Protocol):
    """A source of modality classification evidence."""

    name: str

    def evaluate(
        self,
        filepath: str,
        filename: str,
        sidecars: dict,
        metadata: dict,
    ) -> Optional[ClassificationSignal]:
        """Returns a ClassificationSignal voting for a modality, or None
        if this source has no opinion."""
        ...


# --------------------------------------------------------------------- #
# Concrete signal sources
# --------------------------------------------------------------------- #

class FilenameSignal:
    """Filename keyword/pattern classifier.

    Reuses the regex patterns from `app.detection.classifier` so the
    rule set stays in one place. The patterns are checked in order --
    more-specific patterns (T1+C, FLAIR, T2, T1) before generic ones
    (DWI, fMRI, ...) -- and the first match wins.

    This is the legacy signal: it produces one vote per file, with
    weight 1.0 (configurable). Other signals add evidence on top.
    """

    name = "filename"

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, filepath, filename, sidecars, metadata):
        haystack = f"{filepath} {filename}"
        for modality, pattern in MODALITY_PATTERNS:
            m = pattern.search(haystack)
            if m:
                return ClassificationSignal(
                    source=self.name,
                    modality=modality,
                    weight=self.weight,
                    evidence=(
                        f"filename pattern matched '{m.group(0)}' "
                        f"-> {modality}"
                    ),
                )
        return None


class PathSignal:
    """Examines path components for BIDS-style directory hints.

    BIDS layout uses directories like 'anat/', 'dwi/', 'func/', 'fmap/'.
    Some of these unambiguously pin a modality (dwi/ -> DWI, func/ ->
    fMRI); others (anat/) are ambiguous across structural modalities and
    we abstain.
    """

    name = "path"

    # BIDS directory name -> modality (None = ambiguous, no vote)
    BIDS_DIR_TO_MODALITY = {
        "anat":  None,  # could be T1/T2/FLAIR/PD -- let filename decide
        "dwi":   "DWI",
        "dti":   "DTI",
        "func":  "fMRI",
        "fmap":  None,  # fieldmap, not in our modality list
        "perf":  "ASL",
    }

    def __init__(self, weight: float = 0.7):
        self.weight = weight

    def evaluate(self, filepath, filename, sidecars, metadata):
        parts = filepath.split(os.sep)
        for part in parts:
            part_lower = part.lower()
            if part_lower in self.BIDS_DIR_TO_MODALITY:
                modality = self.BIDS_DIR_TO_MODALITY[part_lower]
                if modality:
                    return ClassificationSignal(
                        source=self.name,
                        modality=modality,
                        weight=self.weight,
                        evidence=f"path component '{part}' suggests {modality}",
                    )
        return None


class SidecarSignal:
    """Examines sidecar file presence.

    This is high-confidence evidence: a bval+bvec pair is essentially
    definitive proof that a file is a DWI acquisition. A confounds
    sidecar is strong (but not exclusive) evidence for fMRI.
    """

    name = "sidecar"

    def __init__(self, weight: float = 2.0):
        self.weight = weight

    def evaluate(self, filepath, filename, sidecars, metadata):
        # bval+bvec pair -> DWI (near-definitive)
        if "bval" in sidecars and "bvec" in sidecars:
            return ClassificationSignal(
                source=self.name,
                modality="DWI",
                weight=self.weight,
                evidence=(
                    f"bval+bvec sidecars present "
                    f"({os.path.basename(sidecars['bval'])}, "
                    f"{os.path.basename(sidecars['bvec'])})"
                ),
            )

        # confounds tsv -> fMRI (strong but not exclusive: a confounds
        # file could in principle accompany other modalities too)
        if "confounds" in sidecars:
            return ClassificationSignal(
                source=self.name,
                modality="fMRI",
                weight=self.weight,
                evidence=(
                    f"confounds sidecar present "
                    f"({os.path.basename(sidecars['confounds'])})"
                ),
            )

        return None


class DimensionSignal:
    """Uses NIfTI dimensionality to narrow the modality family.

    A 4D volume is either fMRI or DWI -- we can't tell which without
    sidecars or filename, so we abstain. A 3D volume is structural
    (T1/T2/FLAIR/PD) -- again, no opinion on which.

    PR2 leaves this as a no-op stub because the existing signals
    (filename + sidecar) already cover the cases we care about. The
    hook is here so a future signal (e.g. "4D + no bval -> likely fMRI")
    can be added without changing the classifier shape.

    Future use cases:
      - 4D + no bval/bvec sidecars + many volumes (>50) -> fMRI (weight ~0.8)
      - 3D + small voxel size (<1mm iso) -> high-res T1 (weight ~0.4)
      - 3D + b0-like intensity distribution -> possibly ADC (weight ~0.3)
    """

    name = "dimension"

    def __init__(self, weight: float = 0.6):
        self.weight = weight

    def evaluate(self, filepath, filename, sidecars, metadata):
        # PR2 placeholder -- structure-only, no rules yet.
        # See class docstring for the rules we plan to add in PR3.
        return None


class HeaderSignal:
    """Examines NIfTI header properties (voxel size, intensity stats).

    PR2 leaves this as a no-op stub. The structure is here so PR3+ can
    add header-derived signals (e.g. mean intensity + tissue contrast
    hint to distinguish T1 vs T2 vs FLAIR when filename is ambiguous)
    without changing the classifier shape.
    """

    name = "header"

    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def evaluate(self, filepath, filename, sidecars, metadata):
        # PR2 placeholder -- structure-only, no rules yet.
        return None


# --------------------------------------------------------------------- #
# Default signal set
# --------------------------------------------------------------------- #

def default_signals() -> list:
    """Returns the default list of signal sources used by the
    MultiSignalClassifier when none are explicitly configured.

    Order is irrelevant -- each signal is independent and the
    aggregator sums weights per modality.
    """
    return [
        FilenameSignal(),
        PathSignal(),
        SidecarSignal(),
        DimensionSignal(),
        HeaderSignal(),
    ]
