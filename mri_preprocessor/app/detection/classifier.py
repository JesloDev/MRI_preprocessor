"""
Modality detection: classifies each discovered NIfTI file into a
modality label (T1, T2, FLAIR, DWI, fMRI, ...).

This is a heuristic filename/path keyword classifier -- simple,
transparent, and fast. For production-grade classification, replace
with a DICOM-header-based classifier (series description, sequence
name) upstream of NIfTI conversion, or a pybids-based classifier for
already-BIDS-organized data. Keeping this behind one function means
swapping the strategy later doesn't touch any other layer.
"""
import re
from typing import Dict, List, Optional

# NOTE: plain `\b` word-boundary regex does NOT work reliably here, because
# filenames use underscores/hyphens as separators (e.g. "sub-001_T1.nii.gz")
# and underscore is itself a \w character in Python's re module -- so
# `\bt1\b` fails to match "_T1." since there's no boundary between two
# word characters ("_" and "T"). We instead use explicit lookaround that
# only treats alphanumerics as "inside a token", so "_", "-", "." all
# correctly count as separators.
_ALNUM_BOUNDARY = r"(?<![a-zA-Z0-9])(?:{token})(?![a-zA-Z0-9])"


def _token(pattern: str) -> re.Pattern:
    return re.compile(_ALNUM_BOUNDARY.format(token=pattern), re.I)


# Ordered so more-specific patterns are checked before generic ones
# (e.g. "T1+C" / "T1CE" before plain "T1").
MODALITY_PATTERNS = [
    ("T1+C", re.compile(r"t1[a-z]*(ce|\+c|gad|contrast)", re.I)),
    ("FLAIR", _token(r"flair")),
    ("T2", _token(r"t2")),
    ("T1", _token(r"t1")),
    ("PD", re.compile(_ALNUM_BOUNDARY.format(token="pd") + r"|proton.?density", re.I)),
    ("SWI", re.compile(_ALNUM_BOUNDARY.format(token="swi") + r"|susceptibility", re.I)),
    ("GRE", _token(r"gre")),
    ("DTI", _token(r"dti")),
    ("DWI", re.compile(_ALNUM_BOUNDARY.format(token="dwi") + r"|diff(usion)?", re.I)),
    ("ADC", _token(r"adc")),
    ("fMRI", re.compile(_ALNUM_BOUNDARY.format(token="fmri") + r"|bold|rest(ing)?|task", re.I)),
    ("ASL", re.compile(_ALNUM_BOUNDARY.format(token="asl") + r"|perfusion", re.I)),
    ("TOF", _token(r"tof")),
    ("MRA", re.compile(_ALNUM_BOUNDARY.format(token="mra") + r"|angio", re.I)),
    ("MRV", re.compile(_ALNUM_BOUNDARY.format(token="mrv") + r"|venograph", re.I)),
]


def classify_modality(filename: str, filepath: str = "") -> Optional[str]:
    haystack = f"{filepath} {filename}"
    for modality, pattern in MODALITY_PATTERNS:
        if pattern.search(haystack):
            return modality
    return None  # "unclassified" -- flagged for manual review by the UI


def classify_manifest(manifest: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for entry in manifest:
        entry["modality"] = classify_modality(entry["filename"], entry["filepath"])
    return manifest


def modality_counts(manifest: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in manifest:
        key = entry.get("modality") or "unclassified"
        counts[key] = counts.get(key, 0) + 1
    return counts
