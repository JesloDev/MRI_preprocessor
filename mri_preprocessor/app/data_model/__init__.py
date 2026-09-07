"""
Data model layer: Dataset -> Subject -> Acquisition (+ relationships).

This package sits between `ingestion/` (which produces a flat manifest of
files) and `pipeline/` (which executes processing steps against files).

The flat manifest treats every file as independent. The Dataset tree
captures the relationships between files that belong to the same subject
or acquisition -- which is what unblocks multimodal operations (FLAIR-to-T1
registration, DWI tensor fitting with bval/bvec sidecars, fMRI confound
regression with motion params, etc.).

Public API:
    from app.data_model import (
        Dataset, Subject, Acquisition, ClassificationResult,
        SubjectContext, build_dataset, discover_sidecars,
    )
"""
from app.data_model.schema import (
    Acquisition,
    ClassificationResult,
    ClassificationSignal,
    Dataset,
    Subject,
)
from app.data_model.relationships import discover_sidecars
from app.data_model.builder import build_dataset
from app.data_model.subject_context import SubjectContext
from app.data_model.subject_inference import (
    AutoSubjectStrategy,
    BIDSSubjectStrategy,
    DirectoryDepthStrategy,
    RegexStrategy,
    SubjectInferenceStrategy,
)

__all__ = [
    "Acquisition",
    "ClassificationResult",
    "ClassificationSignal",
    "Dataset",
    "Subject",
    "SubjectContext",
    "SubjectInferenceStrategy",
    "AutoSubjectStrategy",
    "BIDSSubjectStrategy",
    "DirectoryDepthStrategy",
    "RegexStrategy",
    "build_dataset",
    "discover_sidecars",
]
