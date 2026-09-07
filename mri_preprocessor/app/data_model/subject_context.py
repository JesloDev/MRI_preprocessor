"""
SubjectContext: the seam between the dataset-aware engine and the
file-level step contract.

    fn(input_path, output_dir, params, context) -> dict

The step contract is unchanged. What changes is what `context` is:
previously `{}`, now a SubjectContext that exposes:

  - Structured subject/acquisition/sidecar data via helper methods.
  - Backward-compatible dict-like access (.get(), .update(), __getitem__,
    __setitem__, __contains__) so existing step code that does
    `context.get("bval_path")` or `context["brain_mask"] = path` keeps
    working unchanged.

Flat keys surfaced by default:
  - "subject_id"
  - "reference_image"        : path to the primary T1 (if any)
  - "bval_path"              : from the primary DWI acquisition (if any)
  - "bvec_path"              : from the primary DWI acquisition (if any)
  - "confounds_path"         : from the primary fMRI acquisition (if any)
  - "fmri_reference_image"   : path to the primary fMRI (if any)
  - "t1_path", "t2_path", "flair_path", "dwi_path", "fmri_path"
                              : convenience access to primary acquisitions
                                of each modality

Step functions may also write into the context via .update() -- the most
common pattern is `brain_extraction` writing `{"brain_mask": ...}`. Those
writes go into the flat dict, so subsequent steps see them via .get().

Multiple acquisitions of the same modality are accessible via:
    context.get_acquisitions("T1")          -> list[Acquisition]
    context.get_primary_acquisition("T1")  -> Acquisition | None
    context.get_reference_image("T1")      -> str | None
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.data_model.schema import Acquisition, Subject


# Maps a sidecar key (as stored in Acquisition.associated_files) to the
# flat context key that step code reads. For example, the primary DWI's
# "bval" sidecar is surfaced as context["bval_path"].
SIDECAR_TO_FLAT_KEY = {
    "bval":      "bval_path",
    "bvec":      "bvec_path",
    "json":      "json_sidecar_path",
    "confounds": "confounds_path",
}

# Modalities that should expose a flat "<modality_lower>_path" key
# pointing at the primary acquisition's image path. Keep the list short;
# this is convenience sugar, not the primary API.
MODALITY_FLAT_PATH_KEYS = {
    "T1":   "t1_path",
    "T2":   "t2_path",
    "FLAIR": "flair_path",
    "DWI":  "dwi_path",
    "fMRI": "fmri_path",
}


class SubjectContext:
    """A subject-level processing context.

    Replaces the previous `context={}` placeholder passed into
    `run_pipeline_for_file`. Quacks like a dict for backward compatibility
    with existing step code, while also exposing structured subject /
    acquisition / sidecar data via helper methods.
    """

    def __init__(self, subject: Subject):
        self.subject_id: str = subject.subject_id
        self.subject: Subject = subject
        # acquisitions is keyed by modality -> list of Acquisition objects.
        # This is a list (not a single Acquisition) precisely because a
        # subject may have multiple scans of the same modality.
        self.acquisitions: Dict[str, List[Acquisition]] = {}
        for acq in subject.acquisitions:
            modality_key = acq.modality if acq.modality else "UNK"
            self.acquisitions.setdefault(modality_key, []).append(acq)

        # Structured cross-step shared state.
        self.masks: Dict[str, str] = {}
        self.transforms: Dict[str, str] = {}
        self.sidecars: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = dict(subject.metadata)

        # Flat dict for backward-compatible .get() / .update() / __getitem__.
        # Populated below from the primary acquisitions' sidecars + image paths.
        self._flat: Dict[str, Any] = {"subject_id": self.subject_id}

        self._populate_flat_defaults()

    # ------------------------------------------------------------------ #
    # Structured API
    # ------------------------------------------------------------------ #

    def get_acquisitions(self, modality: str) -> List[Acquisition]:
        """All acquisitions of a given modality for this subject. May be empty."""
        return list(self.acquisitions.get(modality, []))

    def get_primary_acquisition(self, modality: str) -> Optional[Acquisition]:
        """The first acquisition of a given modality, if any."""
        acqs = self.acquisitions.get(modality, [])
        return acqs[0] if acqs else None

    def get_reference_image(self, modality: str) -> Optional[str]:
        """Path to the primary (first) acquisition's image of a modality."""
        acq = self.get_primary_acquisition(modality)
        return acq.path if acq else None

    def modalities(self) -> List[str]:
        """Distinct modalities present for this subject."""
        return [m for m in self.acquisitions.keys() if m != "UNK"]

    # ------------------------------------------------------------------ #
    # Dict-like API (backward compatibility with step code)
    # ------------------------------------------------------------------ #

    def get(self, key: str, default: Any = None) -> Any:
        return self._flat.get(key, default)

    def update(self, other: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Merges a dict into the flat context.

        Used by dag.py to write step outputs (brain_mask, transform, ...)
        into the context for downstream steps. Writes are also routed into
        the structured fields (masks, transforms) when the key matches a
        known prefix, so they're queryable structurally as well.
        """
        if other:
            for k, v in other.items():
                self._set_flat_key(k, v)
        for k, v in kwargs.items():
            self._set_flat_key(k, v)

    def _set_flat_key(self, key: str, value: Any) -> None:
        self._flat[key] = value
        # Route known structured keys into their typed collections too.
        # This means writing context["brain_mask"] = path also exposes it
        # via context.masks["brain_mask"] -- so downstream code can use
        # whichever access pattern is most natural.
        if key.endswith("_mask") or key == "mask":
            self.masks[key] = value
        elif key.endswith("transform") or key.endswith("_transform") or key == "transform":
            self.transforms[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._flat[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._set_flat_key(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._flat

    def keys(self):
        return self._flat.keys()

    def items(self):
        return self._flat.items()

    def values(self):
        return self._flat.values()

    def __repr__(self) -> str:
        return (
            f"SubjectContext(subject_id={self.subject_id!r}, "
            f"modalities={self.modalities()}, "
            f"acquisitions_per_modality="
            f"{{{', '.join(f'{m}: {len(a)}' for m, a in self.acquisitions.items())}}})"
        )

    # ------------------------------------------------------------------ #
    # Internal: populate the flat dict from the primary acquisitions
    # ------------------------------------------------------------------ #

    def _populate_flat_defaults(self) -> None:
        """Surfaces the primary acquisition's image path and sidecars
        as flat keys so existing step code keeps working without change.

        For example, the primary DWI acquisition's bval file is surfaced
        as `context["bval_path"]`. If a subject has multiple DWIs, only
        the first (primary) one's sidecars are surfaced as flat keys;
        step code that needs the secondary DWI's sidecars should use
        `context.get_acquisitions("DWI")[1].sidecars["bval"]` instead.
        """
        for modality, flat_path_key in MODALITY_FLAT_PATH_KEYS.items():
            primary = self.get_primary_acquisition(modality)
            if primary:
                self._flat[flat_path_key] = primary.path

        # Sidecar surfacing -- per modality, take the primary acquisition's
        # sidecars. A subject with multiple DWIs only surfaces the first's
        # bval/bvec as flat keys; the rest are available via .get_acquisitions().
        for modality_acqs in self.acquisitions.values():
            if not modality_acqs:
                continue
            primary = modality_acqs[0]
            for sidecar_key, sidecar_path in primary.associated_files.items():
                flat_key = SIDECAR_TO_FLAT_KEY.get(sidecar_key, f"{sidecar_key}_path")
                # Don't overwrite if a previous modality already populated it
                self._flat.setdefault(flat_key, sidecar_path)
                self.sidecars[flat_key] = sidecar_path

        # "reference_image" is a commonly-used flat key for cross-modality
        # registration. We default it to the primary T1 if present (the
        # most common reference choice in neuroimaging), else to the first
        # available structural acquisition. Steps that need a different
        # reference can override via params.
        if "t1_path" in self._flat:
            self._flat.setdefault("reference_image", self._flat["t1_path"])
        else:
            for mod in ("T2", "FLAIR", "T1"):
                key = f"{mod.lower()}_path"
                if key in self._flat:
                    self._flat.setdefault("reference_image", self._flat[key])
                    break
