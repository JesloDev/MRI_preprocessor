"""
Processing step schema and registry.

Every preprocessing operation in the platform is represented as a
ProcessingStep object plus a callable implementation. Steps declare:
  - which modalities they apply to
  - which other steps they depend on (auto-resolved), with tiered
    semantics: hard (always required), recommended (default-on, can be
    skipped), conditional (required only when a structured `when`
    condition matches the requesting step's params).
  - which derived outputs they produce
  - a parameter schema with defaults (configurable per-operation)

The UI / REST API never hardcodes "does the user have FSL" logic -- it
only ever asks the registry "what steps exist for modality X" and the
dependency resolver "what's the full required chain for step Y".
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- #
# Parameter schema (PR11)
# --------------------------------------------------------------------- #

# A ParameterSpec is a structured declaration of ONE step parameter:
# its name, type, default, allowed values, and a human-readable
# description. This is what lets the CLI, Flask API, future GUI, and
# pipeline builder all discover what parameters a step accepts -- rather
# than each surface hardcoding its own list.
#
# The schema is intentionally lightweight:
#   - `type` is a string ("int", "float", "str", "bool", "list", "tuple")
#     rather than a Python type, so it serializes cleanly to JSON for
#     the API and is easy for non-Python clients to consume.
#   - `allowed_values` is an optional list -- when non-empty, the
#     parameter must take one of these values (e.g. method="zscore" |
#     "minmax" | "percentile" | "robust"). The CLI/API can render this
#     as a dropdown; the engine implementation can assert it.
#   - `validation` is a free-form description string for now; future
#     work may upgrade it to a structured validator (range checks,
#     regex, etc.). Keeping it as a string in PR11 means we don't have
#     to design the full validator framework before any of this is
#     useful.

ParamType = Literal["int", "float", "str", "bool", "list", "tuple"]


class ParameterSpec(BaseModel):
    """Declarative schema for ONE step parameter.

    Examples:
        ParameterSpec(name="engine", type="str",
                      allowed_values=["otsu","synthstrip","hdbet"],
                      default="otsu",
                      description="Which brain extraction engine to use.")

        ParameterSpec(name="voxel_size", type="tuple",
                      default=[1.0, 1.0, 1.0],
                      description="Target voxel spacing in mm (x,y,z).")
    """
    name: str
    type: ParamType
    default: Any = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    required: bool = False

    class Config:
        extra = "ignore"


# --------------------------------------------------------------------- #
# Maturity model (PR4)
# --------------------------------------------------------------------- #

# Maturity levels communicate to the UI / user how much to trust a given
# step implementation. This is a queryable property of the step itself,
# not a buried markdown note -- so the UI can literally badge "Experimental"
# next to Otsu-threshold brain extraction instead of presenting it as
# equivalent to SynthStrip.
#
# Levels (ordered from least to most trusted):
#   "planned"          - declared in the capability matrix but no
#                        implementation exists yet. Examples: cortical
#                        reconstruction (FreeSurfer recon-all), tractography.
#                        Useful so the UI can show "coming soon" rather than
#                        silently omitting the capability.
#
#   "experimental"     - implementation exists but is a placeholder or
#                        proof-of-concept. Not validated against reference
#                        implementations. Examples: the current Otsu-threshold
#                        brain_extraction (placeholders for SynthStrip/HD-BET).
#
#   "reference"        - a correct, working implementation that is suitable
#                        for development and small-scale use, but not the
#                        fastest or most validated option. Example: the
#                        SimpleITK rigid motion correction (a stand-in for
#                        FSL MCFLIRT).
#
#   "production_ready" - validated against reference outputs and suitable
#                        for use in published research. Reserved for steps
#                        that have passed an explicit validation pass.
#                        (No current steps qualify; this is the target for
#                        PR4+ to upgrade specific steps to.)
#
#   "external_engine" - implemented by an external engine (FSL, FreeSurfer,
#                        ANTs) that the user must install separately. The
#                        step itself is just a thin wrapper that shells out
#                        to the external tool. License caveats apply per
#                        engine -- see docs/CAPABILITY_MATRIX.yaml.

MaturityLevel = Literal[
    "planned",
    "experimental",
    "reference",
    "production_ready",
    "external_engine",
]


# --------------------------------------------------------------------- #
# Dependency tier model (PR3)
# --------------------------------------------------------------------- #

# Tier semantics:
#   "hard"        - always inserted by the resolver. The step cannot run
#                   without this prerequisite. Equivalent to the legacy
#                   `depends_on` list. Example: tensor_fitting hard-depends
#                   on dwi_motion_eddy_correction (it genuinely cannot run
#                   without preprocessed gradients).
#
#   "recommended" - inserted by default, but the user can skip it via
#                   `skip_recommended=[...]` on resolve_pipeline. Useful
#                   for "usually you want this, but you might have a
#                   reason to skip" -- e.g. registration usually wants
#                   brain_extraction first, but a user might pre-register
#                   their data and want to skip extraction.
#
#   "conditional" - inserted only if the structured `when` condition
#                   matches the requesting step's merged params
#                   (default_params + step_params[step_id]). The
#                   condition is a flat dict of expected {key: value}
#                   pairs -- NOT a Python expression -- so it's safe
#                   to validate and easy to serialize to JSON for the
#                   UI. Example: spatial_normalization is conditional on
#                   registration.atlas == "MNI".

DependencyTier = str  # "hard" | "recommended" | "conditional"


class Dependency(BaseModel):
    """A tiered dependency declaration.

    `when` is a flat dict mapping param name -> required value. The
    dependency fires only when ALL entries in `when` match the
    requesting step's effective params (default_params merged with
    step_params[step_id]). Empty/None `when` always fires (used for
    hard/recommended tiers where conditions don't apply).
    """
    step_id: str
    tier: DependencyTier = "hard"
    when: Optional[Dict[str, Any]] = None

    class Config:
        extra = "forbid"

    def matches(self, effective_params: Dict[str, Any]) -> bool:
        """Returns True if the `when` condition is satisfied by the
        given effective params. Empty/None `when` always matches.
        """
        if not self.when:
            return True
        for key, expected in self.when.items():
            if effective_params.get(key) != expected:
                return False
        return True


# --------------------------------------------------------------------- #
# Processing step schema
# --------------------------------------------------------------------- #

class ProcessingStep(BaseModel):
    id: str
    name: str
    modalities: List[str]
    engine: str  # e.g. "nibabel", "SimpleITK", "DIPY", "nilearn", "FSL", "FreeSurfer"
    # Legacy field: list of step ids that this step hard-depends on.
    # Kept for backward compatibility with existing step modules and
    # tests. Internally treated as a list of hard-tier Dependencies.
    depends_on: List[str] = Field(default_factory=list)
    # PR3 new field: structured tiered dependencies. Empty by default.
    dependencies: List[Dependency] = Field(default_factory=list)
    produces: List[str] = Field(default_factory=list)
    default_params: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    # PR4 new fields: maturity classification + free-form validation notes.
    # The UI uses `maturity` to badge steps (e.g. "Experimental" pill next
    # to Otsu brain extraction). `validation_notes` is shown in tooltips /
    # warning banners so users understand the caveats before running.
    maturity: MaturityLevel = "reference"
    validation_notes: Optional[str] = None
    # PR11 new field: declarative parameter schema. Empty by default;
    # steps that want their parameters discoverable by the CLI/API/GUI
    # populate this. The schema is the source of truth -- default_params
    # should stay consistent with parameters[*].default, but we don't
    # enforce that at schema level (callers can still pass params the
    # step didn't declare, for forward compatibility).
    parameters: List[ParameterSpec] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def all_dependencies(self) -> List[Dependency]:
        """Returns the merged list of all dependencies: legacy
        `depends_on` entries (as hard-tier Dependencies) followed by
        any structured `dependencies`. The order doesn't matter to the
        resolver; both lists are visited during expansion.
        """
        legacy = [Dependency(step_id=sid, tier="hard") for sid in self.depends_on]
        return legacy + list(self.dependencies)


# Signature every step implementation must follow:
#   fn(input_path: str, output_dir: str, params: dict, context: dict) -> dict
#
# `context` carries cross-step info for a given subject/run (e.g. reference
# image paths, bvals/bvecs paths, confounds path). Returns a result dict:
#   {"output_path": str, "outputs": {...}, "metrics": {...}}
StepFunction = Callable[[str, str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


class StepRegistry:
    def __init__(self) -> None:
        self._steps: Dict[str, ProcessingStep] = {}
        self._functions: Dict[str, StepFunction] = {}

    def register(self, step: ProcessingStep, fn: StepFunction) -> None:
        if step.id in self._steps:
            raise ValueError(f"Step id already registered: {step.id}")
        self._steps[step.id] = step
        self._functions[step.id] = fn

    def get(self, step_id: str) -> ProcessingStep:
        if step_id not in self._steps:
            raise KeyError(f"Unknown step id: {step_id}")
        return self._steps[step_id]

    def get_fn(self, step_id: str) -> StepFunction:
        if step_id not in self._functions:
            raise KeyError(f"No implementation registered for step id: {step_id}")
        return self._functions[step_id]

    def for_modality(self, modality: str) -> List[ProcessingStep]:
        return [s for s in self._steps.values() if modality in s.modalities]

    def all_steps(self) -> List[ProcessingStep]:
        return list(self._steps.values())

    def by_maturity(self, maturity: str) -> List[ProcessingStep]:
        """Returns all steps matching a given maturity level.

        Useful for the UI's "show only production-ready steps" filter,
        and for audit / dashboard views that want to surface what's
        experimental vs validated.
        """
        return [s for s in self._steps.values() if s.maturity == maturity]

    def maturity_counts(self) -> Dict[str, int]:
        """Returns a count of steps at each maturity level. Useful for
        API endpoints that report platform capabilities summary."""
        counts: Dict[str, int] = {m: 0 for m in (
            "planned", "experimental", "reference",
            "production_ready", "external_engine",
        )}
        for s in self._steps.values():
            counts[s.maturity] = counts.get(s.maturity, 0) + 1
        return counts

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serializable view for the REST API / UI."""
        return {sid: step.model_dump() for sid, step in self._steps.items()}


# Single shared registry instance for the whole application.
registry = StepRegistry()


def step(
    id: str,
    name: str,
    modalities: List[str],
    engine: str,
    depends_on: List[str] = None,
    dependencies: Optional[List[Dependency]] = None,
    produces: List[str] = None,
    default_params: Dict[str, Any] = None,
    description: str = "",
    maturity: MaturityLevel = "reference",
    validation_notes: Optional[str] = None,
    parameters: Optional[List[ParameterSpec]] = None,
):
    """Decorator used by step-implementation modules to self-register.

    Backward compatible with existing step modules that use only
    `depends_on`. New modules can additionally pass:
      - `dependencies=[...]` for tiered deps (hard / recommended / conditional)
      - `maturity="..."` to badge the step's trust level in the UI
      - `validation_notes="..."` to surface caveats / warnings
      - `parameters=[ParameterSpec(...)]` to declare the parameter schema
        (PR11) -- lets the CLI/API/GUI discover parameters dynamically
    """

    def decorator(fn: StepFunction) -> StepFunction:
        registry.register(
            ProcessingStep(
                id=id,
                name=name,
                modalities=modalities,
                engine=engine,
                depends_on=depends_on or [],
                dependencies=dependencies or [],
                produces=produces or [],
                default_params=default_params or {},
                description=description,
                maturity=maturity,
                validation_notes=validation_notes,
                parameters=parameters or [],
            ),
            fn,
        )
        return fn

    return decorator
