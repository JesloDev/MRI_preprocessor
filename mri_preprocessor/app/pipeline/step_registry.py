"""
Processing step schema and registry.

Every preprocessing operation in the platform is represented as a
ProcessingStep object plus a callable implementation. Steps declare:
  - which modalities they apply to
  - which other step ids they depend on (auto-resolved, per your spec)
  - which derived outputs they produce
  - a parameter schema with defaults (configurable per-operation)

The UI / REST API never hardcodes "does the user have FSL" logic -- it
only ever asks the registry "what steps exist for modality X" and the
dependency resolver "what's the full required chain for step Y".
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field


class ProcessingStep(BaseModel):
    id: str
    name: str
    modalities: List[str]
    engine: str  # e.g. "nibabel", "SimpleITK", "DIPY", "nilearn", "FSL", "FreeSurfer"
    depends_on: List[str] = Field(default_factory=list)
    produces: List[str] = Field(default_factory=list)
    default_params: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    class Config:
        arbitrary_types_allowed = True


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
    produces: List[str] = None,
    default_params: Dict[str, Any] = None,
    description: str = "",
):
    """Decorator used by step-implementation modules to self-register."""

    def decorator(fn: StepFunction) -> StepFunction:
        registry.register(
            ProcessingStep(
                id=id,
                name=name,
                modalities=modalities,
                engine=engine,
                depends_on=depends_on or [],
                produces=produces or [],
                default_params=default_params or {},
                description=description,
            ),
            fn,
        )
        return fn

    return decorator
