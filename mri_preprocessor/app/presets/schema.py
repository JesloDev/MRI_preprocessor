"""
Preset schema (PR10).

A Preset is a named, saveable configuration that points at existing
processing steps. It contains NO processing logic -- just the
configuration the user would otherwise pass on the command line.

When a preset is applied (via CLI `--preset <name>` or API
`{"preset": "<name>"}`), it expands into the SAME arguments that
resolve_pipeline + run_job already accept:

    resolve_pipeline(
        step_ids=preset.step_ids,
        step_params=preset.step_params,
        skip_recommended=preset.skip_recommended,
    )

That's the whole story. Presets are not a new execution path; they're
a configuration shorthand that the CLI, Flask API, and future GUI all
resolve into the same backend call.

Schema fields:
    name             : unique identifier (also used as filename on disk)
    description      : human-readable purpose, shown in `presets show`
    step_ids         : the user's requested steps (pre-resolution)
    step_params      : per-step param overrides (e.g. engine choices)
    skip_recommended : recommended-dependency step ids to suppress
    target_modalities: optional filter (e.g. ["T1", "FLAIR"])
    is_builtin       : True for the 4 built-in presets (can't be deleted)
    created_at       : timestamp (used for sorting custom presets)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Preset(BaseModel):
    """A named, saveable pipeline configuration.

    The schema mirrors the resolve_pipeline + run_job argument shape
    exactly, so applying a preset is a 1:1 translation -- no
    preset-specific execution logic anywhere.
    """
    name: str
    description: str = ""
    step_ids: List[str] = Field(default_factory=list)
    step_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    skip_recommended: List[str] = Field(default_factory=list)
    target_modalities: Optional[List[str]] = None
    is_builtin: bool = False
    created_at: float = Field(default_factory=time.time)

    class Config:
        # Allows the schema to evolve without breaking saved-on-disk
        # presets from older versions: missing fields fall back to
        # their defaults instead of raising a validation error.
        extra = "ignore"

    def to_apply_kwargs(self) -> Dict[str, Any]:
        """Returns the kwargs dict to pass to resolve_pipeline / run_job
        when applying this preset.

        This is the load-bearing method: it's the single translation
        from Preset -> backend arguments. CLI, API, and future GUI all
        use this.
        """
        return {
            "selected_step_ids": list(self.step_ids),
            "step_params": dict(self.step_params),
            "skip_recommended": list(self.skip_recommended),
            "target_modalities": self.target_modalities,
        }
