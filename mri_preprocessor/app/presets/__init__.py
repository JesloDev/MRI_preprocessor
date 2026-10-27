"""
Preset system (PR10).

Presets are NAMED CONFIGURATIONS, not processing logic. A preset is
just a tuple of:

    (step_ids, step_params, skip_recommended, target_modalities)

that flows into the existing `resolve_pipeline` + `run_job` backend.
No step logic is duplicated. The same `Preset` schema is usable by:

    CLI ─────┐
            ├──→ resolve_pipeline(preset.step_ids, ...) ──→ run_job(...)
    Flask ──┤
            │
    GUI ────┘

This package contains:
    schema.py    - Preset pydantic model
    builtins.py  - The 4 built-in presets shipped with the platform
    storage.py   - JSON file persistence for custom user presets
    registry.py - PresetRegistry (in-memory + load built-ins + load customs)

The registry is the single entry point used by the CLI and the API.
"""
from app.presets.schema import Preset
from app.presets.registry import (
    PresetRegistry,
    get_preset,
    list_presets,
    register_preset,
    delete_preset,
)
from app.presets.builtins import BUILTIN_PRESETS

__all__ = [
    "Preset",
    "PresetRegistry",
    "BUILTIN_PRESETS",
    "get_preset",
    "list_presets",
    "register_preset",
    "delete_preset",
]
