"""
Preset registry (PR10).

The single entry point used by the CLI and the Flask API. Layers
the 4 built-in presets (always present, defined in code) on top of
the user's saved custom presets (JSON files on disk).

The registry is intentionally stateless on the built-in side: every
call to list_presets() / get_preset() re-reads the on-disk custom
presets. This means a preset saved from one CLI invocation is
immediately visible to a subsequent API call -- no cache invalidation
needed.

Module-level singleton functions (get_preset, list_presets,
register_preset, delete_preset) wrap a single PresetRegistry instance
so the CLI and API don't need to pass it around.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from app.presets.builtins import BUILTIN_PRESETS, get_builtin_preset
from app.presets.schema import Preset
from app.presets.storage import (
    delete_preset_file,
    list_saved_preset_names,
    load_preset,
    save_preset,
)


class PresetRegistry:
    """Layers built-in presets on top of user-saved custom presets.

    The registry has two responsibilities:
      1. Always-present built-in presets (defined in code).
      2. Persisting custom presets to disk and reloading them on
         demand.

    The two layers are independent -- a user can delete their custom
    presets, but built-ins are always available.
    """

    def __init__(self) -> None:
        # Built-ins live in a dict for O(1) lookup by name.
        self._builtins: Dict[str, Preset] = {
            p.name: p for p in BUILTIN_PRESETS
        }

    def list_presets(self) -> List[Preset]:
        """Returns all presets: built-ins first (in declared order),
        then customs (sorted by name)."""
        result: List[Preset] = list(self._builtins.values())

        # Load customs from disk. We do this fresh on every call so
        # presets saved by another process (e.g. the API) become
        # visible without a restart.
        for name in list_saved_preset_names():
            preset = load_preset(name)
            if preset is None:
                # Corrupt file -- skip. Don't crash the registry.
                continue
            # Don't shadow a built-in with a saved custom (shouldn't
            # happen because register_preset rejects built-in names,
            # but defensive here anyway).
            if preset.name in self._builtins:
                continue
            result.append(preset)

        # Sort customs by name (built-ins are already in declared order
        # at the front)
        customs = sorted([p for p in result if not p.is_builtin],
                          key=lambda p: p.name)
        return [p for p in result if p.is_builtin] + customs

    def get_preset(self, name: str) -> Optional[Preset]:
        """Returns a preset by name. Built-ins take precedence over
        customs (a custom preset can't shadow a built-in -- this is
        enforced by register_preset at save time)."""
        builtin = self._builtins.get(name)
        if builtin is not None:
            return builtin
        return load_preset(name)

    def register_preset(self, preset: Preset, overwrite: bool = False) -> Preset:
        """Saves a custom preset to disk.

        Raises ValueError if:
          - The preset name collides with a built-in (built-ins can't
            be overwritten -- the user must pick a different name).
          - A custom with the same name already exists and
            `overwrite=False`.

        Returns the saved Preset (with created_at set to now if it
        was a new preset, or preserved from the existing one if it's
        an overwrite).
        """
        if preset.name in self._builtins:
            raise ValueError(
                f"Cannot save preset {preset.name!r}: that name is "
                f"reserved for a built-in preset. Pick a different name."
            )

        if not overwrite:
            existing = load_preset(preset.name)
            if existing is not None:
                raise ValueError(
                    f"A custom preset named {preset.name!r} already "
                    f"exists. Pass overwrite=True to replace it, or "
                    f"delete it first with `presets delete {preset.name}`."
                )

        # Custom presets are always saved with is_builtin=False,
        # regardless of what the caller passed. This is a defense
        # against the on-disk file being mistaken for a built-in on
        # load.
        to_save = preset.model_copy(update={"is_builtin": False})
        save_preset(to_save)
        return to_save

    def delete_preset(self, name: str) -> bool:
        """Deletes a custom preset from disk. Returns True if deleted,
        False if it didn't exist.

        Raises ValueError if the name refers to a built-in preset
        (built-ins can't be deleted -- they're defined in code).
        """
        if name in self._builtins:
            raise ValueError(
                f"Cannot delete preset {name!r}: it's a built-in "
                f"preset and can't be removed."
            )
        return delete_preset_file(name)


# --------------------------------------------------------------------- #
# Module-level singleton (used by CLI and API without passing the
# registry around)
# --------------------------------------------------------------------- #

_registry = PresetRegistry()


def list_presets() -> List[Preset]:
    return _registry.list_presets()


def get_preset(name: str) -> Optional[Preset]:
    return _registry.get_preset(name)


def register_preset(preset: Preset, overwrite: bool = False) -> Preset:
    return _registry.register_preset(preset, overwrite=overwrite)


def delete_preset(name: str) -> bool:
    return _registry.delete_preset(name)
