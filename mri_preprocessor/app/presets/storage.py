"""
JSON file persistence for custom user presets (PR10).

Built-in presets live in code (app/presets/builtins.py) and always
reflect the current platform state. Custom presets created by the user
via `python -m app.cli presets create ...` are persisted to disk so
they survive CLI restarts and are usable by the Flask API and future
GUI without re-creation.

Storage location (priority):
    1. $MRI_PREPROCESSOR_PRESETS_DIR env var (explicit override --
       useful for testing and for shared deployments)
    2. ~/.mripreprocessor/presets/ (default per-user location)

Each preset is one JSON file: <presets_dir>/<name>.json. The filename
matches the preset's name, with unsafe characters sanitized.

The storage layer is intentionally minimal:
    - save(preset) -> writes <name>.json
    - load(name)   -> returns Preset or None
    - delete(name) -> removes <name>.json
    - list_saved() -> returns names of all saved presets

The PresetRegistry (registry.py) layers built-ins on top of these
saved files, providing a unified view to the CLI and API.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

from app.presets.schema import Preset


# --------------------------------------------------------------------- #
# Storage directory resolution
# --------------------------------------------------------------------- #

ENV_VAR = "MRI_PREPROCESSOR_PRESETS_DIR"


def get_presets_dir() -> str:
    """Returns the directory where custom presets are stored, creating
    it if it doesn't exist.

    Resolution order:
        1. $MRI_PREPROCESSOR_PRESETS_DIR (explicit env var)
        2. ~/.mripreprocessor/presets/    (default per-user location)
    """
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        presets_dir = os.path.expanduser(env_path)
    else:
        home = os.path.expanduser("~")
        presets_dir = os.path.join(home, ".mripreprocessor", "presets")

    os.makedirs(presets_dir, exist_ok=True)
    return presets_dir


# --------------------------------------------------------------------- #
# Filename sanitization
# --------------------------------------------------------------------- #

def _sanitize_filename(name: str) -> str:
    """Sanitizes a preset name into a safe filename component.

    Allows alphanumerics, hyphens, underscores, dots. Replaces others
    with underscores. This is a security measure against path
    traversal -- a malicious name like "../../etc/passwd" becomes
    "_____etc_passwd", which is a safe filename.
    """
    # Replace path separators and other unsafe chars with underscores
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    # Strip leading dots (so we don't create hidden files)
    safe = safe.lstrip(".")
    if not safe:
        raise ValueError(f"Preset name {name!r} produces an empty filename")
    return safe + ".json"


# --------------------------------------------------------------------- #
# CRUD operations
# --------------------------------------------------------------------- #

def save_preset(preset: Preset) -> str:
    """Persists a preset to disk. Returns the path written to.

    Overwrites any existing preset with the same name. The caller is
    responsible for confirming overwrite intent at the CLI layer.
    """
    presets_dir = get_presets_dir()
    filename = _sanitize_filename(preset.name)
    path = os.path.join(presets_dir, filename)

    # We use model_dump() so the on-disk format is the full schema
    # (including is_builtin and created_at). When loading, we re-read
    # these so the in-memory representation matches the saved state.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preset.model_dump(), f, indent=2, default=str)
    return path


def load_preset(name: str) -> Optional[Preset]:
    """Loads a preset from disk by name. Returns None if not found."""
    presets_dir = get_presets_dir()
    filename = _sanitize_filename(name)
    path = os.path.join(presets_dir, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Preset(**data)
    except (json.JSONDecodeError, ValueError):
        # A corrupt preset file shouldn't crash the platform. The
        # registry will simply skip it. The user can delete it via
        # `presets delete <name>` (which uses the file path, not the
        # Preset object, so it doesn't need the file to load).
        return None


def delete_preset_file(name: str) -> bool:
    """Removes a preset file from disk. Returns True if deleted, False
    if the file didn't exist."""
    presets_dir = get_presets_dir()
    filename = _sanitize_filename(name)
    path = os.path.join(presets_dir, filename)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    return True


def list_saved_preset_names() -> List[str]:
    """Returns the names of all saved custom presets on disk."""
    presets_dir = get_presets_dir()
    if not os.path.isdir(presets_dir):
        return []
    names = []
    for filename in sorted(os.listdir(presets_dir)):
        if filename.endswith(".json"):
            # Strip .json to get the name
            name = filename[:-5]
            names.append(name)
    return names
