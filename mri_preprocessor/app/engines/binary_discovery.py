"""
Binary discovery for external engines.

External engines (SynthStrip, HD-BET, MCFLIRT, FSL eddy) all need to
locate an external binary before they can run. The discovery chain is:

    0. EngineManager.get_binary_path(engine_id, binary_name) -- if the
       engine is installed via the EngineManager (PR13), use the local
       install. This is the highest-priority path: a managed install
       always wins over whatever happens to be on PATH.
    1. Explicit parameter:   params["binary_path"]
    2. Engine-specific env:  $MRI_PREPROCESSOR_<ENGINE_NAME>
    3. PATH:                 shutil.which(binary_name)
    4. Tool-suite fallback:  $<TOOL_SUITE_HOME>/bin/<binary_name>

This module provides one function, `find_binary`, that runs the chain
and returns either the discovered path or None (with a reason string
listing what was checked).

Keeping this in one place means: adding a new external engine is just
"give it a discovery config" -- no copy-pasted shell-which logic in
every engine file.

PR13 change: the chain now checks the EngineManager's local store
BEFORE env vars / PATH. This means: if a user installs SynthStrip via
`python -m app.cli engines install synthstrip`, the discovery chain
will find it in the local store, even if FreeSurfer isn't installed
on the host system. The user never needs to set $FREESURFER_HOME or
touch their PATH -- the platform manages the install.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DiscoveryConfig:
    """Configuration for one engine's binary discovery.

    Fields:
        engine_id:    the manifest id (e.g. "synthstrip"). Used to look
                      up the engine in the EngineManager's local store.
                      If None, the store-check step is skipped.
        binary_name:  the executable to find (e.g. "mri_synthstrip").
        env_var:      engine-specific env var that may point at the
                      binary directly (e.g. MRI_PREPROCESSOR_SYNTHSTRIP).
        tool_suite_env: optional env var pointing at the tool's install
                          dir (e.g. FREESURFER_HOME). We then check
                          $<tool_suite_env>/bin/<binary_name>.
    """
    engine_id: Optional[str]
    binary_name: str
    env_var: str
    tool_suite_env: Optional[str] = None


def find_binary(
    config: DiscoveryConfig,
    explicit_path: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Runs the discovery chain for one engine's binary.

    Args:
        config: the engine's DiscoveryConfig.
        explicit_path: optional override from params["binary_path"].

    Returns:
        (path_or_None, reason)
        - If found: (path, "")
        - If not found: (None, human-readable message listing what was
          checked and how to fix)
    """
    checked: List[str] = []

    # 0. EngineManager local store (PR13). If the engine was installed
    #    via `python -m app.cli engines install <id>`, the binary is in
    #    <store_root>/<id>/<version>/bin/<binary_name>. This check
    #    runs FIRST so a managed install always wins over PATH.
    if config.engine_id is not None:
        try:
            # Lazy import to avoid circular deps at module load time.
            from app.engines.manager import get_engine_manager
            local_path = get_engine_manager().get_binary_path(
                config.engine_id, config.binary_name,
            )
            if local_path and os.path.isfile(local_path) \
               and os.access(local_path, os.X_OK):
                return local_path, ""
            checked.append(
                f"local engine store ({config.engine_id} not installed)"
            )
        except Exception:  # noqa: BLE001 -- store errors shouldn't crash discovery
            checked.append("local engine store (lookup failed)")

    # 1. Explicit parameter (user override)
    if explicit_path:
        if os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK):
            return explicit_path, ""
        checked.append(f"--param ...binary_path={explicit_path!r} (not executable)")

    # 2. Engine-specific env var
    env_path = os.environ.get(config.env_var)
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path, ""
        checked.append(f"${config.env_var}={env_path!r} (not executable)")

    # 3. PATH lookup
    path_lookup = shutil.which(config.binary_name)
    if path_lookup:
        return path_lookup, ""
    checked.append(f"PATH ({config.binary_name} not found)")

    # 4. Tool-suite fallback (e.g. $FREESURFER_HOME/bin/mri_synthstrip)
    if config.tool_suite_env:
        suite_home = os.environ.get(config.tool_suite_env)
        if suite_home:
            candidate = os.path.join(suite_home, "bin", config.binary_name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate, ""
            checked.append(f"${config.tool_suite_env}/bin/{config.binary_name} (not found)")
        else:
            checked.append(f"${config.tool_suite_env} (env var not set)")

    reason = (
        f"{config.binary_name} not found. Checked: "
        + "; ".join(checked)
        + f". To fix: install the tool via `python -m app.cli engines "
        f"install {config.engine_id or 'unknown'}`, set ${config.env_var} "
        f"to the binary path, or pass --param <step>.binary_path=<path>."
    )
    return None, reason
