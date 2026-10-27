"""
Engine requirement resolver (PR18).

Given a preset (or a list of steps + step_params), this module figures
out which external engines are required and whether they're installed
and ready. This is the "pipeline → required engines → install missing"
flow that lets the user run a preset without manually figuring out
which engines to install.

Resolution logic:
    1. For each step in the selection:
       a. Check if the step supports engine selection (has "engine" in
          default_params).
       b. Determine the effective engine name: preset's step_params
          override the step's default_params.
       c. If the engine name is in the EngineManager's manifests, it's
          an EXTERNAL engine that needs to be installed.
       d. If not, it's a built-in engine (otsu, simpleitk, dipy,
          placeholder) that's always available.
    2. For each external engine, check:
       a. Is it installed in the store?
       b. Is it ready (health check passed)?
    3. Return a report:
       - required: all engines the pipeline needs
       - ready: installed + healthy
       - missing: not installed or broken
       - license_required: installed but needs a license file

Usage:
    # From a preset
    report = resolve_engines_for_preset(preset, engine_manager)

    # From explicit steps + params
    report = resolve_engines_for_steps(step_ids, step_params, engine_manager)

    # Install all missing engines
    for entry in report["missing"]:
        engine_manager.install_bundle(entry.engine_id, bundle_path, ...)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.pipeline.step_registry import registry


# Built-in engine names that don't need installation. These are pure
# Python implementations that ship with the platform. Any engine name
# NOT in this set AND present in the EngineManager's manifests is an
# external engine that needs to be installed.
BUILT_IN_ENGINES = {"otsu", "simpleitk", "dipy", "placeholder"}


def _is_external_engine(engine_name: str, engine_manager) -> bool:
    """Returns True if the engine name refers to an external engine
    (has a manifest, needs installation) vs a built-in (always available)."""
    if engine_name in BUILT_IN_ENGINES:
        return False
    return engine_name in engine_manager.manifests


def _get_step_engines(step_id: str, step_params: Dict[str, Any]) -> Optional[str]:
    """Returns the effective engine name for a step, or None if the
    step doesn't support engine selection.

    Resolution: step_params[step_id]["engine"] overrides the step's
    default_params["engine"].
    """
    try:
        step_def = registry.get(step_id)
    except KeyError:
        return None

    # Check if the step supports engine selection
    default_engine = step_def.default_params.get("engine")
    if default_engine is None:
        return None  # step doesn't support engine selection

    # Override from step_params
    overrides = step_params.get(step_id, {})
    return overrides.get("engine", default_engine)


def resolve_engines_for_steps(
    step_ids: List[str],
    step_params: Dict[str, Dict[str, Any]],
    engine_manager,
) -> Dict[str, Any]:
    """Resolves required engines for a list of steps + params.

    Returns:
        {
            "required_engines": [
                {
                    "step_id": str,
                    "engine_name": str,
                    "is_external": bool,
                    "manifest": Optional[EngineManifest],
                    "installed": bool,
                    "status": str,  # "ready" | "not_installed" | "broken" | "license_required"
                    "install_path": Optional[str],
                },
                ...
            ],
            "ready": [...],      # engine_ids that are ready
            "missing": [...],    # engine_ids not installed
            "license_required": [...],  # installed but needs license
            "all_ready": bool,    # True if all required engines are ready
        }
    """
    from app.engines.health import NOT_INSTALLED, READY, BROKEN, LICENSE_REQUIRED

    required = []
    ready_ids = []
    missing_ids = []
    license_required_ids = []
    seen_engines = set()  # deduplicate

    for step_id in step_ids:
        engine_name = _get_step_engines(step_id, step_params)
        if engine_name is None:
            continue  # step doesn't support engine selection

        is_external = _is_external_engine(engine_name, engine_manager)

        entry = {
            "step_id": step_id,
            "engine_name": engine_name,
            "is_external": is_external,
            "manifest": None,
            "installed": False,
            "status": READY if not is_external else NOT_INSTALLED,
            "install_path": None,
        }

        if is_external:
            manifest = engine_manager.manifests.get(engine_name)
            entry["manifest"] = manifest

            # Check health (without running the binary -- fast check)
            status = engine_manager.get_status(engine_name)
            entry["status"] = status
            entry["installed"] = engine_manager.is_installed(engine_name)
            entry["install_path"] = engine_manager.get_install_path(engine_name)

            if status == READY:
                if engine_name not in seen_engines:
                    ready_ids.append(engine_name)
                    seen_engines.add(engine_name)
            elif status == LICENSE_REQUIRED:
                if engine_name not in seen_engines:
                    license_required_ids.append(engine_name)
                    seen_engines.add(engine_name)
            else:  # NOT_INSTALLED or BROKEN
                if engine_name not in seen_engines:
                    missing_ids.append(engine_name)
                    seen_engines.add(engine_name)
        else:
            # Built-in engine -- always ready
            if engine_name not in seen_engines:
                ready_ids.append(engine_name)
                seen_engines.add(engine_name)

        required.append(entry)

    return {
        "required_engines": required,
        "ready": sorted(ready_ids),
        "missing": sorted(missing_ids),
        "license_required": sorted(license_required_ids),
        "all_ready": len(missing_ids) == 0 and len(license_required_ids) == 0,
    }


def resolve_engines_for_preset(preset, engine_manager) -> Dict[str, Any]:
    """Resolves required engines for a preset.

    Uses the preset's step_ids + step_params (merged with the caller's
    overrides) to determine which engines the pipeline needs.
    """
    return resolve_engines_for_steps(
        step_ids=list(preset.step_ids),
        step_params=dict(preset.step_params),
        engine_manager=engine_manager,
    )


def format_resolution_report(report: Dict[str, Any]) -> str:
    """Returns a human-readable multi-line string of the resolution report.

    Used by the CLI to show:
        Required engines:
          ✓ otsu (built-in)
          ✗ synthstrip (not installed)
          ⚠ mri_deface (license required)
    """
    from app.engines.health import status_display

    lines = ["Required engines:"]
    for entry in report["required_engines"]:
        engine = entry["engine_name"]
        if entry["is_external"]:
            status = entry["status"]
            display = status_display(status)
            manifest = entry.get("manifest")
            provider = manifest.provider if manifest else ""
            lines.append(f"  {display}  {engine} ({provider})")
        else:
            lines.append(f"  ✓ Available (built-in)  {engine}")

    if report["missing"]:
        lines.append("")
        lines.append(f"Missing: {', '.join(report['missing'])}")
    if report["license_required"]:
        lines.append("")
        lines.append(f"License required: {', '.join(report['license_required'])}")
    if report["all_ready"]:
        lines.append("")
        lines.append("All required engines are ready.")

    return "\n".join(lines)
