"""
Dependency resolution: turns a user's raw selection of step ids into a
complete, ordered, valid pipeline -- inserting any missing prerequisite
steps automatically (per the platform spec: "This operation requires N
prerequisite steps. Add automatically?").

Tier model (PR3):
  - hard        : always inserted. The step cannot run without it.
                  (Legacy `depends_on` is treated as hard.)
  - recommended : inserted by default, but the caller can suppress
                  specific ones via `skip_recommended=[...]`. Useful for
                  "usually you want this, but you might have a reason to
                  skip" -- e.g. registration usually wants brain_extraction
                  first, but a user might pre-process and want to skip.
  - conditional : inserted only when the structured `when` condition
                  matches the requesting step's effective params. The
                  condition is a flat dict {key: expected_value} -- safe
                  to validate, easy to serialize for the UI. NOT a Python
                  expression.

API: `resolve_pipeline(selected_step_ids, step_params, skip_recommended)`.

Returns:
  {
    "requested":            [...],
    "ordered_pipeline":     [...],
    "added_dependencies":   [...],
    "tier_breakdown":       {
        "hard":        [...],
        "recommended": [...],
        "conditional": [...],
    },
    "skipped_recommended":  [...],  # what was explicitly skipped
  }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from app.pipeline.step_registry import Dependency, registry


class DependencyError(Exception):
    pass


# --------------------------------------------------------------------- #
# Effective-params computation
# --------------------------------------------------------------------- #

def _effective_params(
    step_id: str,
    step_params: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Returns the merged params for a step: its default_params overridden
    by any user-supplied params for that step.

    Used to evaluate `conditional` dependency `when` clauses against the
    requesting step's actual configuration. For example, if the user
    requested `spatial_normalization` with `atlas="MNI"`, and
    spatial_normalization has a conditional dependency on
    `mni_template_warp` with `when={"atlas": "MNI"}`, the condition
    matches and the warp step is auto-inserted.
    """
    step_def = registry.get(step_id)
    merged: Dict[str, Any] = dict(step_def.default_params)
    overrides = step_params.get(step_id, {})
    merged.update(overrides)
    return merged


# --------------------------------------------------------------------- #
# Tiered expansion
# --------------------------------------------------------------------- #

def expand_dependencies(
    selected_step_ids: List[str],
    step_params: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_recommended: Optional[List[str]] = None,
) -> List[Dependency]:
    """Returns the full list of Dependency objects required to run the
    user's selected steps, including transitive dependencies.

    Tier semantics:
      - hard       -> always included
      - recommended -> included unless its step_id is in skip_recommended
      - conditional -> included only if its `when` condition matches the
                       requesting step's effective params

    Returns a flat list (deduplicated) in no particular order; callers
    that need a topological order should pass the result through
    `topological_order(step_id for step_id in result)`.
    """
    step_params = step_params or {}
    skip_recommended = set(skip_recommended or [])

    # `resolved` holds the FINAL Dependency for each step id (after tier
    # precedence has been applied). We do NOT mark a step as "resolved"
    # until we've actually visited it from the stack -- that's what lets
    # transitive deps get expanded correctly.
    resolved: Dict[str, Dependency] = {}
    # `expanded` tracks which step ids have already had their own
    # dependencies walked, so we don't walk them twice (cycle guard).
    expanded: Set[str] = set()
    stack: List[str] = list(selected_step_ids)

    # Pre-mark the user-selected steps as hard tier in `resolved`. This
    # ensures they appear in the final list even if no other step
    # transitively depends on them (which is the normal case).
    for sid in selected_step_ids:
        resolved[sid] = Dependency(step_id=sid, tier="hard")

    while stack:
        step_id = stack.pop()
        if step_id in expanded:
            continue  # already walked this step's deps
        try:
            step_def = registry.get(step_id)
        except KeyError as e:
            raise DependencyError(str(e))
        expanded.add(step_id)

        # Effective params for THIS step, used to evaluate conditional
        # dependencies' `when` clauses.
        eff_params = _effective_params(step_id, step_params)

        for dep in step_def.all_dependencies():
            # Evaluate conditional dependencies' `when` against the
            # REQUESTING step's effective params.
            if dep.tier == "conditional":
                if not dep.matches(eff_params):
                    continue  # condition not met -> skip this dep

            # Evaluate recommended dependencies' skip list
            if dep.tier == "recommended" and dep.step_id in skip_recommended:
                continue  # user explicitly skipped

            # Hard dependencies: always include.

            # Tier precedence: hard wins over recommended wins over conditional.
            # If a step is requested by multiple steps at different tiers,
            # the strongest tier wins (so the prereq is treated as required,
            # not skippable).
            existing = resolved.get(dep.step_id)
            if existing is not None:
                tier_rank = {"hard": 3, "recommended": 2, "conditional": 1}
                if tier_rank.get(dep.tier, 0) > tier_rank.get(existing.tier, 0):
                    resolved[dep.step_id] = dep
                # else keep existing
            else:
                resolved[dep.step_id] = dep

            # Queue the dependency for its own transitive expansion.
            # We push even if it's already in `resolved`, because being
            # in `resolved` doesn't mean we've walked ITS deps yet --
            # only `expanded` tracks that.
            stack.append(dep.step_id)

    return list(resolved.values())


# --------------------------------------------------------------------- #
# Topological ordering (unchanged from PR1)
# --------------------------------------------------------------------- #

def topological_order(step_ids: List[str]) -> List[str]:
    """Kahn's algorithm; raises DependencyError on cycles.

    Visits dependencies declared via BOTH the legacy `depends_on` list
    and the new `dependencies` field. Tier doesn't affect ordering --
    a hard prereq and a recommended prereq both have to come before the
    step that needs them.
    """
    graph: Dict[str, Set[str]] = {sid: set() for sid in step_ids}
    in_degree: Dict[str, int] = {sid: 0 for sid in step_ids}

    for sid in step_ids:
        step_def = registry.get(sid)
        # Use all_dependencies() so both legacy depends_on and new
        # dependencies are walked.
        for dep in step_def.all_dependencies():
            dep_id = dep.step_id
            if dep_id in graph and dep_id not in graph[sid]:
                graph[sid].add(dep_id)  # sid depends on dep -> edge dep -> sid
                in_degree[sid] += 1

    dependents: Dict[str, List[str]] = {sid: [] for sid in step_ids}
    for sid, deps in graph.items():
        for dep in deps:
            dependents[dep].append(sid)

    ready = [sid for sid in step_ids if in_degree[sid] == 0]
    ordered: List[str] = []

    while ready:
        ready.sort()  # deterministic ordering
        current = ready.pop(0)
        ordered.append(current)
        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)

    if len(ordered) != len(step_ids):
        remaining = set(step_ids) - set(ordered)
        raise DependencyError(f"Cyclic dependency detected among steps: {remaining}")

    return ordered


# --------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------- #

def resolve_pipeline(
    selected_step_ids: List[str],
    step_params: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_recommended: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Main entry point used by the REST API and the engine.

    Args:
        selected_step_ids: the user's raw selection of step ids.
        step_params: per-step param overrides (used to evaluate
            conditional dependencies' `when` clauses against the
            requesting step's effective configuration). Optional --
            if omitted, conditional dependencies only fire when their
            `when` is empty.
        skip_recommended: list of recommended-dependency step ids to
            suppress. The user can pass this to skip "default-on but
            overridable" prerequisites.

    Returns:
        {
          "requested":            [...],   # what the user asked for
          "ordered_pipeline":     [...],   # final topological order
          "added_dependencies":   [...],   # auto-inserted (not in requested)
          "tier_breakdown":       {        # breakdown by tier
              "hard":        [...],
              "recommended": [...],
              "conditional": [...],
          },
          "skipped_recommended":  [...],   # what was explicitly skipped
        }
    """
    step_params = step_params or {}
    skip_recommended = skip_recommended or []

    deps = expand_dependencies(selected_step_ids, step_params, skip_recommended)
    all_step_ids = [d.step_id for d in deps]
    ordered = topological_order(all_step_ids)

    # Added = anything in the resolved set that wasn't in the user's request
    requested_set = set(selected_step_ids)
    added = sorted(s for s in all_step_ids if s not in requested_set)

    # Breakdown by tier -- but only for AUTO-INSERTED deps (not the
    # user's own selection). The user's selection is always "required"
    # by definition; tier breakdown is for the auto-inserted prereqs.
    breakdown = {"hard": [], "recommended": [], "conditional": []}
    for d in deps:
        if d.step_id in requested_set:
            continue  # user-requested, not a tiered auto-insert
        if d.tier in breakdown:
            breakdown[d.tier].append(d.step_id)
    # Sort each tier's list for deterministic API output
    for tier in breakdown:
        breakdown[tier] = sorted(breakdown[tier])

    return {
        "requested": selected_step_ids,
        "ordered_pipeline": ordered,
        "added_dependencies": added,
        "tier_breakdown": breakdown,
        "skipped_recommended": sorted(skip_recommended),
    }
