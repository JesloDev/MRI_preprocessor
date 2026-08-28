"""
Dependency resolution: turns a user's raw selection of step ids into a
complete, ordered, valid pipeline -- inserting any missing prerequisite
steps automatically (per the platform spec: "This operation requires N
prerequisite steps. Add automatically?").
"""
from __future__ import annotations

from typing import Dict, List, Set

from app.pipeline.step_registry import registry


class DependencyError(Exception):
    pass


def expand_dependencies(selected_step_ids: List[str]) -> List[str]:
    """Returns the full set of step ids required, including transitive deps."""
    resolved: Set[str] = set()
    stack = list(selected_step_ids)

    while stack:
        step_id = stack.pop()
        if step_id in resolved:
            continue
        try:
            step_def = registry.get(step_id)
        except KeyError as e:
            raise DependencyError(str(e))
        resolved.add(step_id)
        for dep in step_def.depends_on:
            if dep not in resolved:
                stack.append(dep)

    return list(resolved)


def topological_order(step_ids: List[str]) -> List[str]:
    """Kahn's algorithm; raises DependencyError on cycles."""
    graph: Dict[str, Set[str]] = {sid: set() for sid in step_ids}
    in_degree: Dict[str, int] = {sid: 0 for sid in step_ids}

    for sid in step_ids:
        step_def = registry.get(sid)
        for dep in step_def.depends_on:
            if dep in graph and dep not in graph[sid]:
                graph[sid].add(dep)  # sid depends on dep -> edge dep -> sid
                in_degree[sid] += 1

    # Build reverse adjacency: dep -> [steps that depend on it]
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


def resolve_pipeline(selected_step_ids: List[str]) -> Dict[str, object]:
    """
    Main entry point used by the REST API.

    Returns:
      {
        "requested": [...],
        "added_dependencies": [...],   # steps auto-inserted
        "ordered_pipeline": [...],     # final execution order
      }
    """
    full_set = expand_dependencies(selected_step_ids)
    ordered = topological_order(full_set)
    added = [s for s in full_set if s not in selected_step_ids]

    return {
        "requested": selected_step_ids,
        "added_dependencies": sorted(added),
        "ordered_pipeline": ordered,
    }
