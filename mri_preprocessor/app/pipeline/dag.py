"""
DAG execution: runs an ordered pipeline of steps against a single input
file for a single subject, feeding each step's output into the next,
and recording provenance for every step.
"""
import os
from typing import Any, Dict, List

from app.pipeline.step_registry import registry
from app.provenance.tracker import ProvenanceTracker


def run_pipeline_for_file(
    subject_id: str,
    modality: str,
    input_path: str,
    ordered_step_ids: List[str],
    step_params: Dict[str, Dict[str, Any]],
    work_dir: str,
    context: Dict[str, Any],
    provenance: ProvenanceTracker,
) -> Dict[str, Any]:
    """
    Executes ordered_step_ids in sequence for one file.

    step_params: {step_id: {param_name: value}} -- overrides for that step's
                 default_params (from step_registry).
    context:     shared cross-step info (reference image, bvals/bvecs, etc).
                 Steps may add to this dict (e.g. brain_extraction adds
                 'brain_mask') for downstream steps to consume.
    """
    os.makedirs(work_dir, exist_ok=True)
    current_path = input_path
    step_outputs: Dict[str, Any] = {}

    applicable_steps = [
        sid for sid in ordered_step_ids
        if modality in registry.get(sid).modalities
    ]

    for step_id in applicable_steps:
        step_def = registry.get(step_id)
        fn = registry.get_fn(step_id)
        params = {**step_def.default_params, **step_params.get(step_id, {})}

        step_dir = os.path.join(work_dir, step_id)
        os.makedirs(step_dir, exist_ok=True)

        result = fn(current_path, step_dir, params, context)

        provenance.log_step(
            subject_id=subject_id,
            modality=modality,
            step_id=step_id,
            engine=step_def.engine,
            input_path=current_path,
            output_path=result["output_path"],
            params=params,
            metrics=result.get("metrics", {}),
        )

        # make any produced side outputs (masks, transforms, FA/MD maps...)
        # available to later steps via context
        context.update(result.get("outputs", {}))
        step_outputs[step_id] = result

        current_path = result["output_path"]

    return {"final_output": current_path, "step_outputs": step_outputs}
