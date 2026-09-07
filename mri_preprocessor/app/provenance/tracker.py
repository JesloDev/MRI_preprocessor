"""
Provenance tracking: every processing step applied to every file is
logged in a machine-readable, append-only JSON-lines file so anyone can
later answer "exactly how was this image processed?" without relying on
memory.

One provenance log per job, stored under
  <output_dir>/provenance/<job_id>.jsonl
"""
import json
import os
import time
from typing import Any, Dict


class ProvenanceTracker:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_step(
        self,
        subject_id: str,
        modality: str,
        step_id: str,
        engine: str,
        input_path: str,
        output_path: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        record = {
            "timestamp": time.time(),
            "subject_id": subject_id,
            "modality": modality,
            "step_id": step_id,
            "engine": engine,
            "input_path": input_path,
            "output_path": output_path,
            "params": params,
            "metrics": metrics,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_event(self, event: str, **extra: Any) -> None:
        record = {"timestamp": time.time(), "event": event, **extra}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def history_for_file(self, original_input_path: str) -> list:
        """Reconstructs the processing chain that led from an original
        input file to its final derivative, in order."""
        all_records = [r for r in self.read_all() if "step_id" in r]
        chain = []
        current_targets = {original_input_path}
        changed = True
        while changed:
            changed = False
            for r in all_records:
                if r["input_path"] in current_targets and r not in chain:
                    chain.append(r)
                    current_targets.add(r["output_path"])
                    changed = True
        chain.sort(key=lambda r: r["timestamp"])
        return chain
