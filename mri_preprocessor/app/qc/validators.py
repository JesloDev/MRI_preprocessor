"""
Basic QC / validation layer: confirms every discovered file is actually
a loadable, sane NIfTI volume before it enters the pipeline. This is
what populates "Valid files / Failed files" on the main screen.
"""
from typing import Dict, List

import nibabel as nib
import numpy as np


def validate_file(filepath: str) -> Dict[str, object]:
    result = {"filepath": filepath, "valid": False, "error": None, "shape": None}
    try:
        img = nib.load(filepath)
        data_shape = img.shape
        # Force a minimal read to catch corrupt/truncated files early.
        _ = img.dataobj[tuple(slice(0, 1) for _ in data_shape)]
        if any(dim <= 0 for dim in data_shape):
            raise ValueError(f"Invalid volume shape: {data_shape}")
        result["valid"] = True
        result["shape"] = list(data_shape)
    except Exception as exc:  # noqa: BLE001 - we want to catch and report anything
        result["error"] = str(exc)
    return result


def validate_manifest(manifest: List[Dict[str, str]]) -> Dict[str, object]:
    results = [validate_file(entry["filepath"]) for entry in manifest]
    valid = [r for r in results if r["valid"]]
    failed = [r for r in results if not r["valid"]]
    return {
        "total": len(results),
        "valid_count": len(valid),
        "failed_count": len(failed),
        "failed_files": failed,
    }
