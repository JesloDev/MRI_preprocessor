import os

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_engine_manager():
    """Resets the EngineManager singleton between tests so env var
    changes ($MRI_PREPROCESSOR_ENGINES_DIR) take effect per-test.

    Without this, the singleton would be constructed once (at first
    access) and its store root would be pinned for the rest of the
    test session, causing test isolation bugs.
    """
    from app.engines.manager import _reset_engine_manager
    _reset_engine_manager()
    yield
    _reset_engine_manager()


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Builds a tiny synthetic dataset: 2 subjects, T1 + fMRI each."""
    input_dir = tmp_path / "input"
    for subject in ["sub-001", "sub-002"]:
        sdir = input_dir / subject
        sdir.mkdir(parents=True)

        t1_data = np.random.rand(16, 16, 16).astype(np.float32) * 100
        t1_img = nib.Nifti1Image(t1_data, affine=np.eye(4))
        nib.save(t1_img, str(sdir / f"{subject}_T1.nii.gz"))

        fmri_data = np.random.rand(16, 16, 16, 10).astype(np.float32) * 100
        fmri_img = nib.Nifti1Image(fmri_data, affine=np.eye(4))
        nib.save(fmri_img, str(sdir / f"{subject}_task-rest_bold.nii.gz"))

    output_dir = tmp_path / "output"
    return {"input_dir": str(input_dir), "output_dir": str(output_dir)}
