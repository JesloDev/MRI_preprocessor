"""
Tests for PR11: parameter schema + new privacy/filtering steps.

Covers:
  - ParameterSpec schema (defaults, allowed_values, serialization)
  - ProcessingStep.parameters field is exposed via /api/capabilities
  - @step decorator accepts parameters=[ParameterSpec(...)]
  - Resampling accepts 5-D shape=(batch, x, y, z, channels) parameter
    AND legacy 3-D voxel_size
  - Intensity normalization extended with input_range, output_range,
    clip_min, clip_max, robust method
  - Intensity filtering (new step): high_pass / low_pass / band_pass
  - Defacing step + engine family (placeholder + pydeface + mri_deface stubs)
  - Anonymization step (real nibabel-based, no external tool)
  - Privacy built-in preset (anonymization + defacing)
  - CLI parse_params accepts both JSON and comma-separated value forms
"""
import json
import os
import shutil

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app, parse_params
from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.step_registry import (
    ParameterSpec,
    ProcessingStep,
    registry,
)


# --------------------------------------------------------------------- #
# ParameterSpec schema
# --------------------------------------------------------------------- #

def test_parameter_spec_minimal():
    """A ParameterSpec with just name + type has sensible defaults."""
    p = ParameterSpec(name="x", type="int")
    assert p.name == "x"
    assert p.type == "int"
    assert p.default is None
    assert p.allowed_values is None
    assert p.description == ""
    assert p.required is False


def test_parameter_spec_full():
    p = ParameterSpec(
        name="engine", type="str",
        default="otsu",
        allowed_values=["otsu", "synthstrip"],
        description="Which engine to use",
        required=False,
    )
    assert p.name == "engine"
    assert p.type == "str"
    assert p.default == "otsu"
    assert p.allowed_values == ["otsu", "synthstrip"]
    assert p.description == "Which engine to use"


def test_parameter_spec_serialized_in_model_dump():
    """ParameterSpec is preserved through ProcessingStep.model_dump()."""
    p = ProcessingStep(
        id="test_param_dump", name="Test", modalities=["T1"], engine="test",
        parameters=[
            ParameterSpec(name="foo", type="int", default=5),
            ParameterSpec(name="bar", type="str", default="baz"),
        ],
    )
    d = p.model_dump()
    assert "parameters" in d
    assert len(d["parameters"]) == 2
    assert d["parameters"][0]["name"] == "foo"
    assert d["parameters"][0]["default"] == 5


def test_step_decorator_accepts_parameters():
    """The @step decorator accepts the new parameters=[] kwarg."""
    from app.pipeline.step_registry import step

    @step(
        id="test_param_step",
        name="Test Param Step",
        modalities=["T1"],
        engine="test",
        parameters=[
            ParameterSpec(name="engine", type="str",
                          allowed_values=["a", "b"], default="a"),
        ],
    )
    def my_test_step(input_path, output_dir, params, context):
        return {"output_path": input_path, "outputs": {}, "metrics": {}}

    s = registry.get("test_param_step")
    assert len(s.parameters) == 1
    assert s.parameters[0].name == "engine"
    assert s.parameters[0].allowed_values == ["a", "b"]


# --------------------------------------------------------------------- #
# API exposes parameter schemas
# --------------------------------------------------------------------- #

def test_api_capabilities_returns_parameter_schemas():
    """/api/capabilities surfaces the `parameters` field per step."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/capabilities")
        data = r.get_json()
        # resampling has parameters declared
        resampling = data["resampling"]
        assert "parameters" in resampling
        assert len(resampling["parameters"]) >= 3  # voxel_size, shape, interpolation
        # Find the shape parameter
        shape_param = next(
            (p for p in resampling["parameters"] if p["name"] == "shape"),
            None,
        )
        assert shape_param is not None
        assert shape_param["type"] == "tuple"


def test_api_capabilities_for_modality_includes_new_steps():
    """/api/capabilities/T1 now includes defacing, anonymization, intensity_filtering."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/capabilities/T1")
        data = r.get_json()
        step_ids = {s["id"] for s in data}
        # PR11 additions
        assert "intensity_filtering" in step_ids
        assert "defacing" in step_ids
        assert "anonymization" in step_ids


# --------------------------------------------------------------------- #
# Resampling 5-D shape parameter
# --------------------------------------------------------------------- #

def test_resampling_step_has_shape_parameter():
    """The resampling step declares a 5-D shape parameter in its schema."""
    s = registry.get("resampling")
    shape_param = next(
        (p for p in s.parameters if p.name == "shape"),
        None,
    )
    assert shape_param is not None
    assert shape_param.type == "tuple"


def test_resampling_with_5d_shape_produces_target_size(tmp_path):
    """resampling.shape=(1, 16, 16, 16, 1) produces a 16x16x16 output
    volume (batch and channel are ML conventions; spatial dims are
    resampled)."""
    # Build a 32x32x32 input
    in_path = tmp_path / "in.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(32, 32, 32).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(in_path),
    )

    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("resampling")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"shape": [1, 16, 16, 16, 1]},
        context={},
    )
    out_img = nib.load(result["output_path"])
    assert out_img.shape[:3] == (16, 16, 16)
    # ml_shape metadata should be preserved
    assert result["metrics"]["ml_shape"] == [1, 16, 16, 16, 1]


def test_resampling_legacy_voxel_size_still_works(tmp_path):
    """The legacy voxel_size parameter still works when shape is not given."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(in_path),
    )
    out_dir = str(tmp_path / "out_legacy")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("resampling")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"voxel_size": [2.0, 2.0, 2.0]},
        context={},
    )
    out_img = nib.load(result["output_path"])
    # 16 / 2 = 8
    assert out_img.shape[:3] == (8, 8, 8)


def test_resampling_shape_takes_precedence_over_voxel_size(tmp_path):
    """When both shape and voxel_size are given, shape wins."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(32, 32, 32).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(in_path),
    )
    out_dir = str(tmp_path / "out_both")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("resampling")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"shape": [1, 16, 16, 16, 1], "voxel_size": [2.0, 2.0, 2.0]},
        context={},
    )
    # Shape wins -> 16x16x16 (not the 16x16x16 we'd get from voxel_size
    # either, but the ml_shape metadata is set)
    out_img = nib.load(result["output_path"])
    assert out_img.shape[:3] == (16, 16, 16)
    assert "ml_shape" in result["metrics"]


def test_resampling_rejects_invalid_shape(tmp_path):
    """shape with wrong arity raises ValueError."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(in_path),
    )
    fn = registry.get_fn("resampling")
    with pytest.raises(ValueError):
        fn(input_path=str(in_path), output_dir=str(tmp_path),
           params={"shape": [1, 16, 16, 16]},  # 4-tuple, not 5
           context={})


# --------------------------------------------------------------------- #
# Intensity normalization extended
# --------------------------------------------------------------------- #

def test_intensity_normalization_has_extended_params():
    """The intensity_normalization step declares the extended parameters."""
    s = registry.get("intensity_normalization")
    param_names = {p.name for p in s.parameters}
    assert {"method", "input_range", "output_range",
            "clip_min", "clip_max"} <= param_names


def test_intensity_normalization_minmax_with_input_range(tmp_path):
    """method=minmax with input_range=(0, 100) and output_range=(0, 1)
    produces values in [0, 1]."""
    in_path = tmp_path / "in.nii.gz"
    # Random data in [0, 100]
    data = np.random.rand(8, 8, 8).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("intensity_normalization")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={
            "method": "minmax",
            "input_range": [0, 100],
            "output_range": [0, 1],
        },
        context={},
    )
    out = nib.load(result["output_path"]).get_fdata()
    # All voxels (including background) should be in [0, 1] (or close)
    assert out.min() >= 0 - 1e-5
    assert out.max() <= 1 + 1e-5


def test_intensity_normalization_robust_method(tmp_path):
    """method=robust subtracts median and divides by IQR."""
    in_path = tmp_path / "in.nii.gz"
    data = np.random.rand(8, 8, 8).astype(np.float32) * 100 + 50
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out_robust")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("intensity_normalization")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"method": "robust"},
        context={},
    )
    out = nib.load(result["output_path"]).get_fdata()
    # Foreground (non-zero) voxels should be roughly centered around 0
    fg = out[data > 0]
    assert abs(np.median(fg)) < 5.0  # median should be near 0


def test_intensity_normalization_clip_applied(tmp_path):
    """clip_min and clip_max are applied after normalization."""
    in_path = tmp_path / "in.nii.gz"
    data = np.random.rand(8, 8, 8).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out_clip")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("intensity_normalization")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={
            "method": "minmax",
            "input_range": [0, 100],
            "output_range": [0, 1],
            "clip_min": 0.2,
            "clip_max": 0.8,
        },
        context={},
    )
    out = nib.load(result["output_path"]).get_fdata()
    assert out.min() >= 0.2 - 1e-5
    assert out.max() <= 0.8 + 1e-5


def test_intensity_normalization_unknown_method_raises(tmp_path):
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    fn = registry.get_fn("intensity_normalization")
    with pytest.raises(ValueError):
        fn(input_path=str(in_path), output_dir=str(tmp_path),
           params={"method": "nonexistent_method"}, context={})


# --------------------------------------------------------------------- #
# Intensity filtering (new step)
# --------------------------------------------------------------------- #

def test_intensity_filtering_step_registered():
    s = registry.get("intensity_filtering")
    assert s.maturity == "reference"
    assert "high_pass" in {p.name for p in s.parameters}


def test_intensity_filtering_low_pass_smoothes(tmp_path):
    """A low-pass filter should reduce high-frequency content (smoothing)."""
    in_path = tmp_path / "in.nii.gz"
    data = np.random.rand(16, 16, 16).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out_lp")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("intensity_filtering")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"low_pass": 0.2},
        context={},
    )
    out = nib.load(result["output_path"]).get_fdata()
    # The output should be smoother than the input (lower std on the
    # finite-difference magnitude)
    in_grad = np.gradient(nib.load(str(in_path)).get_fdata())
    out_grad = np.gradient(out)
    in_grad_mag = np.sqrt(sum(g ** 2 for g in in_grad))
    out_grad_mag = np.sqrt(sum(g ** 2 for g in out_grad))
    assert out_grad_mag.std() < in_grad_mag.std()


def test_intensity_filtering_high_pass_preserves_edges(tmp_path):
    """A high-pass filter should preserve fine details."""
    in_path = tmp_path / "in.nii.gz"
    data = np.random.rand(16, 16, 16).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out_hp")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("intensity_filtering")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"high_pass": 0.1},
        context={},
    )
    # Should run without error and produce a high_pass-labeled output
    assert os.path.isfile(result["output_path"])
    assert result["metrics"]["filter_type"] == "high_pass"


def test_intensity_filtering_band_pass_combines_both(tmp_path):
    """Setting both high_pass and low_pass produces a band_pass filter."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out_bp")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("intensity_filtering")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"high_pass": 0.1, "low_pass": 0.4},
        context={},
    )
    assert result["metrics"]["filter_type"] == "band_pass"
    assert result["metrics"]["high_pass"] == 0.1
    assert result["metrics"]["low_pass"] == 0.4


def test_intensity_filtering_no_params_is_noop(tmp_path):
    """With neither high_pass nor low_pass, the step is a no-op (copies)."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out_noop")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("intensity_filtering")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={},
        context={},
    )
    assert result["metrics"]["filter_type"] == "noop"
    assert os.path.isfile(result["output_path"])


# --------------------------------------------------------------------- #
# Defacing step + engine family
# --------------------------------------------------------------------- #

def test_defacing_step_registered():
    s = registry.get("defacing")
    assert s.maturity == "experimental"
    assert "engine" in {p.name for p in s.parameters}


def test_defacing_engines_registered():
    """3 defacing engines: placeholder, pydeface, mri_deface."""
    from app.engines import DEFACING_ENGINES
    engines = DEFACING_ENGINES.available_engines()
    names = {e["name"] for e in engines}
    assert names == {"placeholder", "pydeface", "mri_deface"}


def test_defacing_placeholder_engine_available():
    from app.engines.defacing.placeholder import PlaceholderDefacing
    e = PlaceholderDefacing()
    assert e.is_available() is True
    assert e.availability_reason() == ""


def test_defacing_pydeface_unavailable_by_default(monkeypatch):
    monkeypatch.delenv("MRI_PREPROCESSOR_PYDEFACE", raising=False)
    from app.engines.defacing.pydeface import PyDefaceEngine
    e = PyDefaceEngine()
    assert e.is_available() is False
    assert "pydeface" in e.availability_reason()


def test_defacing_pydeface_available_via_env(monkeypatch, tmp_path):
    fake_bin = tmp_path / "fake_pydeface"
    fake_bin.write_text("#!/bin/bash\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("MRI_PREPROCESSOR_PYDEFACE", str(fake_bin))

    from app.engines.defacing.pydeface import PyDefaceEngine
    e = PyDefaceEngine()
    assert e.is_available() is True


def test_defacing_placeholder_produces_defaced_output(tmp_path):
    """The placeholder engine blurs the inferior portion of the volume."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(32, 32, 32).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("defacing")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"engine": "placeholder"},
        context={},
    )
    assert os.path.isfile(result["output_path"])
    assert result["metrics"]["engine"] == "placeholder"


def test_defacing_default_engine_is_placeholder(tmp_path):
    """Without an explicit engine param, defacing uses the default (placeholder)."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(32, 32, 32).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "out_default")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("defacing")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={},  # no engine key -- default kicks in
        context={},
    )
    assert result["metrics"]["engine"] == "placeholder"


def test_defacing_pydeface_raises_when_binary_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("MRI_PREPROCESSOR_PYDEFACE", raising=False)
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))

    fn = registry.get_fn("defacing")
    with pytest.raises(FileNotFoundError):
        fn(input_path=str(in_path), output_dir=str(tmp_path),
           params={"engine": "pydeface"}, context={})


def test_defacing_mri_deface_raises_when_binary_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("MRI_PREPROCESSOR_MRI_DEFACE", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))

    fn = registry.get_fn("defacing")
    with pytest.raises(FileNotFoundError):
        fn(input_path=str(in_path), output_dir=str(tmp_path),
           params={"engine": "mri_deface"}, context={})


# --------------------------------------------------------------------- #
# Anonymization step
# --------------------------------------------------------------------- #

def test_anonymization_step_registered():
    s = registry.get("anonymization")
    assert s.maturity == "production_ready"
    param_names = {p.name for p in s.parameters}
    assert {"remove_extensions", "pseudonymize", "pseudo_salt"} <= param_names


def test_anonymization_strips_header_fields(tmp_path):
    """The anonymized output has its identifying header fields cleared."""
    in_path = tmp_path / "in.nii.gz"
    img = nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                          affine=np.eye(4))
    # Set some identifying fields
    img.header["db_name"] = b"original_database"
    img.header["aux_file"] = b"original_aux"
    nib.save(img, str(in_path))
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    fn = registry.get_fn("anonymization")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir,
        params={"remove_extensions": True},
        context={},
    )
    out_img = nib.load(result["output_path"])
    # NIfTI header fields are stored as fixed-width char arrays. The
    # anonymized value should be in the first few bytes.
    db_name = bytes(out_img.header["db_name"]).rstrip(b"\x00 ")
    aux_file = bytes(out_img.header["aux_file"]).rstrip(b"\x00 ")
    assert db_name == b"anonymized"
    assert aux_file == b""


def test_anonymization_pseudonymize_produces_stable_pseudo_id(tmp_path):
    """When pseudonymize=True, the metric includes a stable pseudo id."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    out_dir1 = str(tmp_path / "out_pseudo")
    os.makedirs(out_dir1, exist_ok=True)
    out_dir2 = str(tmp_path / "out_pseudo2")
    os.makedirs(out_dir2, exist_ok=True)

    fn = registry.get_fn("anonymization")
    result = fn(
        input_path=str(in_path),
        output_dir=out_dir1,
        params={
            "pseudonymize": True,
            "pseudo_salt": "test-salt",
        },
        context={"subject_id": "sub-001"},
    )
    assert result["metrics"]["pseudonymized"] is True
    assert "pseudo_id" in result["metrics"]
    pseudo1 = result["metrics"]["pseudo_id"]

    # Run again with the same subject_id + salt -> same pseudo id
    result2 = fn(
        input_path=str(in_path),
        output_dir=out_dir2,
        params={
            "pseudonymize": True,
            "pseudo_salt": "test-salt",
        },
        context={"subject_id": "sub-001"},
    )
    assert result2["metrics"]["pseudo_id"] == pseudo1


def test_anonymization_pseudonymize_different_subjects_differ(tmp_path):
    """Different subject_ids produce different pseudo ids."""
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    out1 = str(tmp_path / "o1")
    out2 = str(tmp_path / "o2")
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    fn = registry.get_fn("anonymization")
    r1 = fn(input_path=str(in_path), output_dir=out1,
            params={"pseudonymize": True, "pseudo_salt": "x"},
            context={"subject_id": "sub-001"})
    r2 = fn(input_path=str(in_path), output_dir=out2,
            params={"pseudonymize": True, "pseudo_salt": "x"},
            context={"subject_id": "sub-002"})
    assert r1["metrics"]["pseudo_id"] != r2["metrics"]["pseudo_id"]


def test_anonymization_default_no_pseudonymize(tmp_path):
    in_path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                              affine=np.eye(4)), str(in_path))
    out_dir = str(tmp_path / "o")
    os.makedirs(out_dir, exist_ok=True)
    fn = registry.get_fn("anonymization")
    result = fn(input_path=str(in_path), output_dir=out_dir,
                params={}, context={})
    assert result["metrics"]["pseudonymized"] is False


# --------------------------------------------------------------------- #
# Privacy built-in preset
# --------------------------------------------------------------------- #

def test_privacy_preset_exists():
    from app.presets import get_preset
    p = get_preset("privacy")
    assert p is not None
    assert p.is_builtin is True
    assert "anonymization" in p.step_ids
    assert "defacing" in p.step_ids


def test_privacy_preset_resolves_cleanly():
    """The privacy preset's steps resolve without errors."""
    from app.pipeline.dependency_resolver import resolve_pipeline
    from app.presets import get_preset
    p = get_preset("privacy")
    resolution = resolve_pipeline(
        p.step_ids,
        step_params=p.step_params,
        skip_recommended=p.skip_recommended,
    )
    for sid in p.step_ids:
        assert sid in resolution["ordered_pipeline"]


# --------------------------------------------------------------------- #
# API: /api/engines/defacing
# --------------------------------------------------------------------- #

def test_api_engines_defacing_lists_three_engines():
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/defacing")
        assert r.status_code == 200
        data = r.get_json()
        assert data["step_id"] == "defacing"
        assert data["supports_engine_selection"] is True
        assert data["default_engine"] == "placeholder"
        names = {e["name"] for e in data["available_engines"]}
        assert names == {"placeholder", "pydeface", "mri_deface"}


def test_api_engines_defacing_placeholder_available():
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/engines/defacing")
        data = r.get_json()
        placeholder = next(
            e for e in data["available_engines"] if e["name"] == "placeholder"
        )
        assert placeholder["available"] is True


# --------------------------------------------------------------------- #
# CLI: parse_params accepts comma-separated and JSON forms
# --------------------------------------------------------------------- #

def test_parse_params_json_form():
    """[1.0, 1.0, 1.0] is parsed as a list of floats (JSON form)."""
    result = parse_params(["resampling.voxel_size=[1.0, 1.0, 1.0]"])
    assert result == {"resampling": {"voxel_size": [1.0, 1.0, 1.0]}}


def test_parse_params_comma_form():
    """1,176,256,80,1 is parsed as a list of ints (comma-separated form)."""
    result = parse_params(["resampling.shape=1,176,256,80,1"])
    assert result == {"resampling": {"shape": [1, 176, 256, 80, 1]}}


def test_parse_params_comma_form_with_floats():
    """0.1,0.4 is parsed as a list of floats."""
    result = parse_params(["intensity_filtering.cutoffs=0.1,0.4"])
    assert result == {"intensity_filtering": {"cutoffs": [0.1, 0.4]}}


def test_parse_params_mixed_comma_form():
    """Comma form can mix ints and floats and strings."""
    result = parse_params(["some_step.params=1,2.5,hello"])
    assert result == {"some_step": {"params": [1, 2.5, "hello"]}}


def test_parse_params_string_value_preserved():
    """A non-numeric, non-comma value stays as a string."""
    result = parse_params(["brain_extraction.engine=otsu"])
    assert result == {"brain_extraction": {"engine": "otsu"}}


def test_parse_params_bool_value():
    """True / False / null are parsed via JSON."""
    result = parse_params([
        "anonymization.pseudonymize=true",
        "anonymization.remove_extensions=false",
    ])
    assert result["anonymization"]["pseudonymize"] is True
    assert result["anonymization"]["remove_extensions"] is False


# --------------------------------------------------------------------- #
# CLI: run with new steps end-to-end
# --------------------------------------------------------------------- #

@pytest.fixture
def cli_dataset(tmp_path):
    input_dir = tmp_path / "input"
    sub = input_dir / "sub-001"
    sub.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(16, 16, 16).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "sub-001_T1.nii.gz"),
    )
    return {
        "input_dir": str(input_dir),
        "output_dir": str(tmp_path / "output"),
    }


runner = CliRunner()


def test_cli_run_resampling_with_shape(cli_dataset):
    """`run --steps resampling --param resampling.shape=1,16,16,16,1`
    works end-to-end via the CLI."""
    result = runner.invoke(cli_app, [
        "run", cli_dataset["input_dir"], cli_dataset["output_dir"],
        "--steps", "resampling",
        "--modalities", "T1",
        "--param", "resampling.shape=1,16,16,16,1",
    ])
    assert result.exit_code == 0
    assert "1 succeeded" in result.stdout


def test_cli_run_intensity_normalization_extended(cli_dataset):
    """Extended intensity_normalization params work via the CLI."""
    result = runner.invoke(cli_app, [
        "run", cli_dataset["input_dir"], cli_dataset["output_dir"],
        "--steps", "intensity_normalization",
        "--modalities", "T1",
        "--param", "intensity_normalization.method=minmax",
        "--param", "intensity_normalization.input_range=0,100",
        "--param", "intensity_normalization.output_range=0,1",
        "--param", "intensity_normalization.clip_min=0",
        "--param", "intensity_normalization.clip_max=1",
    ])
    assert result.exit_code == 0
    assert "1 succeeded" in result.stdout


def test_cli_run_intensity_filtering(cli_dataset):
    """The new intensity_filtering step runs via the CLI."""
    result = runner.invoke(cli_app, [
        "run", cli_dataset["input_dir"], cli_dataset["output_dir"],
        "--steps", "intensity_filtering",
        "--modalities", "T1",
        "--param", "intensity_filtering.high_pass=0.1",
        "--param", "intensity_filtering.low_pass=0.4",
    ])
    assert result.exit_code == 0
    assert "1 succeeded" in result.stdout


def test_cli_run_privacy_preset(cli_dataset):
    """The privacy preset runs end-to-end via the CLI."""
    result = runner.invoke(cli_app, [
        "run", cli_dataset["input_dir"], cli_dataset["output_dir"],
        "--preset", "privacy",
        "--modalities", "T1",
    ])
    assert result.exit_code == 0
    assert "Preset: privacy (built-in)" in result.stdout
    assert "1 succeeded" in result.stdout


def test_cli_engines_defacing():
    """`engines defacing` lists all 3 defacing engines."""
    result = runner.invoke(cli_app, ["engines", "adapters", "defacing"])
    assert result.exit_code == 0
    assert "placeholder" in result.stdout
    assert "pydeface" in result.stdout
    assert "mri_deface" in result.stdout


def test_cli_capabilities_includes_new_steps():
    """`capabilities` lists the new PR11 steps."""
    result = runner.invoke(cli_app, ["capabilities"])
    assert result.exit_code == 0
    assert "intensity_filtering" in result.stdout
    assert "defacing" in result.stdout
    assert "anonymization" in result.stdout


def test_cli_presets_list_includes_privacy(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path / "p"))
    result = runner.invoke(cli_app, ["presets", "list"])
    assert result.exit_code == 0
    assert "privacy" in result.stdout


def test_cli_presets_show_privacy(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path / "p"))
    result = runner.invoke(cli_app, ["presets", "show", "privacy"])
    assert result.exit_code == 0
    assert "PRIVACY" in result.stdout
    assert "anonymization" in result.stdout
    assert "defacing" in result.stdout
