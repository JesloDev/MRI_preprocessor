"""
Tests for the preset system (PR10).

Covers:
  - Preset schema: fields, defaults, to_apply_kwargs() translation
  - Built-in presets: 4 shipped, each resolves cleanly via the existing
    dependency resolver (no phantom step ids)
  - Storage layer: save, load, delete, list, filename sanitization,
    path-traversal protection, corrupt-file resilience
  - Registry: layers built-ins on top of customs, can't overwrite a
    built-in, can't delete a built-in
  - API: /api/presets, /api/presets/<name>, /api/presets/<name>/resolve,
    POST /api/presets, DELETE /api/presets/<name>, and the new
    {preset: ...} body form for /api/jobs
  - CLI: presets list, presets show, presets create, presets delete,
    run --preset
  - End-to-end: a custom preset saved via the CLI is immediately
    runnable via `run --preset`
  - Architecture compliance: presets don't duplicate logic -- the
    same resolve_pipeline + run_job backend is used regardless of
    whether the user passed --steps or --preset
"""
import json
import os
import time
from unittest import mock

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.presets.schema import Preset


# --------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------- #

def test_preset_schema_defaults():
    """A minimal Preset has sensible defaults for all optional fields."""
    p = Preset(name="test", step_ids=["reorientation"])
    assert p.description == ""
    assert p.step_params == {}
    assert p.skip_recommended == []
    assert p.target_modalities is None
    assert p.is_builtin is False
    assert p.created_at > 0  # set automatically


def test_preset_to_apply_kwargs():
    """to_apply_kwargs() returns the kwargs dict for resolve_pipeline /
    run_job. This is the load-bearing translation: presets aren't a
    separate execution path, they're a configuration shorthand."""
    p = Preset(
        name="test",
        step_ids=["reorientation", "brain_extraction"],
        step_params={"brain_extraction": {"engine": "otsu"}},
        skip_recommended=["bias_field_correction"],
        target_modalities=["T1"],
    )
    kwargs = p.to_apply_kwargs()
    assert kwargs == {
        "selected_step_ids": ["reorientation", "brain_extraction"],
        "step_params": {"brain_extraction": {"engine": "otsu"}},
        "skip_recommended": ["bias_field_correction"],
        "target_modalities": ["T1"],
    }


def test_preset_serialization_round_trip():
    """model_dump() + Preset(**data) preserves all fields."""
    p = Preset(
        name="round-trip",
        description="test",
        step_ids=["reorientation"],
        step_params={"x": {"y": 1}},
        skip_recommended=[],
        target_modalities=["T1"],
        is_builtin=False,
    )
    data = p.model_dump()
    p2 = Preset(**data)
    assert p2.name == p.name
    assert p2.step_ids == p.step_ids
    assert p2.step_params == p.step_params
    assert p2.target_modalities == p.target_modalities


# --------------------------------------------------------------------- #
# Built-in presets
# --------------------------------------------------------------------- #

def test_four_builtins_exist():
    """The 4 original built-in presets are registered: basic-qc, standard,
    ml-ready, full-neuroimaging. (PR11 adds a 5th, `privacy`, which is
    tested separately.)"""
    from app.presets import list_presets
    names = {p.name for p in list_presets() if p.is_builtin}
    assert {"basic-qc", "standard", "ml-ready", "full-neuroimaging"} <= names


def test_all_builtins_are_marked_builtin():
    from app.presets import list_presets
    for p in list_presets():
        if p.name in ("basic-qc", "standard", "ml-ready", "full-neuroimaging"):
            assert p.is_builtin is True


def test_builtins_resolve_cleanly():
    """Every built-in preset's step_ids resolve via the existing
    dependency resolver without errors. This confirms presets
    reference real step ids, not phantom ones."""
    from app.pipeline.dependency_resolver import resolve_pipeline
    from app.presets import list_presets
    for preset in list_presets():
        if not preset.is_builtin:
            continue
        resolution = resolve_pipeline(
            preset.step_ids,
            step_params=preset.step_params,
            skip_recommended=preset.skip_recommended,
        )
        # All requested steps should be in the resolved pipeline
        for sid in preset.step_ids:
            assert sid in resolution["ordered_pipeline"], (
                f"Preset {preset.name} references step {sid} which isn't "
                f"in the resolved pipeline"
            )


def test_basic_qc_preset_minimal():
    """basic-qc is just reorientation."""
    from app.presets import get_preset
    p = get_preset("basic-qc")
    assert p.step_ids == ["reorientation"]
    assert p.step_params == {}


def test_ml_ready_preset_has_step_params():
    """ml-ready pre-fills resampling.voxel_size and smoothing.fwhm as
    sensible ML defaults."""
    from app.presets import get_preset
    p = get_preset("ml-ready")
    assert "resampling" in p.step_params
    assert p.step_params["resampling"]["voxel_size"] == [1.0, 1.0, 1.0]
    assert "spatial_smoothing" in p.step_params
    assert p.step_params["spatial_smoothing"]["fwhm"] == 6.0


def test_full_neuroimaging_preset_includes_registration():
    """full-neuroimaging is the only built-in that includes registration."""
    from app.presets import get_preset
    p = get_preset("full-neuroimaging")
    assert "registration" in p.step_ids


# --------------------------------------------------------------------- #
# Storage layer
# --------------------------------------------------------------------- #

def test_storage_save_and_load(tmp_path, monkeypatch):
    """A preset saved to disk can be loaded back with all fields intact."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets.storage import save_preset, load_preset

    p = Preset(
        name="storable",
        description="test preset",
        step_ids=["reorientation", "brain_extraction"],
        step_params={"brain_extraction": {"engine": "otsu"}},
        skip_recommended=[],
        target_modalities=["T1"],
        is_builtin=False,
    )
    path = save_preset(p)
    assert os.path.isfile(path)

    loaded = load_preset("storable")
    assert loaded is not None
    assert loaded.name == "storable"
    assert loaded.step_ids == ["reorientation", "brain_extraction"]
    assert loaded.step_params == {"brain_extraction": {"engine": "otsu"}}
    assert loaded.target_modalities == ["T1"]


def test_storage_load_nonexistent_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets.storage import load_preset
    assert load_preset("nonexistent") is None


def test_storage_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets.storage import save_preset, delete_preset_file, load_preset

    p = Preset(name="deletable", step_ids=["reorientation"])
    save_preset(p)
    assert delete_preset_file("deletable") is True
    assert load_preset("deletable") is None
    # Delete again -> False (already gone)
    assert delete_preset_file("deletable") is False


def test_storage_list_saved_names(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets.storage import save_preset, list_saved_preset_names

    save_preset(Preset(name="alpha", step_ids=["reorientation"]))
    save_preset(Preset(name="beta", step_ids=["reorientation"]))
    save_preset(Preset(name="gamma", step_ids=["reorientation"]))

    names = list_saved_preset_names()
    assert names == ["alpha", "beta", "gamma"]  # sorted


def test_storage_filename_sanitization(tmp_path, monkeypatch):
    """Unsafe characters in preset names are sanitized into safe
    filenames. Path-traversal attempts become safe filenames (no
    path separators, no leading dots)."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets.storage import save_preset, _sanitize_filename, get_presets_dir

    # Path traversal attempt: the file ends up as a single safe filename
    # in the presets directory, not as /etc/passwd
    safe_name = _sanitize_filename("../../etc/passwd")
    # The security property: no path separators in the result
    assert "/" not in safe_name
    assert os.sep not in safe_name
    # And it must end with .json
    assert safe_name.endswith(".json")
    # Leading dots stripped (so we don't create hidden files)
    assert not safe_name.startswith(".")

    # The on-disk file is in the presets dir, not /etc/
    p = Preset(name="../../etc/passwd", step_ids=["reorientation"])
    save_preset(p)
    files = os.listdir(get_presets_dir())
    assert all(not f.startswith("/") for f in files)
    assert all(os.sep not in f for f in files)


def test_storage_corrupt_file_returns_none(tmp_path, monkeypatch):
    """A corrupt JSON file shouldn't crash the storage layer."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets.storage import load_preset, get_presets_dir

    # Write a corrupt JSON file
    corrupt_path = os.path.join(get_presets_dir(), "corrupt.json")
    with open(corrupt_path, "w") as f:
        f.write("{not valid json")

    # Loading should return None, not raise
    assert load_preset("corrupt") is None


def test_storage_env_var_overrides_default(tmp_path, monkeypatch):
    """$MRI_PREPROCESSOR_PRESETS_DIR overrides the default ~/.mripreprocessor/presets/."""
    custom_dir = tmp_path / "custom_presets"
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(custom_dir))
    from app.presets.storage import get_presets_dir
    assert get_presets_dir() == str(custom_dir)
    # The dir should be created if it doesn't exist
    assert os.path.isdir(custom_dir)


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #

def test_registry_lists_builtins_first(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets import list_presets
    presets = list_presets()
    # Built-ins come first (in declared order), then customs (sorted)
    builtins = [p for p in presets if p.is_builtin]
    customs = [p for p in presets if not p.is_builtin]
    # PR11 adds `privacy` as a 5th built-in
    builtin_names = [p.name for p in builtins]
    assert "basic-qc" in builtin_names
    assert "standard" in builtin_names
    assert "ml-ready" in builtin_names
    assert "full-neuroimaging" in builtin_names
    assert "privacy" in builtin_names
    # Built-ins are in declared order
    assert builtins[0].name == "basic-qc"
    assert builtins[1].name == "standard"
    assert builtins[2].name == "ml-ready"
    assert builtins[3].name == "full-neuroimaging"
    assert builtins[4].name == "privacy"
    assert customs == []  # no customs yet


def test_registry_save_and_retrieve_custom(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets import register_preset, get_preset
    p = Preset(name="custom1", step_ids=["reorientation"], description="x")
    register_preset(p)
    retrieved = get_preset("custom1")
    assert retrieved is not None
    assert retrieved.name == "custom1"
    assert retrieved.is_builtin is False  # forced on save


def test_registry_cannot_save_over_builtin(tmp_path, monkeypatch):
    """A custom preset can't shadow a built-in."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets import register_preset
    p = Preset(name="ml-ready", step_ids=["reorientation"])  # collides with built-in
    with pytest.raises(ValueError) as exc_info:
        register_preset(p)
    assert "reserved" in str(exc_info.value).lower() or "built-in" in str(exc_info.value).lower()


def test_registry_cannot_delete_builtin(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets import delete_preset
    with pytest.raises(ValueError) as exc_info:
        delete_preset("ml-ready")
    assert "built-in" in str(exc_info.value).lower()


def test_registry_get_preset_built_in_takes_precedence(tmp_path, monkeypatch):
    """If both a built-in and a custom have the same name (only possible
    via direct disk manipulation), the built-in wins."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets import get_preset
    from app.presets.storage import save_preset

    # Save a custom with the same name as a built-in (bypass the registry's
    # collision check by going direct to storage)
    save_preset(Preset(name="ml-ready", step_ids=["reorientation"],
                       is_builtin=False))

    p = get_preset("ml-ready")
    assert p.is_builtin is True  # built-in wins
    assert "reorientation" in p.step_ids  # the built-in's full step list


def test_registry_corrupt_custom_is_skipped(tmp_path, monkeypatch):
    """A corrupt custom preset file is skipped, not crash."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.presets import list_presets
    from app.presets.storage import get_presets_dir

    corrupt_path = os.path.join(get_presets_dir(), "corrupt.json")
    with open(corrupt_path, "w") as f:
        f.write("{invalid json")

    presets = list_presets()
    # Built-ins still load; corrupt file is silently skipped
    assert any(p.name == "ml-ready" for p in presets)


# --------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------- #

def test_api_presets_list():
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/presets")
        assert r.status_code == 200
        data = r.get_json()
        names = {p["name"] for p in data}
        assert {"basic-qc", "standard", "ml-ready", "full-neuroimaging"} <= names
        # Each entry has the documented fields
        for entry in data:
            assert "name" in entry
            assert "is_builtin" in entry
            assert "num_steps" in entry


def test_api_preset_detail():
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/presets/ml-ready")
        assert r.status_code == 200
        data = r.get_json()
        assert data["name"] == "ml-ready"
        assert data["is_builtin"] is True
        assert "reorientation" in data["step_ids"]
        assert "resampling" in data["step_params"]


def test_api_preset_detail_unknown_returns_404():
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/presets/nonexistent")
        assert r.status_code == 404


def test_api_preset_resolve():
    """/api/presets/<name>/resolve returns what the preset would expand to."""
    from app.main import app
    with app.test_client() as c:
        r = c.get("/api/presets/standard/resolve")
        assert r.status_code == 200
        data = r.get_json()
        assert "preset" in data
        assert "resolution" in data
        # standard preset has 5 steps; resolved pipeline should include
        # all of them plus any auto-inserted deps
        ordered = data["resolution"]["ordered_pipeline"]
        for sid in data["preset"]["step_ids"]:
            assert sid in ordered


def test_api_preset_create_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.main import app
    with app.test_client() as c:
        # Create
        r = c.post("/api/presets", json={
            "name": "api-custom",
            "description": "created via API",
            "step_ids": ["reorientation"],
        })
        assert r.status_code == 201
        # Verify
        r = c.get("/api/presets/api-custom")
        assert r.status_code == 200
        assert r.get_json()["name"] == "api-custom"
        # Delete
        r = c.delete("/api/presets/api-custom")
        assert r.status_code == 200
        # Verify gone
        r = c.get("/api/presets/api-custom")
        assert r.status_code == 404


def test_api_preset_create_rejects_builtin_name(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.main import app
    with app.test_client() as c:
        r = c.post("/api/presets", json={
            "name": "ml-ready",  # collides with built-in
            "step_ids": ["reorientation"],
        })
        assert r.status_code == 400


def test_api_jobs_with_preset_body(tmp_path, monkeypatch):
    """POST /api/jobs accepts {preset: "basic-qc"} instead of {step_ids: ...}."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))

    # Build a tiny dataset
    input_dir = tmp_path / "input"
    sub = input_dir / "sub-001"
    sub.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "sub-001_T1.nii.gz"),
    )
    output_dir = str(tmp_path / "output")

    from app.main import app
    import time
    with app.test_client() as c:
        r = c.post("/api/jobs", json={
            "input_dir": str(input_dir),
            "output_dir": output_dir,
            "preset": "basic-qc",
        })
        assert r.status_code == 202
        job_id = r.get_json()["job_id"]

        # Poll until done
        deadline = time.time() + 30
        while True:
            jr = c.get(f"/api/jobs/{job_id}")
            data = jr.get_json()
            if data["status"] in ("completed", "failed"):
                break
            if time.time() > deadline:
                pytest.fail("Job did not complete in time")
            time.sleep(0.05)

        assert data["status"] == "completed"
        # basic-qc only runs reorientation
        ordered = data["result"]["pipeline"]["ordered_pipeline"]
        assert ordered == ["reorientation"]


def test_api_jobs_with_unknown_preset_returns_400(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path))
    from app.main import app
    with app.test_client() as c:
        r = c.post("/api/jobs", json={
            "input_dir": "/tmp",
            "output_dir": "/tmp/out",
            "preset": "nonexistent",
        })
        assert r.status_code == 400


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

runner = CliRunner()


@pytest.fixture
def isolated_presets_dir(tmp_path, monkeypatch):
    """Isolates the presets directory so tests don't pollute the user's
    real ~/.mripreprocessor/presets/."""
    monkeypatch.setenv("MRI_PREPROCESSOR_PRESETS_DIR", str(tmp_path / "presets"))
    return tmp_path / "presets"


@pytest.fixture
def tiny_dataset(tmp_path):
    input_dir = tmp_path / "input"
    sub = input_dir / "sub-001"
    sub.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32) * 100,
                        affine=np.eye(4)),
        str(sub / "sub-001_T1.nii.gz"),
    )
    return {
        "input_dir": str(input_dir),
        "output_dir": str(tmp_path / "output"),
    }


def test_cli_presets_list_shows_all_builtins(isolated_presets_dir):
    result = runner.invoke(cli_app, ["presets", "list"])
    assert result.exit_code == 0
    for name in ("basic-qc", "standard", "ml-ready", "full-neuroimaging"):
        assert name in result.stdout


def test_cli_presets_show_builtin(isolated_presets_dir):
    result = runner.invoke(cli_app, ["presets", "show", "ml-ready"])
    assert result.exit_code == 0
    assert "ML-READY" in result.stdout
    assert "Purpose" in result.stdout
    assert "Requested steps" in result.stdout
    assert "Final execution order" in result.stdout
    assert "Component maturity" in result.stdout


def test_cli_presets_show_unknown_returns_error(isolated_presets_dir):
    result = runner.invoke(cli_app, ["presets", "show", "nonexistent"])
    assert result.exit_code != 0


def test_cli_presets_create_comma_separated_steps(isolated_presets_dir):
    """`--steps reorientation,brain_extraction` works (comma-separated)."""
    result = runner.invoke(cli_app, [
        "presets", "create", "my-preset",
        "--steps", "reorientation,brain_extraction,resampling",
        "--description", "Test comma form",
    ])
    assert result.exit_code == 0
    assert "Saved preset" in result.stdout

    # Verify it's listed
    result = runner.invoke(cli_app, ["presets", "list"])
    assert "my-preset" in result.stdout


def test_cli_presets_create_repeated_steps(isolated_presets_dir):
    """`--steps reorientation --steps brain_extraction` works (repeated)."""
    result = runner.invoke(cli_app, [
        "presets", "create", "my-preset-2",
        "--steps", "reorientation",
        "--steps", "brain_extraction",
    ])
    assert result.exit_code == 0


def test_cli_presets_create_with_param(isolated_presets_dir):
    """`--param step.key=value` flows into the saved preset's step_params."""
    result = runner.invoke(cli_app, [
        "presets", "create", "param-preset",
        "--steps", "reorientation,brain_extraction,resampling",
        "--param", "resampling.voxel_size=[2.0,2.0,2.0]",
    ])
    assert result.exit_code == 0

    # Verify via the show command
    result = runner.invoke(cli_app, ["presets", "show", "param-preset"])
    assert "resampling.voxel_size = [2.0, 2.0, 2.0]" in result.stdout


def test_cli_presets_create_cannot_shadow_builtin(isolated_presets_dir):
    result = runner.invoke(cli_app, [
        "presets", "create", "ml-ready",  # collides with built-in
        "--steps", "reorientation",
    ])
    assert result.exit_code != 0


def test_cli_presets_delete_custom(isolated_presets_dir):
    # Create
    runner.invoke(cli_app, [
        "presets", "create", "to-delete",
        "--steps", "reorientation",
    ])
    # Delete
    result = runner.invoke(cli_app, ["presets", "delete", "to-delete"])
    assert result.exit_code == 0
    assert "Deleted" in result.stdout
    # Verify gone
    result = runner.invoke(cli_app, ["presets", "list"])
    assert "to-delete" not in result.stdout


def test_cli_presets_delete_builtin_fails(isolated_presets_dir):
    result = runner.invoke(cli_app, ["presets", "delete", "ml-ready"])
    assert result.exit_code != 0


def test_cli_presets_delete_unknown_fails(isolated_presets_dir):
    result = runner.invoke(cli_app, ["presets", "delete", "nonexistent"])
    assert result.exit_code != 0


def test_cli_run_with_preset(tiny_dataset, isolated_presets_dir):
    """`run --preset basic-qc` works end-to-end."""
    result = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--preset", "basic-qc",
    ])
    assert result.exit_code == 0
    assert "Job completed" in result.stdout
    assert "Preset: basic-qc" in result.stdout
    assert "1 succeeded" in result.stdout


def test_cli_run_with_custom_preset(tiny_dataset, isolated_presets_dir):
    """A custom preset created via the CLI is immediately runnable via
    `run --preset`."""
    # Create
    runner.invoke(cli_app, [
        "presets", "create", "custom-run",
        "--steps", "reorientation,brain_extraction,resampling",
        "--param", "resampling.voxel_size=[2.0,2.0,2.0]",
    ])
    # Run
    result = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--preset", "custom-run",
        "--modalities", "T1",
    ])
    assert result.exit_code == 0
    assert "Preset: custom-run (custom)" in result.stdout
    assert "1 succeeded" in result.stdout


def test_cli_run_preset_and_steps_mutually_exclusive(tiny_dataset, isolated_presets_dir):
    """--steps and --preset can't both be given."""
    result = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--steps", "reorientation",
        "--preset", "basic-qc",
    ])
    assert result.exit_code != 0


def test_cli_run_neither_preset_nor_steps_fails(tiny_dataset, isolated_presets_dir):
    """At least one of --steps or --preset must be given."""
    result = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
    ])
    assert result.exit_code != 0


def test_cli_run_with_unknown_preset_fails(tiny_dataset, isolated_presets_dir):
    result = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--preset", "nonexistent",
    ])
    assert result.exit_code != 0


def test_cli_run_preset_with_extra_param_override(tiny_dataset, isolated_presets_dir):
    """`--param` on top of `--preset` overrides the preset's step_params."""
    # Create a preset with a specific resampling voxel size
    runner.invoke(cli_app, [
        "presets", "create", "overridable",
        "--steps", "reorientation,resampling",
        "--param", "resampling.voxel_size=[2.0,2.0,2.0]",
    ])
    # Run with an override
    result = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], tiny_dataset["output_dir"],
        "--preset", "overridable",
        "--modalities", "T1",
        "--param", "resampling.voxel_size=[1.0,1.0,1.0]",
    ])
    assert result.exit_code == 0

    # Check the provenance log for the override
    import os
    prov_dir = os.path.join(tiny_dataset["output_dir"], "provenance")
    prov_files = [f for f in os.listdir(prov_dir) if f.endswith(".jsonl")]
    assert len(prov_files) == 1
    with open(os.path.join(prov_dir, prov_files[0])) as f:
        lines = f.readlines()
    # Find the resampling step record
    resampling_records = [
        json.loads(l) for l in lines
        if "resampling" in l and "step_id" in json.loads(l)
    ]
    assert len(resampling_records) >= 1
    # The caller's override (1.0,1.0,1.0) should win
    assert resampling_records[0]["params"]["voxel_size"] == [1.0, 1.0, 1.0]


# --------------------------------------------------------------------- #
# Architecture compliance: no duplicated logic
# --------------------------------------------------------------------- #

def test_preset_uses_same_backend_as_explicit_steps(tiny_dataset, isolated_presets_dir):
    """A run with --preset should produce the same resolved pipeline as a
    run with --steps matching the preset's step_ids. This confirms
    presets aren't a separate execution path -- they're a configuration
    shorthand that flows into the same resolve_pipeline + run_job.
    """
    # Run with explicit steps
    out_explicit = tiny_dataset["output_dir"] + "_explicit"
    result1 = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], out_explicit,
        "--steps", "reorientation",
    ])
    assert result1.exit_code == 0

    # Run with a preset that has the same single step
    runner.invoke(cli_app, [
        "presets", "create", "matches-explicit",
        "--steps", "reorientation",
    ])
    out_preset = tiny_dataset["output_dir"] + "_preset"
    result2 = runner.invoke(cli_app, [
        "run", tiny_dataset["input_dir"], out_preset,
        "--preset", "matches-explicit",
    ])
    assert result2.exit_code == 0

    # Both should have processed 1 acquisition successfully
    assert "1 succeeded" in result1.stdout
    assert "1 succeeded" in result2.stdout

    # Both should have written a provenance log with the same set of
    # executed step_ids
    def get_executed_step_ids(out_dir):
        prov_dir = os.path.join(out_dir, "provenance")
        files = [f for f in os.listdir(prov_dir) if f.endswith(".jsonl")]
        with open(os.path.join(prov_dir, files[0])) as f:
            return sorted({
                json.loads(l)["step_id"]
                for l in f
                if "step_id" in json.loads(l)
            })

    steps1 = get_executed_step_ids(out_explicit)
    steps2 = get_executed_step_ids(out_preset)
    assert steps1 == steps2 == ["reorientation"]
