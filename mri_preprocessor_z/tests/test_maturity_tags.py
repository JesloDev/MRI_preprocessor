"""
Tests for PR4: maturity tags + validation_notes on ProcessingStep.

Covers:
  - Schema: ProcessingStep has `maturity` and `validation_notes` fields
    with sensible defaults (maturity="reference", validation_notes=None).
  - All 5 maturity levels are accepted.
  - Existing step modules tagged correctly:
      * brain_extraction -> experimental (Otsu placeholder)
      * fmri_motion_correction -> reference (SimpleITK stand-in for MCFLIRT)
      * dwi_motion_eddy_correction -> reference (DIPY stand-in for FSL eddy)
      * registration -> reference (SimpleITK stand-in for ANTs/FSL)
      * reorientation, bias_field_correction, resampling, intensity_normalization,
        remove_initial_volumes, temporal_filtering, confound_regression,
        spatial_smoothing, dwi_denoise, gibbs_ringing_correction, tensor_fitting
        -> production_ready
  - Registry helpers: by_maturity(), maturity_counts()
  - API surface: /api/health returns maturity_counts, /api/capabilities?maturity=
    filter works, /api/capabilities/summary endpoint exists.
  - validation_notes is preserved through model_dump() (serialization).
  - Backward compat: existing step modules that don't pass maturity still
    get the default ("reference") without breaking.
"""
import pytest

from app.pipeline.bootstrap import *  # noqa: F401,F403
from app.pipeline.step_registry import (
    Dependency,
    MaturityLevel,
    ProcessingStep,
    registry,
    step,
)


# --------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------- #

def test_processing_step_has_maturity_field():
    """maturity is a field on ProcessingStep, defaulting to 'reference'."""
    s = ProcessingStep(
        id="test_schema_step",
        name="Test",
        modalities=["T1"],
        engine="test",
    )
    assert s.maturity == "reference"


def test_processing_step_has_validation_notes_field():
    """validation_notes is a field on ProcessingStep, defaulting to None."""
    s = ProcessingStep(
        id="test_schema_step_2",
        name="Test",
        modalities=["T1"],
        engine="test",
    )
    assert s.validation_notes is None


def test_processing_step_accepts_all_maturity_levels():
    """All 5 maturity levels are accepted."""
    for level in ("planned", "experimental", "reference",
                  "production_ready", "external_engine"):
        s = ProcessingStep(
            id=f"test_step_{level}",
            name=f"Test {level}",
            modalities=["T1"],
            engine="test",
            maturity=level,
        )
        assert s.maturity == level


def test_processing_step_validation_notes_serialized_in_model_dump():
    """model_dump() includes maturity and validation_notes -- critical for
    the REST API to expose them to the UI."""
    s = ProcessingStep(
        id="test_dump_step",
        name="Test",
        modalities=["T1"],
        engine="test",
        maturity="experimental",
        validation_notes="This is a placeholder implementation.",
    )
    d = s.model_dump()
    assert d["maturity"] == "experimental"
    assert d["validation_notes"] == "This is a placeholder implementation."


def test_step_decorator_accepts_maturity_and_validation_notes():
    """The @step decorator accepts the new kwargs."""
    @step(
        id="test_decorator_step",
        name="Test Decorator Step",
        modalities=["T1"],
        engine="test",
        maturity="planned",
        validation_notes="Not implemented yet.",
    )
    def my_step(input_path, output_dir, params, context):
        return {"output_path": input_path, "outputs": {}, "metrics": {}}

    s = registry.get("test_decorator_step")
    assert s.maturity == "planned"
    assert s.validation_notes == "Not implemented yet."


# --------------------------------------------------------------------- #
# Tagging of existing built-in steps
# --------------------------------------------------------------------- #

def test_brain_extraction_is_tagged_experimental():
    """brain_extraction is the Otsu placeholder -> must be 'experimental'
    with a clear validation_notes warning."""
    s = registry.get("brain_extraction")
    assert s.maturity == "experimental"
    assert s.validation_notes is not None
    # The notes should warn about the placeholder nature
    assert "placeholder" in s.validation_notes.lower() or "otsu" in s.validation_notes.lower()
    # And mention SynthStrip or HD-BET as the recommended replacement
    notes_lower = s.validation_notes.lower()
    assert "synthstrip" in notes_lower or "hd-bet" in notes_lower


def test_fmri_motion_correction_is_tagged_reference():
    """fmri_motion_correction is a SimpleITK stand-in for MCFLIRT."""
    s = registry.get("fmri_motion_correction")
    assert s.maturity == "reference"
    assert s.validation_notes is not None
    assert "mcflirt" in s.validation_notes.lower()


def test_dwi_motion_eddy_correction_is_tagged_reference():
    """dwi_motion_eddy_correction is a DIPY stand-in for FSL eddy."""
    s = registry.get("dwi_motion_eddy_correction")
    assert s.maturity == "reference"
    assert s.validation_notes is not None
    assert "eddy" in s.validation_notes.lower()


def test_registration_is_tagged_reference():
    """registration is a SimpleITK stand-in for ANTs / FSL."""
    s = registry.get("registration")
    assert s.maturity == "reference"
    assert s.validation_notes is not None


# List-driven test: each tuple is (step_id, expected_maturity)
PRODUCTION_READY_STEPS = [
    "reorientation",
    "bias_field_correction",
    "resampling",
    "intensity_normalization",
    "remove_initial_volumes",
    "temporal_filtering",
    "confound_regression",
    "spatial_smoothing",
    "dwi_denoise",
    "gibbs_ringing_correction",
    "tensor_fitting",
]


@pytest.mark.parametrize("step_id", PRODUCTION_READY_STEPS)
def test_step_is_tagged_production_ready(step_id):
    """Standard nibabel/SimpleITK/DIPY/nilearn steps should be tagged
    production_ready -- they use validated reference implementations."""
    s = registry.get(step_id)
    assert s.maturity == "production_ready", (
        f"{step_id} should be production_ready, got {s.maturity}"
    )


def test_at_least_one_step_is_experimental():
    """Sanity: the platform has at least one experimental step (brain_extraction).
    This confirms the maturity system is actually being used, not just
    defaulted everywhere to 'reference'."""
    experimental_steps = registry.by_maturity("experimental")
    assert len(experimental_steps) >= 1
    assert any(s.id == "brain_extraction" for s in experimental_steps)


def test_at_least_one_step_is_reference():
    """Sanity: at least one reference-tier step exists (the swap-for-X cases)."""
    reference_steps = registry.by_maturity("reference")
    assert len(reference_steps) >= 1


def test_at_least_one_step_is_production_ready():
    """Sanity: at least one production_ready step exists."""
    production_steps = registry.by_maturity("production_ready")
    assert len(production_steps) >= 1


# --------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------- #

def test_registry_by_maturity_returns_filtered_list():
    """by_maturity(level) returns only steps with that maturity."""
    experimental = registry.by_maturity("experimental")
    for s in experimental:
        assert s.maturity == "experimental"

    production = registry.by_maturity("production_ready")
    for s in production:
        assert s.maturity == "production_ready"


def test_registry_by_maturity_empty_for_unused_level():
    """by_maturity returns empty list for a level with no steps.

    We use 'external_engine' here because no built-in steps are tagged
    with that level (they'd require an actual external tool integration).
    Note: 'planned' is also unused by built-in steps, but earlier tests
    in this file register a throwaway 'planned' test step -- so 'planned'
    wouldn't reliably be empty across the whole test session. 'external_engine'
    is cleaner because nothing in this test file uses it."""
    external = registry.by_maturity("external_engine")
    assert external == []


def test_registry_maturity_counts_returns_all_levels():
    """maturity_counts returns a dict with all 5 maturity levels as keys."""
    counts = registry.maturity_counts()
    assert set(counts.keys()) == {
        "planned", "experimental", "reference",
        "production_ready", "external_engine",
    }


def test_registry_maturity_counts_sums_to_total():
    """Sum of counts across all maturity levels equals total steps."""
    counts = registry.maturity_counts()
    total_steps = len(registry.all_steps())
    assert sum(counts.values()) == total_steps


# --------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------- #

def test_api_health_includes_maturity_counts():
    """/api/health returns maturity_counts in addition to registered_steps."""
    from app.main import app
    with app.test_client() as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.get_json()
        assert "maturity_counts" in data
        assert "registered_steps" in data
        assert data["status"] == "ok"
        # Sanity: at least one experimental step exists (brain_extraction)
        assert data["maturity_counts"]["experimental"] >= 1


def test_api_capabilities_returns_maturity_and_validation_notes():
    """/api/capabilities surfaces maturity and validation_notes per step."""
    from app.main import app
    with app.test_client() as client:
        r = client.get("/api/capabilities")
        assert r.status_code == 200
        data = r.get_json()
        # brain_extraction should be in the response with maturity=experimental
        assert "brain_extraction" in data
        be = data["brain_extraction"]
        assert be["maturity"] == "experimental"
        assert be["validation_notes"] is not None
        # A production_ready step should also be there
        assert "reorientation" in data
        assert data["reorientation"]["maturity"] == "production_ready"


def test_api_capabilities_filter_by_maturity_experimental():
    """/api/capabilities?maturity=experimental returns only experimental steps."""
    from app.main import app
    with app.test_client() as client:
        r = client.get("/api/capabilities?maturity=experimental")
        assert r.status_code == 200
        data = r.get_json()
        # All returned steps should have maturity == 'experimental'
        for sid, step in data.items():
            assert step["maturity"] == "experimental", (
                f"{sid} should be experimental, got {step['maturity']}"
            )
        # brain_extraction should be in there
        assert "brain_extraction" in data


def test_api_capabilities_filter_by_maturity_multiple():
    """/api/capabilities?maturity=experimental&maturity=reference returns
    both experimental and reference steps."""
    from app.main import app
    with app.test_client() as client:
        r = client.get("/api/capabilities?maturity=experimental&maturity=reference")
        assert r.status_code == 200
        data = r.get_json()
        for sid, step in data.items():
            assert step["maturity"] in ("experimental", "reference")


def test_api_capabilities_filter_excludes_non_matching():
    """/api/capabilities?maturity=external_engine returns empty dict when
    no steps match (no built-in step is tagged external_engine).

    Note: we use 'external_engine' instead of 'planned' because earlier
    tests in this file register a throwaway 'planned' test step into the
    global registry, which would make the 'planned' filter non-empty."""
    from app.main import app
    with app.test_client() as client:
        r = client.get("/api/capabilities?maturity=external_engine")
        assert r.status_code == 200
        data = r.get_json()
        assert data == {}


def test_api_capabilities_summary_endpoint():
    """/api/capabilities/summary returns total_steps, maturity_counts,
    modalities_covered."""
    from app.main import app
    with app.test_client() as client:
        r = client.get("/api/capabilities/summary")
        assert r.status_code == 200
        data = r.get_json()
        assert "total_steps" in data
        assert "maturity_counts" in data
        assert "modalities_covered" in data
        assert isinstance(data["modalities_covered"], list)
        assert data["total_steps"] == sum(data["maturity_counts"].values())


# --------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------- #

def test_step_module_without_maturity_kwargs_gets_default():
    """A step module that doesn't pass `maturity` (e.g. an old module
    from before PR4) gets the default 'reference' maturity -- doesn't
    break, doesn't crash."""
    # We can't easily un-register the existing test_decorator_step, so
    # we use a fresh id.
    @step(
        id="test_no_maturity_kwarg_step",
        name="Test",
        modalities=["T1"],
        engine="test",
        # No maturity kwarg
    )
    def my_step(input_path, output_dir, params, context):
        return {"output_path": input_path, "outputs": {}, "metrics": {}}

    s = registry.get("test_no_maturity_kwarg_step")
    assert s.maturity == "reference"
    assert s.validation_notes is None
