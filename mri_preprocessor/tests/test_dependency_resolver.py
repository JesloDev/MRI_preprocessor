import pytest

from app.pipeline.bootstrap import *  # noqa: F401,F403 -- registers steps
from app.pipeline.dependency_resolver import DependencyError, resolve_pipeline


def test_resolves_missing_prerequisites():
    result = resolve_pipeline(["registration"])
    # registration depends on brain_extraction -> bias_field_correction -> reorientation
    assert "brain_extraction" in result["ordered_pipeline"]
    assert "bias_field_correction" in result["ordered_pipeline"]
    assert "reorientation" in result["ordered_pipeline"]
    # reorientation must come before bias_field_correction, which must come
    # before brain_extraction, which must come before registration
    order = result["ordered_pipeline"]
    assert order.index("reorientation") < order.index("bias_field_correction")
    assert order.index("bias_field_correction") < order.index("brain_extraction")
    assert order.index("brain_extraction") < order.index("registration")


def test_unknown_step_raises():
    with pytest.raises(DependencyError):
        resolve_pipeline(["not_a_real_step"])


def test_no_duplicate_steps_in_output():
    result = resolve_pipeline(["registration", "brain_extraction"])
    assert len(result["ordered_pipeline"]) == len(set(result["ordered_pipeline"]))
