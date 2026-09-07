"""
Tests for tiered dependency resolution (PR3).

Covers:
  - Backward compat: existing `depends_on` (list of step ids) still works
    and is treated as hard tier.
  - New `dependencies` field with explicit Dependency objects.
  - Hard tier: always inserted.
  - Recommended tier: inserted by default, suppressible via skip_recommended.
  - Conditional tier: inserted only when `when` matches effective params.
  - Tier precedence: hard wins over recommended wins over conditional.
  - Structured `when` conditions: not Python expressions.
  - API output: tier_breakdown groups auto-inserted deps by tier.
  - skip_recommended surfaces in the output.
  - Backward compat: existing tests in test_dependency_resolver.py still pass.

The tests register their own throwaway ProcessingSteps in the global
registry; they're careful to use unique step ids that don't collide
with the real ones (so the built-in registry stays clean for other tests).
"""
import pytest

from app.pipeline.bootstrap import *  # noqa: F401,F403 -- registers real steps
from app.pipeline.dependency_resolver import (
    DependencyError,
    expand_dependencies,
    resolve_pipeline,
    topological_order,
)
from app.pipeline.step_registry import Dependency, ProcessingStep, registry


# --------------------------------------------------------------------- #
# Test fixtures: throwaway step ids prefixed with `test_` to avoid
# colliding with the real built-in steps.
# --------------------------------------------------------------------- #

def _register_test_step(step_id, depends_on=None, dependencies=None,
                         default_params=None, modalities=("T1",)):
    """Registers a throwaway ProcessingStep for testing.

    Safe to call multiple times with the same id during a single test
    run -- it skips registration if already present.
    """
    try:
        registry.get(step_id)
        return  # already registered
    except KeyError:
        pass
    registry.register(
        ProcessingStep(
            id=step_id,
            name=f"Test step {step_id}",
            modalities=list(modalities),
            engine="test",
            depends_on=depends_on or [],
            dependencies=dependencies or [],
            default_params=default_params or {},
        ),
        # No-op step implementation: just returns the input path.
        # We don't run these steps; we only test the resolver.
        lambda input_path, output_dir, params, context: {
            "output_path": input_path, "outputs": {}, "metrics": {},
        },
    )


# Register a fixed set of test steps once at module import time.
# Layout:
#
#   test_root                -- no deps
#   test_root_with_hard      -- hard dep on test_root
#   test_root_with_recm      -- recommended dep on test_root
#   test_root_with_cond      -- conditional dep on test_root (when foo == "bar")
#
#   test_chain_a             -- no deps
#   test_chain_b             -- hard dep on test_chain_a
#   test_chain_c             -- hard dep on test_chain_b
#   test_chain_d             -- recommended dep on test_chain_c
#
#   test_dual_tier_step      -- hard dep on test_root AND recommended dep on
#                                test_chain_a (which has its own hard chain)

_register_test_step("test_root")
_register_test_step("test_chain_a")
_register_test_step("test_chain_b", depends_on=["test_chain_a"])
_register_test_step("test_chain_c", depends_on=["test_chain_b"])
_register_test_step("test_chain_d",
                    dependencies=[Dependency(step_id="test_chain_c", tier="recommended")])
_register_test_step("test_root_with_hard",
                    depends_on=["test_root"])
_register_test_step("test_root_with_recm",
                    dependencies=[Dependency(step_id="test_root", tier="recommended")])
_register_test_step("test_root_with_cond",
                    dependencies=[Dependency(
                        step_id="test_root", tier="conditional",
                        when={"foo": "bar"},
                    )])
_register_test_step("test_dual_tier_step",
                    depends_on=["test_root"],
                    dependencies=[Dependency(step_id="test_chain_a", tier="recommended")])


# --------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------- #

def test_legacy_depends_on_treated_as_hard():
    """Existing step modules use `depends_on=[...]` (list of step ids).
    After PR3, this should still work and be treated as a hard-tier dep."""
    # test_chain_c hard-depends on test_chain_b -> test_chain_a (transitive)
    result = resolve_pipeline(["test_chain_c"])
    assert "test_chain_a" in result["ordered_pipeline"]
    assert "test_chain_b" in result["ordered_pipeline"]
    assert "test_chain_c" in result["ordered_pipeline"]

    # All auto-inserted deps should be in the "hard" tier breakdown
    breakdown = result["tier_breakdown"]
    assert "test_chain_a" in breakdown["hard"]
    assert "test_chain_b" in breakdown["hard"]
    assert breakdown["recommended"] == []
    assert breakdown["conditional"] == []


def test_existing_real_step_resolution_still_works():
    """The real `registration` step (uses legacy depends_on) must still
    resolve identically to pre-PR3 behaviour. This is the test from
    test_dependency_resolver.py."""
    result = resolve_pipeline(["registration"])
    assert "brain_extraction" in result["ordered_pipeline"]
    assert "bias_field_correction" in result["ordered_pipeline"]
    assert "reorientation" in result["ordered_pipeline"]
    order = result["ordered_pipeline"]
    assert order.index("reorientation") < order.index("bias_field_correction")
    assert order.index("bias_field_correction") < order.index("brain_extraction")
    assert order.index("brain_extraction") < order.index("registration")


# --------------------------------------------------------------------- #
# Hard tier
# --------------------------------------------------------------------- #

def test_hard_dependency_always_inserted():
    """Hard dep is always inserted; nothing the user can do to skip it."""
    result = resolve_pipeline(["test_root_with_hard"])
    assert "test_root" in result["ordered_pipeline"]
    assert "test_root_with_hard" in result["ordered_pipeline"]
    assert "test_root" in result["tier_breakdown"]["hard"]


def test_hard_dependency_cannot_be_skipped():
    """skip_recommended has no effect on hard deps."""
    result = resolve_pipeline(
        ["test_root_with_hard"],
        skip_recommended=["test_root"],  # irrelevant: test_root is hard
    )
    assert "test_root" in result["ordered_pipeline"]  # still included


# --------------------------------------------------------------------- #
# Recommended tier
# --------------------------------------------------------------------- #

def test_recommended_dependency_inserted_by_default():
    """Recommended dep is inserted when no skip_recommended is provided."""
    result = resolve_pipeline(["test_root_with_recm"])
    assert "test_root" in result["ordered_pipeline"]
    assert "test_root_with_recm" in result["ordered_pipeline"]
    # The auto-inserted test_root should appear in the recommended tier
    assert "test_root" in result["tier_breakdown"]["recommended"]


def test_recommended_dependency_can_be_skipped():
    """skip_recommended=[step_id] suppresses the recommended dep."""
    result = resolve_pipeline(
        ["test_root_with_recm"],
        skip_recommended=["test_root"],
    )
    # test_root should NOT be in the pipeline (it was skipped)
    assert "test_root" not in result["ordered_pipeline"]
    assert "test_root_with_recm" in result["ordered_pipeline"]
    # And it should appear in the skipped_recommended list
    assert "test_root" in result["skipped_recommended"]


def test_recommended_dependency_transitive_skipped():
    """If test_chain_d (recommended dep on test_chain_c) is skipped, the
    whole test_chain_c -> test_chain_b -> test_chain_a chain should be
    absent -- nothing else pulls them in."""
    result = resolve_pipeline(
        ["test_chain_d"],
        skip_recommended=["test_chain_c"],
    )
    assert "test_chain_d" in result["ordered_pipeline"]
    # test_chain_c was the recommended dep we skipped
    assert "test_chain_c" not in result["ordered_pipeline"]
    # Its transitive hard deps also absent (nothing else needs them)
    assert "test_chain_b" not in result["ordered_pipeline"]
    assert "test_chain_a" not in result["ordered_pipeline"]


# --------------------------------------------------------------------- #
# Conditional tier
# --------------------------------------------------------------------- #

def test_conditional_dependency_inserted_when_when_matches():
    """test_root_with_cond has a conditional dep on test_root when
    params.foo == "bar". When the user provides foo="bar", the dep fires."""
    result = resolve_pipeline(
        ["test_root_with_cond"],
        step_params={"test_root_with_cond": {"foo": "bar"}},
    )
    assert "test_root" in result["ordered_pipeline"]
    assert "test_root" in result["tier_breakdown"]["conditional"]


def test_conditional_dependency_not_inserted_when_when_doesnt_match():
    """When the user provides foo="baz" (or doesn't provide foo), the
    conditional dep doesn't fire."""
    result = resolve_pipeline(
        ["test_root_with_cond"],
        step_params={"test_root_with_cond": {"foo": "baz"}},
    )
    assert "test_root_with_cond" in result["ordered_pipeline"]
    assert "test_root" not in result["ordered_pipeline"]
    assert result["tier_breakdown"]["conditional"] == []


def test_conditional_dependency_not_inserted_when_param_absent():
    """If the param referenced by `when` is not in effective_params, the
    condition is treated as not matching."""
    result = resolve_pipeline(["test_root_with_cond"])
    assert "test_root_with_cond" in result["ordered_pipeline"]
    assert "test_root" not in result["ordered_pipeline"]


def test_conditional_dependency_uses_default_params_when_no_override():
    """If the step's default_params already has foo="bar", the conditional
    fires without the user needing to pass step_params."""
    # Register a new test step with foo="bar" in its defaults
    _register_test_step(
        "test_cond_with_default",
        dependencies=[Dependency(
            step_id="test_root", tier="conditional", when={"foo": "bar"},
        )],
        default_params={"foo": "bar"},  # default already satisfies the when
    )
    result = resolve_pipeline(["test_cond_with_default"])
    # No step_params provided -- defaults are used, condition matches
    assert "test_root" in result["ordered_pipeline"]


def test_conditional_dependency_user_override_wins_over_default():
    """If default has foo="bar" but user passes foo="baz", the condition
    should NOT match (user override wins)."""
    result = resolve_pipeline(
        ["test_cond_with_default"],
        step_params={"test_cond_with_default": {"foo": "baz"}},
    )
    assert "test_cond_with_default" in result["ordered_pipeline"]
    assert "test_root" not in result["ordered_pipeline"]


def test_conditional_when_multiple_keys_all_must_match():
    """A `when` with multiple {key: value} pairs requires ALL to match."""
    _register_test_step(
        "test_cond_multi_key",
        dependencies=[Dependency(
            step_id="test_root", tier="conditional",
            when={"foo": "bar", "baz": "qux"},
        )],
        default_params={"foo": "bar", "baz": "qux"},
    )
    result = resolve_pipeline(["test_cond_multi_key"])
    assert "test_root" in result["ordered_pipeline"]

    # Now override baz to something else -> condition no longer matches
    result2 = resolve_pipeline(
        ["test_cond_multi_key"],
        step_params={"test_cond_multi_key": {"baz": "wrong"}},
    )
    assert "test_root" not in result2["ordered_pipeline"]


# --------------------------------------------------------------------- #
# Tier precedence
# --------------------------------------------------------------------- #

def test_hard_wins_over_recommended_when_both_request_same_dep():
    """If step A hard-depends on X, and step B recommends X, X must be
    treated as hard (not skippable) because at least one step requires it
    unconditionally.

    Setup: test_dual_tier_step hard-depends on test_root (via depends_on)
    AND recommends test_chain_a. If we ALSO request test_root_with_recm
    (which recommends test_root), test_root should be HARD (from
    test_dual_tier_step's hard dep) -- not skippable via skip_recommended.
    """
    result = resolve_pipeline(
        ["test_dual_tier_step", "test_root_with_recm"],
        skip_recommended=["test_root"],
    )
    # test_root should STILL be in the pipeline because test_dual_tier_step
    # hard-depends on it; the user's skip_recommended only affects the
    # recommended-tier request from test_root_with_recm.
    assert "test_root" in result["ordered_pipeline"]
    # And it should be in the HARD breakdown (since one requester treated it as hard)
    assert "test_root" in result["tier_breakdown"]["hard"]
    assert "test_root" not in result["tier_breakdown"]["recommended"]


# --------------------------------------------------------------------- #
# API output shape
# --------------------------------------------------------------------- #

def test_resolve_pipeline_output_shape():
    """resolve_pipeline returns the documented shape: requested,
    ordered_pipeline, added_dependencies, tier_breakdown, skipped_recommended.
    """
    result = resolve_pipeline(["test_chain_c"])
    assert "requested" in result
    assert "ordered_pipeline" in result
    assert "added_dependencies" in result
    assert "tier_breakdown" in result
    assert "skipped_recommended" in result

    assert result["requested"] == ["test_chain_c"]
    assert isinstance(result["ordered_pipeline"], list)
    assert isinstance(result["added_dependencies"], list)
    assert isinstance(result["tier_breakdown"], dict)
    assert isinstance(result["skipped_recommended"], list)


def test_tier_breakdown_excludes_user_requested_steps():
    """tier_breakdown only includes AUTO-INSERTED steps. The user's own
    selection isn't part of the tier breakdown (it's always required by
    definition)."""
    result = resolve_pipeline(["test_root_with_hard"])
    assert "test_root_with_hard" not in result["tier_breakdown"]["hard"]
    assert "test_root_with_hard" not in result["tier_breakdown"]["recommended"]
    # But test_root (the auto-inserted hard dep) IS in the breakdown
    assert "test_root" in result["tier_breakdown"]["hard"]


def test_added_dependencies_lists_all_auto_inserted():
    """added_dependencies is a flat list of all auto-inserted step ids,
    regardless of tier. Useful for the UI's 'N prerequisites added' message."""
    result = resolve_pipeline(
        ["test_root_with_hard", "test_root_with_recm"],
    )
    # test_root was added once (deduplicated), tier hard wins
    assert "test_root" in result["added_dependencies"]


def test_skipped_recommended_surfaces_in_output():
    """The skipped_recommended list is echoed back so the UI can confirm
    what the user chose to skip."""
    result = resolve_pipeline(
        ["test_root_with_recm"],
        skip_recommended=["test_root"],
    )
    assert result["skipped_recommended"] == ["test_root"]


# --------------------------------------------------------------------- #
# Topological ordering with tiered deps
# --------------------------------------------------------------------- #

def test_topological_order_respects_both_depends_on_and_dependencies():
    """topological_order walks both legacy depends_on and the new
    dependencies field. Tier doesn't matter for ordering."""
    # test_chain_c -> test_chain_b -> test_chain_a (all hard)
    order = topological_order(["test_chain_a", "test_chain_b", "test_chain_c"])
    assert order.index("test_chain_a") < order.index("test_chain_b")
    assert order.index("test_chain_b") < order.index("test_chain_c")


# --------------------------------------------------------------------- #
# Expand_dependencies returns Dependency objects
# --------------------------------------------------------------------- #

def test_expand_dependencies_returns_dependency_objects():
    """expand_dependencies returns the raw Dependency objects (not just ids),
    so callers can inspect tier and condition."""
    deps = expand_dependencies(["test_root_with_recm"])
    # 2 entries: the user-requested test_root_with_recm (hard), and
    # test_root (recommended).
    assert len(deps) == 2
    by_id = {d.step_id: d for d in deps}
    assert by_id["test_root_with_recm"].tier == "hard"
    assert by_id["test_root"].tier == "recommended"


# --------------------------------------------------------------------- #
# Integration: skip_recommended chains through transitive hard deps
# --------------------------------------------------------------------- #

def test_skip_recommended_does_not_skip_transitive_hard_deps():
    """If we skip test_chain_d (recommended), the chain below it
    (test_chain_c, test_chain_b, test_chain_a) should also disappear
    -- because nothing else pulls them in. But if some OTHER step in the
    request hard-depends on test_chain_c, then test_chain_c stays (and
    so does its chain)."""
    # Register a step that HARD-depends on test_chain_c
    _register_test_step("test_hard_user_of_c", depends_on=["test_chain_c"])

    # Without skip_recommended: test_chain_d + test_hard_user_of_c
    # both run. test_chain_c, test_chain_b, test_chain_a all included.
    result = resolve_pipeline(["test_chain_d", "test_hard_user_of_c"])
    assert "test_chain_c" in result["ordered_pipeline"]
    assert "test_chain_b" in result["ordered_pipeline"]
    assert "test_chain_a" in result["ordered_pipeline"]

    # With skip_recommended=["test_chain_c"]: only the recommended
    # request from test_chain_d is suppressed; test_hard_user_of_c still
    # hard-requires test_chain_c, so test_chain_c stays.
    result2 = resolve_pipeline(
        ["test_chain_d", "test_hard_user_of_c"],
        skip_recommended=["test_chain_c"],
    )
    assert "test_chain_c" in result2["ordered_pipeline"]  # still pulled in by hard dep
    # And test_chain_c should now be in the HARD tier breakdown (because
    # the only remaining requester is test_hard_user_of_c which is hard)
    assert "test_chain_c" in result2["tier_breakdown"]["hard"]
    assert "test_chain_c" not in result2["tier_breakdown"]["recommended"]


# --------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------- #

def test_unknown_step_still_raises():
    """Unknown step id in the user's selection raises DependencyError."""
    with pytest.raises(DependencyError):
        resolve_pipeline(["does_not_exist"])


def test_unknown_step_in_dependency_raises():
    """If a step declares a dep on a non-existent step id, the resolver
    raises during transitive expansion."""
    _register_test_step("test_dep_missing",
                        depends_on=["truly_nonexistent_step"])
    with pytest.raises(DependencyError):
        resolve_pipeline(["test_dep_missing"])
