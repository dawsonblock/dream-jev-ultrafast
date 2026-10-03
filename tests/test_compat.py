"""Import-compatibility and behavior-preservation tests for the Phase-4
internal split of ``dream.py``/``dreamlearn.py`` into ``_dream``/``_learning``.

The facades must keep the pre-split module contract: every public symbol,
private helper historically importable from the module, monkeypatchability of
module-level names, and deterministic digest/serialization outputs.
"""

import jev_ultrafast._dream as _dream
import jev_ultrafast._learning as _learning
import jev_ultrafast.dream as dream
import jev_ultrafast.dreamlearn as dreamlearn


def test_dream_facade_reexports_every_implementation_name():
    for name in _dream.__all__:
        assert getattr(dream, name) is getattr(_dream, name), name


def test_dreamlearn_facade_reexports_every_implementation_name():
    for name in _learning.__all__:
        assert getattr(dreamlearn, name) is getattr(_learning, name), name


def test_dream_public_names_stable():
    expected = {
        "ACTION_KINDS", "ATTESTATION_DOMAIN", "CanaryDecision",
        "CanaryEvidence", "CanaryGate", "CanaryMetrics", "CanaryRunSummary",
        "DreamImprover", "DreamReport", "EXPERIMENT_PLAN_SCHEMA",
        "ExperienceStore", "ExplorationPolicy", "HealthDecision",
        "HealthGate", "ObjectiveWeights", "PolicyRegistry",
        "PromotionDecision", "PromotionGate", "RecordedTransition",
        "ReplayMetrics", "ReplayResult", "ReplaySimulator", "ReplayWorld",
        "SCHEMA_VERSION", "SUPPORTED_SCHEMAS", "SUPPORTED_TCB_VERSIONS",
        "TCB_VERSION", "candidate_catalog_digest",
        "experiment_plan_authority", "experiment_plan_digest",
        "experiment_plan_signature", "experiment_plan_signer_from_env",
        "mutate_policies", "new_run_id", "replay_pool_digest",
        "split_manifest_digest", "split_worlds", "summarize_usage",
        "task_key",
    }
    assert expected <= set(vars(dream))
    for name in expected:
        assert getattr(dream, name) is getattr(_dream, name), name


def test_dreamlearn_public_names_stable():
    expected = {
        "TERMINATION_REASONS", "TRIAL_CELL_LEN", "TRIAL_COUNT_LEN",
        "CausalChoicePolicy", "ChoiceModel", "CostModel",
        "CounterfactualTrials", "ExperimentScheduler", "OutcomeModel",
        "TrialChoiceModel", "UtilityWeights", "overlap_bucket",
        "phase_bucket", "rank_bucket",
    }
    assert expected <= set(vars(dreamlearn))
    for name in expected:
        assert getattr(dreamlearn, name) is getattr(_learning, name), name


def test_private_helpers_remain_importable_from_facades():
    # Callers (tests, scripts) historically reached these private helpers.
    from jev_ultrafast.dream import (  # noqa: F401
        _file_lock,
        _fsync_dir,
        _sign_test_p_value,
        _stable_hash,
    )
    from jev_ultrafast.dreamlearn import (  # noqa: F401
        _SIGNATURE_FIELDS,
        _SIGNATURE_INDEX,
        _bernoulli_kl,
        _coord,
        _sequential_alpha,
        _signature_masks,
        _wilson_interval,
    )


def test_facade_forwards_monkeypatch_to_implementation_modules(monkeypatch):
    import jev_ultrafast._dream.improver as improver
    import jev_ultrafast._dream.replay as replay

    sentinel = object()
    monkeypatch.setattr(dream, "ReplaySimulator", sentinel)
    assert improver.ReplaySimulator is sentinel
    assert replay.ReplaySimulator is sentinel


def test_facade_forwards_time_patch_to_all_owners(monkeypatch):
    import jev_ultrafast._dream.evidence as evidence
    import jev_ultrafast._dream.registry as registry

    class Clock:
        @staticmethod
        def time():
            return 1234.5

    monkeypatch.setattr(dream, "time", Clock)
    assert evidence.time is Clock
    assert registry.time is Clock


def test_policy_and_mutation_digests_unchanged():
    assert dream.ExplorationPolicy().digest == (
        "aae1c8f45eb2e7c2e5f1a1dd6102615eb25b69c244d28bea7141181b0cd62226"
    )
    digests = [m.digest for m in dream.mutate_policies(dream.ExplorationPolicy())]
    assert len(digests) == 52
    assert digests[0] == (
        "828e8d89efb0a08a89f622dad00f2b4678e9e9c6e85c7b26dbdef41547dcad94"
    )
    assert digests[-1] == (
        "424ad83c9d47f646b4291677db1d6a4e9cb21cbada0d7964a2763b20c4ebed9e"
    )


def test_plan_and_catalog_digests_unchanged():
    plan = {
        "schema": "jev-experiment-plan/1", "task": "t", "state": "s",
        "model_choice": {"id": "c1"}, "offered": [1, 2],
        "policy": {"digest": "d"}, "origin_model": "m",
    }
    assert dream.experiment_plan_digest(plan) == (
        "2e3f2ebccf2af55ad4b1755df40b17788508f77221318927f013f6bbf0d69a9f"
    )
    assert dream.candidate_catalog_digest(
        [{"id": "a", "kind": "click"}, {"id": "b", "kind": "fill"}]
    ) == "b2fe3399b30990c9b5b2b326343258767889a0fe5bba37dfc3b9b817fc2e506d"
    assert dream.task_key("close the modal") == (
        "e196f415749ad987d02751f0abdbfd06b79b7afc289fef8607bb0f55b2776354"
    )


def test_causal_serialization_and_statistics_unchanged():
    # Cell layout values updated for the jev-trials/8 outcome-vector block
    # (Phase 5): 15-key + 10 stats + 3 costs + 9 reasons + 6 outcome counters.
    assert dreamlearn.TRIAL_CELL_LEN == 43
    assert dreamlearn.TRIAL_COUNT_LEN == 28
    assert dreamlearn.CounterfactualTrials().to_dict() == {
        "cells": (), "duplicates": 0, "action_index": (),
        "invalid_units": 0, "rejected": 0, "registered": (),
        "version": "jev-trials/10",
    }
    masks = dreamlearn._signature_masks(
        ["click", "high", "navigation", "primary", "1", "0-4", "fill"])
    assert masks == [
        ("full", ("click", "high", "navigation", "primary", "1", "0-4", "fill"))]
    assert dreamlearn._sequential_alpha(2, base=0.05) == 0.006766894940483515
    assert dreamlearn._bernoulli_kl(0.3, 0.7) == 0.33891914415488134
    assert dreamlearn._wilson_interval(0.5, 40) == (
        0.35199278797099753, 0.6480072120290025)
