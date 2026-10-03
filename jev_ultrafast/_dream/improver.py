"""_dream.improver — extracted from jev_ultrafast.dream."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import TYPE_CHECKING

from .common import SCHEMA_VERSION, TCB_VERSION
from .experiments import EXPERIMENT_PLAN_SCHEMA, experiment_plan_digest, experiment_plan_signature
from .policy import ObjectiveWeights, mutate_policies
from .promotion import PromotionDecision, PromotionGate
from .replay import ReplaySimulator, replay_pool_digest, split_manifest_digest, split_worlds

if TYPE_CHECKING:
    from typing import Iterable

    from .policy import ExplorationPolicy
    from .replay import ReplayResult, ReplayWorld

__all__ = [
    'DreamImprover',
    'DreamReport',
]


@dataclass(frozen=True)
class DreamReport:
    selected: ExplorationPolicy
    baseline: ExplorationPolicy
    promotion: PromotionDecision
    candidates: tuple[dict, ...]
    split_sizes: dict[str, int]
    world_pool_digest: str
    split_manifest_digest: str
    evidence_head_hash: str | None = None
    tcb_versions: tuple[str, ...] = ()
    live_canary_required: bool = True
    cost_model_digest: str | None = None
    outcome_model_digest: str | None = None
    choice_model_digest: str | None = None
    trial_model_digest: str | None = None
    experiment_proposals: tuple[dict, ...] = ()
    trials_digest: str | None = None
    trial_estimates: dict | None = None

    def to_dict(self):
        return {
            "schema": SCHEMA_VERSION,
            "selected": self.selected.to_dict(),
            "selected_digest": self.selected.digest,
            "selected_behavior_digest": self.selected.behavior_digest,
            "baseline": self.baseline.to_dict(),
            "baseline_digest": self.baseline.digest,
            "baseline_behavior_digest": self.baseline.behavior_digest,
            "promotion": {
                "approved": self.promotion.approved,
                "reason": self.promotion.reason,
                "baseline": asdict(self.promotion.baseline),
                "candidate": asdict(self.promotion.candidate),
                "validation": asdict(self.promotion.validation) if self.promotion.validation else None,
                "holdout": asdict(self.promotion.holdout) if self.promotion.holdout else None,
            },
            "candidates": list(self.candidates),
            "split_sizes": self.split_sizes,
            "world_pool_digest": self.world_pool_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "evidence_head_hash": self.evidence_head_hash,
            "tcb_versions": list(self.tcb_versions),
            "live_canary_required": self.live_canary_required,
            "cost_model_digest": self.cost_model_digest,
            "outcome_model_digest": self.outcome_model_digest,
            "choice_model_digest": self.choice_model_digest,
            # Digest of the causal prior (TrialChoiceModel) that generated any
            # trial-backed proposals — distinct from the observational prior.
            "trial_model_digest": self.trial_model_digest,
            "experiment_proposals": list(self.experiment_proposals),
            # Randomized-trial estimates are annotation, never gate input:
            # the ITT arm estimates exist so an operator (or a future learned
            # layer) can see what assigned experiments measured.
            "trials_digest": self.trials_digest,
            "trial_estimates": self.trial_estimates,
            "tcb_version": TCB_VERSION,
        }


class DreamImprover:
    """Evaluate bounded policy variants using historical replay.

    Split methodology, strictly ordered: train drives the improvement gates
    and the candidate search, validation participates in candidate selection,
    and the holdout split is examined exactly once — after a single candidate
    has already been selected — as a confirmatory check on that selection. No
    candidate is ever evaluated on holdout while selection is still open: a
    dataset that screens policies is a validation set, not a holdout, and
    re-examining it across the search would bias selection toward whatever
    happens to score well on that sample.
    """

    # Splits that may participate in the candidate search and selection. The
    # holdout is deliberately absent — it is read once, post-selection.
    SELECTION_SPLITS = ("train", "validation")

    def __init__(self, *, weights: ObjectiveWeights | None = None, gate: PromotionGate | None = None):
        self.weights = weights or ObjectiveWeights()
        self.gate = gate or PromotionGate()

    @staticmethod
    def _robust_gain(base_results: dict[str, ReplayResult | None], candidate_results: dict[str, ReplayResult | None]):
        """Worst-case per-world gain over the *selection* splits only."""
        deltas = []
        for name in DreamImprover.SELECTION_SPLITS:
            base_result, candidate_result = base_results.get(name), candidate_results.get(name)
            if base_result is not None and candidate_result is not None and base_result.metrics.worlds:
                deltas.append(candidate_result.metrics.score_per_world - base_result.metrics.score_per_world)
        return min(deltas) if deltas else float("-inf")

    def improve(
        self,
        worlds: Iterable[ReplayWorld],
        base: ExplorationPolicy,
        *,
        evidence_head_hash: str | None = None,
        cost_model=None,
        outcome_model=None,
        choice_model=None,
        trial_model=None,
        trials=None,
        scheduler=None,
        plan_signer=None,
    ) -> DreamReport:
        worlds = list(worlds)
        splits = split_worlds(worlds)
        pool_digest = replay_pool_digest(worlds)
        split_digest = split_manifest_digest(splits)
        tcb_versions = tuple(sorted({world.tcb_version for world in worlds}))
        if len(tcb_versions) > 1:
            raise ValueError(
                "Replay worlds mix DREAM TCB versions; improve each evidence generation separately"
            )
        split_sizes = {k: len(v) for k, v in splits.items()}
        train = splits["train"]
        if not train:
            empty = ReplaySimulator([], self.weights).evaluate(base)
            decision = PromotionDecision(False, "no replay worlds", empty.metrics, empty.metrics)
            return DreamReport(
                base,
                base,
                decision,
                (),
                split_sizes,
                pool_digest,
                split_digest,
                evidence_head_hash,
                tcb_versions,
            )

        simulators = {
            name: (ReplaySimulator(items, self.weights) if items else None)
            for name, items in splits.items()
        }
        base_results = {
            name: (
                sim.evaluate(
                    base, cost_model=cost_model, outcome_model=outcome_model,
                    choice_model=choice_model,
                    trial_model=trial_model,
                    # Experiment proposals come from the baseline evaluation:
                    # they are "what the prior would test differently under the
                    # currently deployed behavior", not candidate-policy notes.
                    collect_proposals=(name == "train"),
                ) if sim else None
            )
            for name, sim in simulators.items()
        }

        def robust_estimated_gain(candidate_results):
            deltas = []
            for name in self.SELECTION_SPLITS:
                base_result, candidate_result = base_results.get(name), candidate_results.get(name)
                if (
                    base_result is None
                    or candidate_result is None
                    or not base_result.metrics.worlds
                    or base_result.metrics.estimated_score is None
                    or candidate_result.metrics.estimated_score is None
                ):
                    continue
                deltas.append(
                    candidate_result.metrics.estimated_score / candidate_result.metrics.worlds
                    - base_result.metrics.estimated_score / base_result.metrics.worlds
                )
            return min(deltas) if deltas else None

        evaluated = []
        # Each entry carries the candidate's selection-split replay results so
        # the one post-selection holdout examination can reuse them.
        passing: list[tuple[float, float, float, ExplorationPolicy, PromotionDecision, dict]] = []
        for policy in mutate_policies(base):
            # The search touches train + validation only. The holdout
            # simulator exists but is never handed a candidate here.
            candidate_results = {
                name: (
                    sim.evaluate(
                        policy, cost_model=cost_model, outcome_model=outcome_model,
                        choice_model=choice_model, trial_model=trial_model,
                    ) if sim else None
                )
                for name, sim in simulators.items()
                if name in self.SELECTION_SPLITS
            }
            if policy.behavior_digest == base.behavior_digest:
                decision = PromotionDecision(
                    False,
                    "baseline behavior",
                    base_results["train"].metrics,
                    candidate_results["train"].metrics,
                )
            else:
                decision = self.gate.assess(
                    base_results["train"],
                    candidate_results["train"],
                    validation_baseline=base_results["validation"],
                    validation_candidate=candidate_results["validation"],
                )
            robust_gain = self._robust_gain(base_results, candidate_results)
            average_gain = fmean([
                candidate_results[name].metrics.score_per_world - base_results[name].metrics.score_per_world
                for name in self.SELECTION_SPLITS
                if base_results[name] is not None and candidate_results[name] is not None
            ])
            estimated_gain = robust_estimated_gain(candidate_results)
            entry = {
                "policy": policy.to_dict(),
                "digest": policy.digest,
                "behavior_digest": policy.behavior_digest,
                "train": asdict(candidate_results["train"].metrics),
                "validation": (
                    asdict(candidate_results["validation"].metrics) if candidate_results["validation"] else None
                ),
                # Not evaluated during selection — backfilled once, after the
                # winner is chosen, when a holdout split exists.
                "holdout": None,
                "replay_approved": decision.approved,
                "reason": decision.reason,
                "robust_gain_per_world": robust_gain,
                "average_gain_per_world": average_gain,
            }
            if estimated_gain is not None:
                entry["estimated_gain_per_world"] = estimated_gain
            if outcome_model is not None and outcome_model.samples:
                entry["predicted_page_changes_per_world"] = fmean(
                    s["predicted_change"] for s in candidate_results["train"].per_world
                )
            if choice_model is not None and choice_model.samples:
                # Advisory annotation only: the choice prior evaluated under the
                # candidate's own offered ordering, plus how often its
                # counterfactual proposal differs from the action the recorded
                # policy actually took. None of this enters a gate. The two
                # subjects are reported separately: the *selected* action's
                # prior uncertainty/confidence and the *proposal's* — mixing
                # them would make a confident divergent proposal look like a
                # confident recorded action or vice versa.
                pw = candidate_results["train"].per_world
                steps = sum(s["actions"] for s in pw)
                if steps:
                    selected_uncertainty = fmean(
                        s["choice_uncertainty"] / max(1, s["actions"]) for s in pw
                    )
                    selected_confident = sum(s["choice_confident"] for s in pw) / steps
                    entry["choice_model"] = {
                        "predicted_progress_per_step": fmean(
                            s["choice_predicted_change"] / max(1, s["actions"]) for s in pw
                        ),
                        "proposed_progress_per_step": fmean(
                            s["choice_proposed_change"] / max(1, s["actions"]) for s in pw
                        ),
                        "divergence_rate": sum(s["choice_divergences"] for s in pw) / steps,
                        "selected_mean_uncertainty": selected_uncertainty,
                        "selected_confident_fraction": selected_confident,
                        "proposal_mean_uncertainty": fmean(
                            s["choice_proposal_uncertainty"] / max(1, s["actions"]) for s in pw
                        ),
                        "proposal_confident_fraction": sum(
                            s["choice_proposal_confident"] for s in pw
                        ) / steps,
                        # Pre-split names retained as aliases; their subject was
                        # always the recorded (selected) action.
                        "mean_uncertainty": selected_uncertainty,
                        "confident_fraction": selected_confident,
                    }
            evaluated.append(entry)
            if decision.approved:
                passing.append(
                    (robust_gain, average_gain, estimated_gain or float("-inf"), policy, decision, candidate_results)
                )

        if passing:
            passing.sort(
                key=lambda item: (item[0], item[1], item[2], item[3].digest), reverse=True
            )
            _robust, _average, _estimated, selected, promotion, selected_results = passing[0]
            # The one holdout examination of the whole run: the already-
            # selected candidate against the incumbent, once, as a
            # confirmatory check on the selection. If it fails, the report
            # rejects — the search does not reopen with holdout knowledge,
            # which would silently turn it back into a second validation set.
            holdout_sim = simulators["holdout"]
            if holdout_sim is not None:
                holdout_candidate = holdout_sim.evaluate(
                    selected,
                    cost_model=cost_model,
                    outcome_model=outcome_model,
                    choice_model=choice_model,
                    trial_model=trial_model,
                )
                promotion = self.gate.assess(
                    base_results["train"],
                    selected_results["train"],
                    validation_baseline=base_results["validation"],
                    validation_candidate=selected_results["validation"],
                    holdout_baseline=base_results["holdout"],
                    holdout_candidate=holdout_candidate,
                )
                for entry in evaluated:
                    if entry["digest"] == selected.digest:
                        entry["holdout"] = asdict(holdout_candidate.metrics)
                        entry["holdout_examined"] = "post_selection"
                        break
        else:
            selected = base
            base_train = base_results["train"]
            promotion = PromotionDecision(
                False,
                "no changed policy passed all replay gates",
                base_train.metrics,
                base_train.metrics,
                base_results["validation"].metrics if base_results["validation"] else None,
                base_results["holdout"].metrics if base_results["holdout"] else None,
            )

        experiment_proposals = ()
        if (choice_model is not None or trial_model is not None) and base_results[
            "train"
        ] is not None:
            collected = [
                proposal
                for summary in base_results["train"].per_world
                for proposal in summary.get("experiment_proposals", ())
            ]
            # The scheduler decides which *unresolved* hypothesis is worth a
            # real browser experiment (expected information gain × practical
            # importance ÷ evidence coverage); settled hypotheses are dropped
            # entirely. Without a scheduler the historical expected-delta
            # order is the fallback. A proposal is a hypothesis to test under
            # the real authority plane, not a finding.
            if scheduler is not None:
                collected = scheduler.rank(collected, estimates=trials)
            else:
                collected.sort(key=lambda p: (-p["expected_delta"], p["world"], p["step"]))
            stamped = []
            for proposal in collected[:24]:
                # A stamped plan is an immutable ExperimentPlan: the digest
                # binds the hypothesis *and* the bindings the agent
                # revalidates live (task, family, state, model choice, offered
                # catalogue, policy behavior, originating model, world pool
                # and evidence head). A plan whose bindings no longer hold is
                # stale and must be discarded, never reinterpreted. With a
                # signer configured the plan additionally carries a
                # domain-separated Ed25519 authority block — digest integrity
                # alone is not provenance.
                item = {
                    k: v
                    for k, v in {"schema": EXPERIMENT_PLAN_SCHEMA, **proposal}.items()
                    if k not in {"estimate", "scheduler"}
                }
                item["world_pool_digest"] = pool_digest
                item["evidence_head_hash"] = evidence_head_hash
                item["digest"] = experiment_plan_digest(item)
                authority = experiment_plan_signature(item, plan_signer)
                if authority:
                    item["authority"] = authority
                stamped.append(item)
            experiment_proposals = tuple(stamped)

        return DreamReport(
            selected=selected,
            baseline=base,
            promotion=promotion,
            candidates=tuple(evaluated),
            split_sizes=split_sizes,
            world_pool_digest=pool_digest,
            split_manifest_digest=split_digest,
            evidence_head_hash=evidence_head_hash,
            tcb_versions=tcb_versions,
            experiment_proposals=experiment_proposals,
            cost_model_digest=(cost_model.digest if cost_model is not None and cost_model.samples else None),
            outcome_model_digest=(
                outcome_model.digest if outcome_model is not None and outcome_model.samples else None
            ),
            choice_model_digest=(
                choice_model.digest if choice_model is not None and choice_model.samples else None
            ),
            trial_model_digest=(
                trial_model.digest
                if trial_model is not None and getattr(trial_model, "trials", None) is not None
                else None
            ),
            # Randomized assignments are the only causal evidence the store
            # holds — surface the fitted estimates so they annotate the report
            # instead of staying a diagnostic object nobody reads.
            trials_digest=trials.digest if trials is not None else None,
            trial_estimates=trials.estimate() if trials is not None else None,
        )
