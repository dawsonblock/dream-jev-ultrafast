"""_dream.replay — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .common import _stable_hash, candidate_catalog_digest
from .policy import ExplorationPolicy, ObjectiveWeights, ReplayMetrics

if TYPE_CHECKING:
    from typing import Iterable


__all__ = [
    'RecordedTransition',
    'REPLAY_VERDICTS',
    'ReplayResult',
    'ReplaySimulator',
    'ReplayWorld',
    'replay_pool_digest',
    'split_manifest_digest',
    'split_worlds',
]


# Per-world replay verdicts (Phase 13): exactly one per evaluated world,
# never conflated. ``verified_success`` — the recorded terminal was
# verifier-confirmed done; ``measured_terminal`` — a recorded terminal was
# reached but was not verified success (the world's measured ending, kept
# distinct from replay-imposed stops); ``coverage_miss`` — the candidate
# policy's offered catalogue excluded the action the recording actually
# took, so the counterfactual is *truncated evidence*, not a measured
# failure; ``blocked`` — replay's own no-progress window fired;
# ``inconclusive`` — the trajectory ended before any terminal event, so
# replay ran out of evidence entirely.
REPLAY_VERDICTS = (
    "verified_success",
    "measured_terminal",
    "coverage_miss",
    "blocked",
    "inconclusive",
)


@dataclass(frozen=True)
class RecordedTransition:
    run_id: str
    task_key: str
    state: str
    next_state: str
    selected_id: str
    selected_kind: str
    candidate_actions: tuple[dict, ...]
    candidate_digest: str
    selected_observed_rank: int | None
    page_changed: bool
    latency_ms: int
    model_calls: int
    tokens: int
    stale_or_failure: int
    risk_events: int
    terminal: str | None = None
    verified: bool = False
    catalog: str = "recorded"
    offered_digest: str | None = None
    offered_count: int | None = None
    selected_offered_rank: int | None = None
    selected_propensity: float | None = None
    experiment: dict | None = None
    causal_override: dict | None = None
    # Whether the run declared an upload allowlist (recorded on run_started).
    # Upload actions sit in the observed catalogue whenever the page has a
    # file input; they were only offerable — and are only re-offerable —
    # when this is true.
    uploads_declared: bool = False


@dataclass
class ReplayWorld:
    """One recorded online run used as an empirical replay world.

    Worlds are intentionally not merged across live runs. This avoids stitching
    together a counterfactual trajectory from states that only happen to share a
    fingerprint. Multiple runs of the same task remain separate worlds in the
    simulator pool, matching the paper's history-of-trees framing.
    """

    key: str
    task_key: str
    goal: str
    start_state: str
    tcb_version: str
    trajectory: tuple[RecordedTransition, ...]
    family_key: str = ""

    @property
    def digest(self) -> str:
        payload = {
            "key": self.key,
            "task_key": self.task_key,
            "family_key": self.family_key,
            "start_state": self.start_state,
            "tcb_version": self.tcb_version,
            "trajectory": [asdict(item) for item in self.trajectory],
        }
        return _stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    @classmethod
    def from_events(cls, events: Iterable[dict]) -> list["ReplayWorld"]:
        runs: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            run_id = event.get("run_id")
            if run_id:
                runs[run_id].append(event)

        worlds = []
        for run_id, run_events in runs.items():
            run_events.sort(key=lambda e: (e.get("sequence", 0), e.get("recorded_at_ms", 0)))
            start = next((e for e in run_events if e.get("event") == "run_started"), None)
            if not start:
                continue
            key = start["task_key"]
            family_key = str(start.get("task_family") or "").strip().lower() or key
            run_policy = None
            if start.get("policy") is not None:
                run_policy = ExplorationPolicy.from_dict(start["policy"])
            # The run's recorded upload availability — observed evidence, not
            # something replay may infer. Traces without it predate upload
            # support and can carry no upload candidates, so the default of
            # "not declared" is fail-closed, not a guess.
            uploads_declared = bool(start.get("upload_count"))
            run_tcbs = {e.get("tcb_version") for e in run_events}
            if len(run_tcbs) != 1:
                raise ValueError(f"Run {run_id} crosses DREAM TCB versions")
            run_tcb = next(iter(run_tcbs))
            final = next((e for e in reversed(run_events) if e.get("event") == "run_finished"), None)
            raw_transitions = [e for e in run_events if e.get("event") == "transition"]
            transitions = []
            for index, event in enumerate(raw_transitions):
                # Dynamic pages may change between actions without a Jev mutation.
                # Preserve the ordered empirical trajectory instead of inventing an
                # action-caused edge or rejecting valid asynchronous state drift.
                terminal = None
                verified = False
                if index == len(raw_transitions) - 1 and final:
                    terminal = final.get("status")
                    verified = bool(final.get("verified"))
                event_candidates = tuple(event.get("candidates", ()))
                observed_digest = candidate_catalog_digest(event_candidates)
                stored_digest = event.get("candidate_digest")
                if stored_digest and stored_digest != observed_digest:
                    raise ValueError(f"Run {run_id} has a candidate catalogue digest mismatch")
                selected_id = event["selected"]["id"]
                observed_rank = event.get("selected_observed_rank")
                if observed_rank is None:
                    # Pre-0.9 traces recorded the observed-catalogue rank under
                    # the ambiguous name ``selected_rank``.
                    observed_rank = event.get("selected_rank")
                if observed_rank is None:
                    observed_rank = next(
                        (i for i, item in enumerate(event_candidates) if item.get("id") == selected_id),
                        None,
                    )
                stored_offered = event.get("offered_digest")
                recorded_offered_rank = event.get("selected_offered_rank")
                recorded_offered_rank = (
                    int(recorded_offered_rank) if recorded_offered_rank is not None else None
                )
                # A *distinct* offered catalogue exists only when the trace says
                # so: catalog=="observed", an offered digest/count, or an
                # explicit offered rank. A recorded catalogue means the observed
                # catalogue itself was the offer.
                has_offered = (
                    event.get("catalog") == "observed"
                    or stored_offered is not None
                    or event.get("offered_count") is not None
                    or recorded_offered_rank is not None
                )
                if has_offered and run_policy is not None:
                    # The recorded policy must be able to re-derive the catalogue
                    # the model was offered from the observed catalogue, or the
                    # trace evidence is inconsistent.
                    expected_offered = ReplaySimulator._retained_candidates(
                        run_policy, event_candidates, uploads_declared=uploads_declared
                    )
                    if stored_offered is not None and candidate_catalog_digest(expected_offered) != stored_offered:
                        raise ValueError(f"Run {run_id} has an offered catalogue digest mismatch")
                    if event.get("offered_count") is not None and int(event["offered_count"]) != len(
                        expected_offered
                    ):
                        raise ValueError(f"Run {run_id} has an offered count mismatch")
                    selected_offered_rank = next(
                        (i for i, item in enumerate(expected_offered) if item.get("id") == selected_id),
                        None,
                    )
                    if recorded_offered_rank is not None and recorded_offered_rank != selected_offered_rank:
                        raise ValueError(f"Run {run_id} has a selected offered-rank mismatch")
                elif has_offered:
                    # Offered catalogue recorded but the policy is unknown; the
                    # recorded coordinate is the only honest value available.
                    selected_offered_rank = recorded_offered_rank
                else:
                    # "recorded" catalogues: offered == observed, so a recorded
                    # offered rank must equal the observed rank.
                    if recorded_offered_rank is not None and recorded_offered_rank != observed_rank:
                        raise ValueError(f"Run {run_id} has a selected offered-rank mismatch")
                    selected_offered_rank = observed_rank
                # ``selected_propensity`` (v0.7.x traces) recorded a model score
                # mislabeled as a propensity; ``behavior_propensity`` records
                # the honest value — None for deterministic argmax selection,
                # the assignment probability for a randomized trial step.
                propensity = event.get("selected_propensity")
                if propensity is None:
                    propensity = event.get("behavior_propensity")
                tr = RecordedTransition(
                    run_id=run_id,
                    task_key=key,
                    state=event["state"],
                    next_state=event["next_state"],
                    selected_id=selected_id,
                    selected_kind=event["selected"].get("kind", ""),
                    candidate_actions=event_candidates,
                    candidate_digest=observed_digest,
                    selected_observed_rank=observed_rank,
                    selected_offered_rank=selected_offered_rank,
                    selected_propensity=float(propensity) if propensity is not None else None,
                    page_changed=bool(event.get("page_changed")),
                    latency_ms=max(0, int(event.get("latency_ms", 0))),
                    model_calls=max(0, int(event.get("model_calls", 1))),
                    tokens=max(0, int(event.get("tokens", 0))),
                    stale_or_failure=max(0, int(event.get("stale_or_failure", 0))),
                    risk_events=max(0, int(event.get("risk_events", 0))),
                    terminal=terminal,
                    verified=verified,
                    catalog=str(event.get("catalog") or "recorded"),
                    offered_digest=stored_offered,
                    offered_count=(
                        max(0, int(event["offered_count"]))
                        if event.get("offered_count") is not None
                        else len(event_candidates)
                    ),
                    experiment=event.get("experiment"),
                    uploads_declared=uploads_declared,
                    causal_override=event.get("causal_override"),
                )
                transitions.append(tr)
            worlds.append(cls(
                key=run_id,
                task_key=key,
                goal=start.get("goal", ""),
                start_state=start.get("state", "ROOT"),
                tcb_version=run_tcb,
                trajectory=tuple(transitions),
                family_key=family_key,
            ))
        return worlds

    def recorded_actions(self) -> tuple[str, ...]:
        return tuple(t.selected_id for t in self.trajectory)


def replay_pool_digest(worlds: Iterable[ReplayWorld]) -> str:
    digests = sorted(world.digest for world in worlds)
    return _stable_hash(json.dumps(digests, separators=(",", ":")))


def split_manifest_digest(splits: dict[str, list[ReplayWorld]]) -> str:
    payload = {name: sorted(world.digest for world in worlds) for name, worlds in sorted(splits.items())}
    return _stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class ReplayResult:
    metrics: ReplayMetrics
    per_world: tuple[dict, ...]


class ReplaySimulator:
    """Conservative replay over realized browser trajectories.

    The exploration policy is allowed to change candidate *allocation* and
    stopping budgets. Replay follows a recorded transition only if the action
    actually taken online would still be offered by the candidate policy. It
    never assumes that the fixed decision model would choose a different action
    merely because the candidate catalogue changed.
    """

    def __init__(self, worlds: Iterable[ReplayWorld], weights: ObjectiveWeights | None = None):
        self.worlds = tuple(worlds)
        self.weights = weights or ObjectiveWeights()

    def evaluate(
        self,
        policy: ExplorationPolicy,
        *,
        max_steps: int | None = None,
        cost_model=None,
        outcome_model=None,
        choice_model=None,
        trial_model=None,
        collect_proposals: bool = False,
    ) -> ReplayResult:
        estimated = bool(cost_model is not None and getattr(cost_model, "reliable", False))
        summaries = [
            self._evaluate_world(
                world,
                policy,
                max_steps=max_steps,
                cost_model=cost_model if estimated else None,
                outcome_model=outcome_model,
                choice_model=choice_model,
                trial_model=trial_model,
                collect_proposals=collect_proposals,
            )
            for world in self.worlds
        ]
        return ReplayResult(metrics=self._aggregate(summaries, estimated=estimated), per_world=tuple(summaries))

    def _evaluate_world(
        self,
        world: ReplayWorld,
        policy: ExplorationPolicy,
        *,
        max_steps: int | None,
        cost_model=None,
        outcome_model=None,
        choice_model=None,
        trial_model=None,
        collect_proposals: bool = False,
    ):
        summary = self._empty_world(world)
        step_limit = min(policy.max_actions, max_steps or policy.max_actions)
        no_progress = 0
        for step_index, transition in enumerate(world.trajectory[:step_limit]):
            offered = self._retained_candidates(
                policy, transition.candidate_actions,
                uploads_declared=transition.uploads_declared,
            )
            summary["offered_candidates"] += len(offered)
            if transition.selected_id not in {item["id"] for item in offered}:
                summary["coverage_misses"] += 1
                # Truncated evidence: the policy cannot even replay the
                # recorded path — not a measured failure of the policy.
                summary["verdict"] = "coverage_miss"
                break
            summary["actions"] += 1
            summary["model_calls"] += transition.model_calls
            summary["tokens"] += transition.tokens
            summary["latency_ms"] += transition.latency_ms
            summary["page_progress"] += int(transition.page_changed)
            summary["stale_or_failures"] += transition.stale_or_failure
            summary["risk_events"] += transition.risk_events
            if cost_model is not None:
                # Anchored efficiency estimate: keep the recorded measurement and
                # adjust only by the learned marginal cost of the offered-count
                # delta. This is hypothesis prioritization, not evidence.
                recorded_offered = transition.offered_count or len(transition.candidate_actions)
                predicted = cost_model.predict(len(offered))
                recorded = cost_model.predict(recorded_offered)
                summary["est_tokens"] += max(0.0, transition.tokens + predicted["tokens"] - recorded["tokens"])
                summary["est_latency_ms"] += max(
                    0.0, transition.latency_ms + predicted["latency_ms"] - recorded["latency_ms"]
                )
            if outcome_model is not None or choice_model is not None:
                selected_overlap = next(
                    (
                        int(c.get("goal_overlap", 0))
                        for c in transition.candidate_actions
                        if c.get("id") == transition.selected_id
                    ),
                    0,
                )
            if outcome_model is not None:
                prediction = outcome_model.predict(
                    kind=transition.selected_kind,
                    goal_overlap=selected_overlap,
                    rank=transition.selected_offered_rank,
                )
                summary["predicted_change"] += prediction["p_page_changed"]
                summary["prediction_samples"] += prediction["n"]
            if choice_model is not None or trial_model is not None:
                # The candidate policy's recomputed offered rank — the rank the
                # recorded action would have had under *this* policy's offered
                # catalogue — so the signal is policy-dependent. Counterfactual
                # proposals annotate the summary only; they are never evidence
                # and never reach a gate.
                offered_rank = next(
                    (
                        i for i, item in enumerate(offered)
                        if item.get("id") == transition.selected_id
                    ),
                    None,
                )
                prediction = None
                if choice_model is not None:
                    prediction = choice_model.predict(
                        kind=transition.selected_kind,
                        goal_overlap=selected_overlap,
                        rank=offered_rank,
                        task_family=world.family_key,
                    )
                    summary["choice_predicted_change"] += prediction["p_progress"]
                    summary["choice_uncertainty"] += prediction["uncertainty"]
                    summary["choice_confident"] += int(prediction["confident"])
                # The causal prior speaks first: a candidate arm with a
                # reliable positive measured effect is a hypothesis already
                # supported by randomized evidence. The observational prior
                # proposes only what trials have not refuted — a reliably
                # non-positive divergence is settled, not re-tested.
                proposal = None
                proposal_source = None
                if trial_model is not None:
                    proposal = trial_model.choose(
                        offered,
                        model_choice={
                            "id": transition.selected_id,
                            "kind": transition.selected_kind,
                            "role": next(
                                (
                                    c.get("role")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            # The effect class recorded when the transition was
                            # compacted (full structural context); older stores
                            # carry none and resolve as an honest wildcard —
                            # never re-classify a compacted candidate whose ctx
                            # is gone, a degraded class could match the wrong
                            # signature cell instead of matching everything.
                            "effect": next(
                                (
                                    c.get("effect")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            "goal_overlap": selected_overlap,
                        },
                        task_family=world.family_key,
                        phase=step_index,
                    )
                    if proposal is not None:
                        proposal_source = trial_model
                if proposal is None and choice_model is not None:
                    proposal = choice_model.choose(offered, task_family=world.family_key)
                    if proposal is not None:
                        proposal_overlap = next(
                            (
                                int(c.get("goal_overlap", 0) or 0)
                                for c in offered
                                if c.get("id") == proposal["id"]
                            ),
                            0,
                        )
                        if trial_model is not None and trial_model.refuted(
                            kind=str(proposal.get("kind") or "unknown"),
                            goal_overlap=proposal_overlap,
                            model_kind=transition.selected_kind,
                            model_overlap=selected_overlap,
                            model_effect=next(
                                (
                                    c.get("effect")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            model_role=next(
                                (
                                    c.get("role")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            model_rank=offered_rank,
                            proposal_effect=next(
                                (
                                    c.get("effect")
                                    for c in offered
                                    if c.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            proposal_role=next(
                                (
                                    c.get("role")
                                    for c in offered
                                    if c.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            proposal_rank=next(
                                (
                                    i
                                    for i, item in enumerate(offered)
                                    if item.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            phase=step_index,
                            task_family=world.family_key,
                        ):
                            proposal = None
                        else:
                            proposal_source = choice_model
                if proposal is not None:
                    summary["choice_proposed_change"] += proposal.get("p_progress") or 0.0
                    summary["choice_proposal_uncertainty"] += proposal.get("uncertainty") or 0.0
                    summary["choice_proposal_confident"] += int(bool(proposal.get("confident")))
                    divergent = proposal["id"] != transition.selected_id
                    summary["choice_divergences"] += int(divergent)
                    if collect_proposals and divergent:
                        # An experiment candidate: the state, the action the
                        # recorded policy actually took, and the action the
                        # prior would substitute — plus the bindings a live
                        # agent revalidates before executing the plan: the
                        # task, the model-choice the hypothesis is conditioned
                        # on, the offered catalogue it diverged inside, and
                        # the policy/model identities that generated it.
                        # Executing it for real — under the same authority
                        # plane — is how a counterfactual hypothesis becomes
                        # signed evidence.
                        expected_delta = proposal.get("expected_delta")
                        if expected_delta is None:
                            expected_delta = (
                                proposal["p_progress"] - prediction["p_progress"]
                                if prediction is not None
                                else 0.0
                            )
                        summary["experiment_proposals"].append({
                            "world": world.key,
                            "task_key": world.task_key,
                            "family_key": world.family_key,
                            "step": step_index,
                            "state": transition.state,
                            "historical": {
                                "id": transition.selected_id,
                                "kind": transition.selected_kind,
                                "effect": next(
                                    (
                                        c.get("effect")
                                        for c in transition.candidate_actions
                                        if c.get("id") == transition.selected_id
                                    ),
                                    None,
                                ),
                                "role": next(
                                    (
                                        c.get("role")
                                        for c in transition.candidate_actions
                                        if c.get("id") == transition.selected_id
                                    ),
                                    None,
                                ),
                                "goal_overlap": selected_overlap,
                                "offered_rank": offered_rank,
                                "propensity": transition.selected_propensity,
                            },
                            # The stamped proposal carries the recorded
                            # signature coordinates — the scheduler and any
                            # later refutation query resolve against exactly
                            # the treatment class this plan was stamped under.
                            "proposal": {
                                **proposal,
                                "effect": next(
                                    (
                                        c.get("effect")
                                        for c in offered
                                        if c.get("id") == proposal["id"]
                                    ),
                                    proposal.get("effect"),
                                ),
                                "role": next(
                                    (
                                        c.get("role")
                                        for c in offered
                                        if c.get("id") == proposal["id"]
                                    ),
                                    proposal.get("role"),
                                ),
                            },
                            "proposal_offered_rank": next(
                                (
                                    i
                                    for i, item in enumerate(offered)
                                    if item.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            "offered_catalogue_digest": candidate_catalog_digest(offered),
                            "policy_behavior_digest": policy.behavior_digest,
                            # The digest of whichever prior generated this
                            # hypothesis — observational ChoiceModel or causal
                            # TrialChoiceModel — bound into the stamped plan,
                            # plus the explicit channel so a live agent can
                            # tell causal provenance from correlational.
                            "choice_model_digest": getattr(proposal_source, "digest", None),
                            "origin": (
                                "randomized" if proposal_source is trial_model else "observational"
                            ),
                            "causal_model_digest": (
                                getattr(trial_model, "digest", None)
                                if proposal_source is trial_model
                                else None
                            ),
                            "selected_prior": (
                                {
                                    "p_progress": prediction["p_progress"],
                                    "uncertainty": prediction["uncertainty"],
                                    "confident": prediction["confident"],
                                    "source": prediction.get("source"),
                                }
                                if prediction is not None
                                else None
                            ),
                            "expected_delta": expected_delta,
                            "expected_uncertainty": (
                                proposal.get("uncertainty")
                                if proposal.get("uncertainty") is not None
                                else (prediction or {}).get("uncertainty")
                            ),
                            "created_at_ms": int(time.time() * 1000),
                        })
            if transition.selected_kind != "wait" and not transition.page_changed:
                no_progress += 1
            else:
                no_progress = 0
            if transition.terminal:
                summary["status"] = transition.terminal
                summary["verified"] = transition.verified
                summary["success"] = transition.terminal == "done" and transition.verified
                summary["verdict"] = (
                    "verified_success" if summary["success"] else "measured_terminal"
                )
                break
            if no_progress >= policy.no_progress_window:
                summary["status"] = "blocked"
                summary["verdict"] = "blocked"
                break
        summary["covered_steps"] = summary["actions"]
        return summary

    @staticmethod
    def _retained_candidates(
        policy: ExplorationPolicy,
        candidates: Iterable[dict],
        *,
        uploads_declared: bool = False,
    ) -> list[dict]:
        """Re-derive the offered catalogue; must match model.candidate_actions.

        Stored ``goal_overlap`` is computed on sanitized labels, as is live
        scoring, so the replayed catalogue is identical to what the model was
        offered online. Candidates lacking ``node`` metadata (older schema)
        are treated as unique nodes so ``duplicate_node_cap`` cannot merge them.
        ``uploads_declared`` is the run's recorded allowlist state: an upload
        target is only offerable when the operator declared files — replay
        must apply the same filter it cannot otherwise reconstruct.
        """
        candidates = [dict(c) for c in candidates if c.get("id")]
        controls = [c for c in candidates if c.get("kind") not in {"click", "fill", "select", "upload"}]
        regular = []
        for index, candidate in enumerate(candidates):
            if candidate.get("kind") not in {"click", "fill", "select", "upload"}:
                continue
            # Mirrors the live upload_names filter: observed evidence, never
            # offerable to the model without declared files.
            if candidate.get("kind") == "upload" and not uploads_declared:
                continue
            overlap = max(0, int(candidate.get("goal_overlap", 0)))
            if overlap < policy.min_goal_overlap:
                continue
            regular.append((index, candidate, overlap))

        def score(pair):
            index, action, overlap = pair
            bonus = {
                "fill": policy.fill_bonus,
                "select": policy.select_bonus,
                "click": policy.click_bonus,
            }.get(action.get("kind"), 0.0)
            overlap_term = (overlap ** policy.overlap_exponent) * policy.goal_overlap_weight
            return overlap_term + bonus - index * policy.order_penalty

        def node_group(index, action):
            node = action.get("node")
            return ("n", node) if node is not None else ("~", index)

        ranked = sorted(regular, key=score, reverse=True)
        quotas = {
            "click": policy.click_quota,
            "fill": policy.fill_quota,
            "select": policy.select_quota,
            # Mirrors live quota_for("upload") — the whole regular budget.
            "upload": policy.model_action_limit,
        }
        regular_budget = max(0, policy.model_action_limit - len(controls))
        counts = {kind: 0 for kind in quotas}
        node_counts: dict = {}
        selected = []
        used = set()
        for index, action, _overlap in ranked:
            kind = action["kind"]
            group = node_group(index, action)
            if (
                counts[kind] >= quotas[kind]
                or len(selected) >= regular_budget
                or node_counts.get(group, 0) >= policy.duplicate_node_cap
            ):
                continue
            selected.append((index, action))
            used.add(index)
            counts[kind] += 1
            node_counts[group] = node_counts.get(group, 0) + 1
        if len(selected) < regular_budget:
            for index, action, _overlap in ranked:
                if index in used:
                    continue
                group = node_group(index, action)
                if node_counts.get(group, 0) >= policy.duplicate_node_cap:
                    continue
                selected.append((index, action))
                used.add(index)
                node_counts[group] = node_counts.get(group, 0) + 1
                if len(selected) >= regular_budget:
                    break
        result = [action for _, action in sorted(selected, key=lambda pair: pair[0])]
        result.extend(controls)
        return result[: policy.model_action_limit]

    @staticmethod
    def _empty_world(world: ReplayWorld, coverage_miss: int = 0):
        return {
            "world": world.key,
            "task_key": world.task_key,
            "success": False,
            "verified": False,
            "status": None,
            "actions": 0,
            "model_calls": 0,
            "tokens": 0,
            "latency_ms": 0,
            "offered_candidates": 0,
            "page_progress": 0,
            "stale_or_failures": 0,
            "risk_events": 0,
            "coverage_misses": coverage_miss,
            "covered_steps": 0,
            # Default verdict: a world that ends without any explicit exit
            # ran out of recorded evidence — inconclusive, never a failure.
            "verdict": "coverage_miss" if coverage_miss else "inconclusive",
            "est_tokens": 0.0,
            "est_latency_ms": 0.0,
            "predicted_change": 0.0,
            "prediction_samples": 0,
            "choice_predicted_change": 0.0,
            "choice_proposed_change": 0.0,
            "choice_uncertainty": 0.0,
            "choice_confident": 0,
            "choice_divergences": 0,
            "choice_proposal_uncertainty": 0.0,
            "choice_proposal_confident": 0,
            "experiment_proposals": [],
        }

    def _aggregate(self, summaries: list[dict], *, estimated: bool = False) -> ReplayMetrics:
        w = self.weights
        worlds = len(summaries)
        successes = sum(int(s["success"]) for s in summaries)
        verified = sum(int(s["verified"] and s["success"]) for s in summaries)
        actions = sum(s["actions"] for s in summaries)
        model_calls = sum(s["model_calls"] for s in summaries)
        tokens = sum(s["tokens"] for s in summaries)
        latency = sum(s["latency_ms"] for s in summaries)
        offered = sum(s["offered_candidates"] for s in summaries)
        progress = sum(s["page_progress"] for s in summaries)
        failures = sum(s["stale_or_failures"] for s in summaries)
        risk = sum(s["risk_events"] for s in summaries)
        misses = sum(s["coverage_misses"] for s in summaries)
        potential = actions + misses
        coverage = actions / potential if potential else (1.0 if worlds else 0.0)
        # Verdict tally (Phase 13): every world reports exactly one verdict —
        # truncated evidence (coverage_miss / inconclusive) is counted apart
        # from measured endings, never silently read as failure.
        verdicts = {v: 0 for v in REPLAY_VERDICTS}
        for s in summaries:
            verdict = s.get("verdict")
            verdicts[verdict if verdict in verdicts else "inconclusive"] += 1

        def score(token_total, latency_total):
            return (
                w.success * successes
                + w.verified * verified
                + w.page_progress * progress
                - w.latency_seconds * (latency_total / 1000)
                - w.action * actions
                - w.model_call * model_calls
                - w.thousand_tokens * (token_total / 1000)
                - w.offered_candidate * offered
                - w.stale_or_failure * failures
                - w.risk_event * risk
                - w.coverage_miss * misses
            )

        estimated_score = None
        est_tokens = est_latency = 0
        if estimated:
            est_tokens = sum(s["est_tokens"] for s in summaries)
            est_latency = sum(s["est_latency_ms"] for s in summaries)
            estimated_score = score(est_tokens, est_latency)
        return ReplayMetrics(
            score=score(tokens, latency),
            worlds=worlds,
            successes=successes,
            verified_successes=verified,
            actions=actions,
            model_calls=model_calls,
            tokens=tokens,
            latency_ms=latency,
            offered_candidates=offered,
            page_progress=progress,
            stale_or_failures=failures,
            risk_events=risk,
            coverage_misses=misses,
            coverage=coverage,
            estimated_tokens=int(round(est_tokens)),
            estimated_latency_ms=int(round(est_latency)),
            estimated_score=estimated_score,
            verdicts=verdicts,
        )


def split_worlds(worlds: Iterable[ReplayWorld]):
    """Deterministic, disjoint split that keeps each task family in one partition.

    A family is the explicit ``task_family`` recorded at run start, falling back
    to the goal-hash task key for traces without family metadata.
    """
    groups: dict[str, list[ReplayWorld]] = defaultdict(list)
    for world in worlds:
        groups[world.family_key or world.task_key].append(world)
    ordered = sorted(groups.items(), key=lambda item: (int(_stable_hash(item[0])[:16], 16), item[0]))
    n = len(ordered)
    if n == 0:
        return {"train": [], "validation": [], "holdout": []}
    if n == 1:
        assignments = (n, 0, 0)
    elif n == 2:
        assignments = (1, 1, 0)
    else:
        holdout_n = max(1, round(n * 0.15))
        validation_n = max(1, round(n * 0.15))
        if holdout_n + validation_n >= n:
            holdout_n = validation_n = 1
        assignments = (n - validation_n - holdout_n, validation_n, holdout_n)
    train_n, validation_n, _holdout_n = assignments
    train_groups = ordered[:train_n]
    validation_groups = ordered[train_n : train_n + validation_n]
    holdout_groups = ordered[train_n + validation_n :]

    def flatten(items):
        return [world for _key, group in items for world in group]

    return {
        "train": flatten(train_groups),
        "validation": flatten(validation_groups),
        "holdout": flatten(holdout_groups),
    }
