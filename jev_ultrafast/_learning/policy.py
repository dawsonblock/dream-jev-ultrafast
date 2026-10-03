"""_learning.policy — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable

    from .causal import TrialChoiceModel
    from .observational import ChoiceModel

__all__ = [
    'CausalChoicePolicy',
]


@dataclass(frozen=True)
class CausalChoicePolicy:
    """Bounded combination of the base decision score with learned priors.

    Modes are *operator-gated* — the policy never promotes its own mode, and
    a mode change is a configuration decision, not a learned one:

    - ``shadow`` — produce the ranking as an annotation only; nothing it
      prefers can influence execution.
    - ``canary`` — its top proposal may be used as the *experimental*
      candidate under the existing randomized assignment (recorded
      propensity, real authority plane, at most one trial per run). Pooled
      randomized evidence may nominate here: a canary is exactly how pooled
      evidence earns context-specific support.
    - ``active`` — a *beneficial*, randomized-established proposal may be
      used as the executed choice, still bounded to the offered catalogue and
      still passing effect classification, payload review, approvals, and the
      browser guards. Active execution additionally requires evidence from a
      *context-specific* stratum — ``active_trial_levels`` defaults to
      ``family+site``/``site``/``family`` and never includes ``pooled``:
      pooled randomized evidence generalizes over every task at once and is
      sufficient to schedule an experiment, never to silently substitute a
      choice in an unseen context.

    Every entry carries structured provenance, not a single overloaded
    label: ``evidence_origin`` (``randomized`` | ``observational`` |
    ``none``), ``trial_level``, ``support_status``, ``effect_status``, and
    ``generalization_level`` — origin describes where the evidence came
    from *independently* of whether it sufficed, so a thin pooled contrast
    is ``evidence_origin=randomized, trial_level=pooled,
    support_status=insufficient``, never mislabeled ``randomized`` as if it
    were context-specific. The legacy ``source`` label is preserved with
    the same corrected semantics (pooled stays ``pooled_randomized``
    whether or not support sufficed).

    Every entry also carries ``execution_blocker``: the named reason it
    cannot execute deterministically (``no_causal_evidence``,
    ``insufficient_support``, ``unresolved``, ``below_effect_threshold``,
    ``missing_probability``, ``safety_regression``, ``pooled_only``,
    ``unsupported_stratum``, ``unverified_generalization``, ``harmful``),
    or ``None``.

    Scores are separated by purpose instead of one overloaded value:
    ``causal_score`` (weighted measured effect), ``observational_score``
    (weighted associational prior), ``experiment_priority_score`` (what a
    trial of this candidate would teach — importance × remaining
    uncertainty), and ``deployment_score`` (the measured causal effect the
    active path actually deploys on). ``score`` remains the legacy
    shadow-mode combination; in ``active`` mode ordering follows causal
    terms only — observational evidence can annotate an entry but never
    outrank superior causal evidence.
    """

    mode: str = "shadow"
    trial_model: TrialChoiceModel | None = None
    choice_model: ChoiceModel | None = None
    w_causal: float = 1.0
    w_observational: float = 0.5
    min_causal_delta: float = 0.0
    # Strata whose established randomized evidence may drive an *active*
    # override. Pooled evidence is hypothesis/nomination authority, not
    # execution authority — an operator may weaken this list explicitly, but
    # never accidentally: unknown level names fail closed at construction.
    active_trial_levels: tuple[str, ...] = ("family+site", "site", "family")
    # Generalization levels whose evidence may drive an *active* override.
    # ``exact`` — this exact action identity was itself the randomized
    # candidate arm in the answering stratum — is the only default: a
    # never-randomized action cannot deploy solely because its treatment
    # class looked beneficial. ``same_context_class`` evidence earns active
    # authority only when a contextual canary of this action is on record —
    # derived from the trial store (``action_randomized_in_context``) or
    # listed in ``confirmed_action_keys`` for confirmations evidenced
    # outside this model. An operator may widen the set explicitly; unknown
    # level names fail closed at construction.
    active_generalization_levels: tuple[str, ...] = ("exact",)
    confirmed_action_keys: tuple[str, ...] = ()

    MODES = ("shadow", "canary", "active")
    TRIAL_LEVELS = ("family+site", "site", "family", "pooled")
    GENERALIZATION_LEVELS = (
        "exact",
        "same_context_class",
        "cross_context_class",
        "pooled_class",
        "none",
    )

    def __post_init__(self):
        if self.mode not in self.MODES:
            raise ValueError(f"CausalChoicePolicy mode must be one of {self.MODES}")
        unknown = {str(level) for level in self.active_trial_levels} - set(self.TRIAL_LEVELS)
        if unknown:
            raise ValueError(
                f"active_trial_levels entries must be known trial levels "
                f"{self.TRIAL_LEVELS}; got {sorted(unknown)}"
            )
        unknown_gen = {
            str(level) for level in self.active_generalization_levels
        } - set(self.GENERALIZATION_LEVELS)
        if unknown_gen:
            raise ValueError(
                f"active_generalization_levels entries must be known "
                f"generalization levels {self.GENERALIZATION_LEVELS}; "
                f"got {sorted(unknown_gen)}"
            )
        object.__setattr__(
            self, "_confirmed_keys", {str(k) for k in self.confirmed_action_keys}
        )

    def rank(
        self,
        candidates: Iterable[dict],
        *,
        model_choice: dict,
        task_family: str | None = None,
        site: str | None = None,
        phase=None,
    ) -> dict:
        candidates = list(candidates)
        causal = {}
        if self.trial_model is not None:
            causal = {
                entry["id"]: entry
                for entry in self.trial_model.rank(
                    candidates,
                    model_choice=model_choice,
                    task_family=task_family,
                    site=site,
                    phase=phase,
                )
            }
        mc_rank = next(
            (i for i, c in enumerate(candidates)
             if c.get("id") == model_choice.get("id")),
            None,
        )
        entries = []
        for index, candidate in enumerate(candidates):
            if candidate.get("id") == model_choice.get("id"):
                continue
            causal_entry = causal.get(candidate.get("id"))
            observational = None
            if self.choice_model is not None:
                observational = self.choice_model.predict(
                    kind=str(candidate.get("kind") or "unknown"),
                    goal_overlap=int(candidate.get("goal_overlap", 0) or 0),
                    rank=index,
                    task_family=task_family,
                )
            # The causal channel omits established-harmful divergences rather
            # than ranking them; a missing entry is either "no evidence" or
            # "refuted" — ask once, so the blocker says which.
            refuted = bool(
                causal_entry is None
                and self.trial_model is not None
                and callable(getattr(self.trial_model, "refuted", None))
                and self.trial_model.refuted(
                    kind=str(candidate.get("kind") or "unknown"),
                    goal_overlap=candidate.get("goal_overlap"),
                    model_kind=str(model_choice.get("kind") or "unknown"),
                    model_overlap=model_choice.get("goal_overlap"),
                    model_effect=model_choice.get("effect"),
                    model_role=model_choice.get("role"),
                    model_rank=mc_rank,
                    proposal_effect=candidate.get("effect"),
                    proposal_role=candidate.get("role"),
                    proposal_rank=index,
                    phase=phase,
                    task_family=task_family,
                    site=site,
                )
            )
            supported = bool(causal_entry and causal_entry.get("support_sufficient"))
            # Provenance by origin and level, independent of significance:
            # a thin pooled contrast is randomized evidence at the pooled
            # level — it never wears the context-specific label.
            if causal_entry is not None:
                source = (
                    "pooled_randomized"
                    if causal_entry.get("trial_level") == "pooled"
                    else "randomized"
                )
            else:
                source = "observational"
            causal_score = (
                self.w_causal * float(causal_entry.get("expected_delta") or 0.0)
                if causal_entry is not None
                else 0.0
            )
            observational_score = (
                self.w_observational * (float(observational["p_progress"]) - 0.5)
                if observational is not None and observational.get("p_progress") is not None
                else 0.0
            )
            # Experiment priority: what executing this candidate as a trial
            # would teach — the importance of the divergence scaled by how
            # much uncertainty remains. Not a causal estimate, not a
            # deployment score: a prioritization annotation only.
            importance = abs(causal_score) or abs(observational_score)
            uncertainty = (
                float(causal_entry["uncertainty"])
                if causal_entry is not None and causal_entry.get("uncertainty") is not None
                else 0.5
            )
            experiment_priority = max(importance, 0.02) * uncertainty
            # Deployment score: the measured causal effect only — the
            # quantity active mode actually deploys on. Observational
            # evidence contributes nothing here.
            deployment_score = (
                float(causal_entry.get("expected_delta") or 0.0)
                if causal_entry is not None
                else None
            )
            blocker = self._execution_blocker(causal_entry, refuted=refuted)
            entries.append({
                "id": candidate.get("id"),
                "kind": candidate.get("kind"),
                "score": causal_score + observational_score,
                "causal_score": causal_score,
                "observational_score": observational_score,
                "experiment_priority_score": experiment_priority,
                "deployment_score": deployment_score,
                "source": source,
                # Structured provenance dimensions (Phase 7): origin, level,
                # sufficiency, sign, and generalization are separate facts —
                # never a single overloaded label.
                "evidence_origin": "randomized" if causal_entry is not None else "observational",
                "trial_level": causal_entry.get("trial_level") if causal_entry else None,
                "support_status": (
                    "sufficient" if supported else ("insufficient" if causal_entry else "none")
                ),
                "effect_status": (causal_entry or {}).get("effect_status"),
                "generalization_level": (causal_entry or {}).get("generalization_level"),
                "action_key": (causal_entry or {}).get("action_key"),
                "exact_action_randomized": (causal_entry or {}).get(
                    "exact_action_randomized"
                ),
                "action_randomized_anywhere": (causal_entry or {}).get(
                    "action_randomized_anywhere"
                ),
                "action_randomized_in_context": (
                    causal_entry or {}
                ).get("action_randomized_in_context"),
                "hypothesis_registered": (causal_entry or {}).get(
                    "hypothesis_registered"
                ),
                "hypothesis_count": (causal_entry or {}).get("hypothesis_count"),
                "causal": causal_entry,
                "observational": observational,
                "execution_blocker": blocker,
                "executable": bool(self.mode == "active" and blocker is None),
            })
        tiers = {"beneficial": 0, "unresolved": 1, "insufficient_data": 2}
        if self.mode == "active":
            # Active ordering uses causal deployment terms only: established
            # effect, then uncertainty, then verified operational cost.
            # The observational channel annotates but cannot reorder.
            entries.sort(
                key=lambda entry: (
                    0 if entry["executable"] else 1,
                    tiers.get(entry["effect_status"], 3),
                    -(entry["deployment_score"] or 0.0),
                    (
                        entry["causal"].get("uncertainty")
                        if entry["causal"] and entry["causal"].get("uncertainty") is not None
                        else float("inf")
                    ),
                    -float((entry["causal"] or {}).get("utility_delta") or 0.0),
                    str(entry["id"]),
                )
            )
        elif self.mode == "canary":
            # Canary ordering is experiment value: which candidate would a
            # randomized trial teach the most about. Settled-harmful and
            # safety-regressing entries are still reported but nominated
            # last — spending a real arm re-testing them is never warranted.
            entries.sort(
                key=lambda entry: (
                    1 if entry["execution_blocker"] in {"harmful", "safety_regression"} else 0,
                    -entry["experiment_priority_score"],
                    tiers.get(entry["effect_status"], 3),
                    str(entry["id"]),
                )
            )
        else:
            entries.sort(
                key=lambda entry: (
                    tiers.get(entry["effect_status"], 3),
                    -entry["score"],
                    str(entry["id"]),
                )
            )
        return {
            "mode": self.mode,
            "shadow": self.mode == "shadow",
            "proposals": entries,
            "abstain": not entries,
        }

    def _execution_blocker(
        self, causal_entry: dict | None, *, refuted: bool = False
    ) -> str | None:
        """The named reason this candidate cannot execute under ``active`` —
        or ``None`` when every active-evidence requirement holds.

        Requirements, in evaluation order: the causal channel must have
        answered at all (an established-harmful divergence is reported
        ``harmful``, not merely absent); its estimate must carry enough
        randomized support; the established sign must be beneficial; the
        effect must clear the configured practical threshold; the contrast
        must supply a valid finite causal probability; and the evidence must
        come from a context-specific stratum this policy trusts for active
        use — pooled randomized evidence is a hypothesis source, not
        execution authority; and the evidence must generalize to *this
        exact action identity* — class-level evidence for an action that
        was itself never randomized is a canary hypothesis, not active
        authority, unless the confirmed list records a completed
        contextual canary for that identity.
        """
        if causal_entry is None:
            return "harmful" if refuted else "no_causal_evidence"
        if not causal_entry.get("support_sufficient"):
            return "insufficient_support"
        if causal_entry.get("effect_status") != "beneficial":
            return "unresolved"
        if not (float(causal_entry.get("expected_delta") or 0.0) > self.min_causal_delta):
            return "below_effect_threshold"
        if causal_entry.get("p_progress") is None:
            return "missing_probability"
        # Safety measurements are constraints, not utility: a measured
        # regression on any safety endpoint (authority burden, guard
        # failures, indeterminate executions) is a veto — no success delta,
        # latency gain, or token saving can compensate for it. ``None``
        # (unmeasured) does not veto: it is absence of evidence, and the
        # support gates already govern what absent evidence may do.
        if causal_entry.get("safety_regression"):
            return "safety_regression"
        trial_level = causal_entry.get("trial_level")
        if trial_level not in self.active_trial_levels:
            return "pooled_only" if trial_level == "pooled" else "unsupported_stratum"
        # Exact-action authority: the answering stratum must have
        # randomized *this action identity* (``generalization_level`` in the
        # allowed set — ``exact`` by default), or class-level evidence must
        # be bridged by a contextual canary of this action — either derived
        # from the trial store itself (``action_randomized_in_context``: the
        # action was itself the randomized arm inside the answering
        # context's scope, which *is* a completed contextual canary — the
        # "successful" half is supplied by the established-beneficial
        # contrast this gate already required) or recorded explicitly in
        # ``confirmed_action_keys`` for confirmations whose evidence lives
        # outside this model. ``None`` (an entry built before generalization
        # annotation existed) or an unrecognized value fails closed.
        level = causal_entry.get("generalization_level")
        if level is not None and str(level) not in self.GENERALIZATION_LEVELS:
            return "unverified_generalization"
        if str(level or "") not in set(self.active_generalization_levels):
            confirmed = bool(causal_entry.get("action_randomized_in_context")) or (
                causal_entry.get("action_key") is not None
                and causal_entry.get("action_key")
                in getattr(self, "_confirmed_keys", set())
            )
            if not (level == "same_context_class" and confirmed):
                return "unverified_generalization"
        return None

    def proposal(
        self,
        candidates: Iterable[dict],
        *,
        model_choice: dict,
        task_family: str | None = None,
        site: str | None = None,
        phase=None,
    ) -> dict | None:
        """The top entry when this mode may act on it, else ``None``.

        ``canary`` returns the top entry for the randomized-assignment path;
        ``active`` returns it only when it is a randomized-established
        beneficial proposal; ``shadow`` always returns ``None`` — shadow mode
        observes, it does not act.
        """
        ranking = self.rank(
            candidates, model_choice=model_choice, task_family=task_family,
            site=site, phase=phase,
        )
        if not ranking["proposals"] or self.mode == "shadow":
            return None
        if self.mode == "active":
            top = ranking["proposals"][0]
            if not top["executable"]:
                return None
        else:
            # canary: nominate the first entry that is neither refuted nor
            # safety-regressing — an established-harmful divergence is
            # settled and negative, and preferentially nominating a
            # safety-suspect arm is not how evidence against it should be
            # gathered. Spending a real randomized trial re-testing either
            # is never warranted.
            top = next(
                (
                    entry
                    for entry in ranking["proposals"]
                    if entry.get("execution_blocker")
                    not in {"harmful", "safety_regression"}
                ),
                None,
            )
            if top is None:
                return None
        causal = top.get("causal") or {}
        observational = top.get("observational") or {}
        causal_p = causal.get("p_progress")
        observational_p = observational.get("p_progress")
        if causal.get("expected_delta") is not None and causal.get("support_sufficient"):
            # Supported randomized evidence governs the proposal's implied
            # success probability. It is the measured control rate plus the
            # measured effect, propagated from the estimation layer — never
            # reconstructed from a fabricated 0.5 baseline, and never
            # silently replaced by the observational probability when the
            # causal contrast cannot supply one.
            p_progress = causal_p
            uncertainty = causal.get("uncertainty")
        else:
            p_progress = observational_p
            uncertainty = observational.get("uncertainty")
        return {
            "id": top["id"],
            "kind": top["kind"],
            "p_progress": p_progress,
            "causal_p_progress": causal_p,
            "observational_p_progress": observational_p,
            "uncertainty": uncertainty,
            "confident": bool(causal.get("support_sufficient") or observational.get("confident")),
            "expected_delta": causal.get("expected_delta"),
            "control_p": causal.get("control_p"),
            "effect_status": causal.get("effect_status"),
            "safety_regression": causal.get("safety_regression"),
            "trial_level": causal.get("trial_level"),
            "signature_level": causal.get("signature_level"),
            "generalization_level": causal.get("generalization_level"),
            "action_key": causal.get("action_key"),
            "exact_action_randomized": causal.get("exact_action_randomized"),
            "action_randomized_anywhere": causal.get("action_randomized_anywhere"),
            "action_randomized_in_context": causal.get("action_randomized_in_context"),
            "hypothesis_registered": causal.get("hypothesis_registered"),
            "hypothesis_count": causal.get("hypothesis_count"),
            "evidence_origin": top.get("evidence_origin"),
            "support_status": top.get("support_status"),
            "deployment_score": top.get("deployment_score"),
            "experiment_priority_score": top.get("experiment_priority_score"),
            "source": top["source"],
            "mode": self.mode,
            "executable": top["executable"],
            "execution_blocker": top.get("execution_blocker"),
        }
