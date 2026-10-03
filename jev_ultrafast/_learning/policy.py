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

    Every entry carries provenance: ``randomized`` (supported randomized
    evidence at a specific stratum), ``pooled_randomized`` (randomized
    evidence that had to back off to the pooled stratum), or
    ``observational`` (trajectory association only). The two evidence
    channels are never silently mixed — a combined score reports which
    channel each component came from. Every entry also carries
    ``execution_blocker``: the named reason it cannot execute
    deterministically (``no_causal_evidence``, ``insufficient_support``,
    ``unresolved``, ``below_effect_threshold``, ``missing_probability``,
    ``pooled_only``, ``unsupported_stratum``), or ``None``.
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

    MODES = ("shadow", "canary", "active")
    TRIAL_LEVELS = ("family+site", "site", "family", "pooled")

    def __post_init__(self):
        if self.mode not in self.MODES:
            raise ValueError(f"CausalChoicePolicy mode must be one of {self.MODES}")
        unknown = {str(level) for level in self.active_trial_levels} - set(self.TRIAL_LEVELS)
        if unknown:
            raise ValueError(
                f"active_trial_levels entries must be known trial levels "
                f"{self.TRIAL_LEVELS}; got {sorted(unknown)}"
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
            if supported:
                source = (
                    "pooled_randomized"
                    if causal_entry.get("trial_level") == "pooled"
                    else "randomized"
                )
            elif causal_entry is not None:
                source = "randomized"
            elif observational is not None and observational.get("confident"):
                source = "observational"
            else:
                source = "observational"
            score = 0.0
            if causal_entry is not None:
                score += self.w_causal * float(causal_entry.get("expected_delta") or 0.0)
            if observational is not None and observational.get("p_progress") is not None:
                score += self.w_observational * (float(observational["p_progress"]) - 0.5)
            entries.append({
                "id": candidate.get("id"),
                "kind": candidate.get("kind"),
                "score": score,
                "source": source,
                "causal": causal_entry,
                "observational": observational,
                "execution_blocker": self._execution_blocker(
                    causal_entry, refuted=refuted
                ),
                "executable": bool(
                    self.mode == "active"
                    and self._execution_blocker(causal_entry, refuted=refuted)
                    is None
                ),
            })
        tiers = {"beneficial": 0, "unresolved": 1, "insufficient_data": 2}
        entries.sort(
            key=lambda entry: (
                tiers.get((entry["causal"] or {}).get("effect_status"), 3),
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
        execution authority.
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
        trial_level = causal_entry.get("trial_level")
        if trial_level not in self.active_trial_levels:
            return "pooled_only" if trial_level == "pooled" else "unsupported_stratum"
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
            # canary: nominate the first non-refuted entry — an established-
            # harmful divergence is settled and negative; spending a real
            # randomized trial re-testing it is never warranted.
            top = next(
                (
                    entry
                    for entry in ranking["proposals"]
                    if entry.get("execution_blocker") != "harmful"
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
            "trial_level": causal.get("trial_level"),
            "signature_level": causal.get("signature_level"),
            "source": top["source"],
            "mode": self.mode,
            "executable": top["executable"],
            "execution_blocker": top.get("execution_blocker"),
        }
