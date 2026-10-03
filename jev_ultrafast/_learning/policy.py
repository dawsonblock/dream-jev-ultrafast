"""_learning.policy — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..signing import (
    CONFIRMATION_DOMAIN,
    verify_keys_from_env,
    verify_signature,
)
from .causal import CounterfactualTrials

if TYPE_CHECKING:
    from typing import Iterable

    from .causal import TrialChoiceModel
    from .observational import ChoiceModel

__all__ = [
    'CONFIRMATION_VERIFY_KEYS_ENV',
    'CausalChoicePolicy',
    'action_confirmation_digest',
    'mint_action_confirmation',
]


CONFIRMATION_VERIFY_KEYS_ENV = "JEV_CONFIRMATION_VERIFY_KEYS"


# The shrinkage prior weight: the class estimate enters worth one
# MIN_ESS cell of evidence against the action's own thin contrast.
_MIN_ESS = CounterfactualTrials.MIN_ESS


def mint_action_confirmation(
    signer,
    *,
    action_key: str,
    task_family: str | None = None,
    site: str | None = None,
    evidence_digest: str | None = None,
    confirmed_at_ms: int | None = None,
) -> dict:
    """Mint a signed ``jev-action-confirmation/1`` record.

    The operator's attestation that a contextual canary of this action
    completed outside this model — bound to the action identity, an
    optional task-family/site scope, and the evidence digest it cites.
    ``signer`` is any object exposing ``key_id`` and ``sign_hex`` (an
    ``EvidenceSigner``, or a KMS/HSM-backed shim).
    """
    import time

    record = {
        "schema": "jev-action-confirmation/1",
        "action_key": str(action_key),
        "task_family": task_family,
        "site": site,
        "evidence_digest": evidence_digest,
        "confirmed_at_ms": int(
            confirmed_at_ms if confirmed_at_ms is not None else time.time() * 1000
        ),
        "key_id": signer.key_id,
    }
    record["signature"] = signer.sign_hex(
        action_confirmation_digest(record), domain=CONFIRMATION_DOMAIN
    )
    return record


def action_confirmation_digest(record: dict) -> str:
    """Canonical digest of a signed action-confirmation record.

    The digest covers the attestation — the action identity, its context
    scope, the evidence it cites, and when it was made — everything except
    the signature fields themselves.
    """
    import hashlib

    payload = {
        key: record.get(key)
        for key in (
            "schema", "action_key", "task_family", "site",
            "evidence_digest", "confirmed_at_ms",
        )
        if key in record
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    # Signed contextual-canary confirmations (schema
    # ``jev-action-confirmation/1``): each record attests that a completed
    # canary of this action identity exists, bound to an evidence digest
    # and optionally scoped to a task family/site. When
    # ``confirmation_verify_keys`` is configured — explicitly or through
    # ``JEV_CONFIRMATION_VERIFY_KEYS`` — only cryptographically verified
    # records count and the unsigned ``confirmed_action_keys`` compatibility
    # path is closed; leaving everything unconfigured keeps the legacy
    # unsigned behavior (documented compatibility, never silently strict).
    action_confirmations: tuple[dict, ...] = ()
    confirmation_verify_keys: tuple[str, ...] | None = None

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
        verify_keys = self.confirmation_verify_keys
        if verify_keys is None:
            verify_keys = tuple(sorted(verify_keys_from_env(CONFIRMATION_VERIFY_KEYS_ENV)))
        verify_keys = tuple(str(k).strip() for k in verify_keys if str(k).strip())
        object.__setattr__(self, "confirmation_verify_keys", verify_keys)
        if verify_keys and self.confirmed_action_keys:
            raise ValueError(
                "confirmed_action_keys is the unsigned compatibility path — "
                "it cannot be combined with configured confirmation_verify_keys "
                "(use signed action_confirmations instead)"
            )
        if self.action_confirmations and not verify_keys:
            raise ValueError(
                "action_confirmations require confirmation_verify_keys — "
                "an unverified signed record would be an untrusted bypass"
            )
        confirmations = []
        for record in self.action_confirmations:
            record = dict(record)
            if record.get("schema") != "jev-action-confirmation/1":
                raise ValueError("action_confirmations records must carry schema jev-action-confirmation/1")
            if not isinstance(record.get("action_key"), str) or not record["action_key"]:
                raise ValueError("action_confirmations records require an action_key")
            digest = action_confirmation_digest(record)
            ok = verify_signature(
                str(record.get("key_id") or ""),
                digest,
                str(record.get("signature") or ""),
                domain=CONFIRMATION_DOMAIN,
            ) and str(record.get("key_id") or "") in set(verify_keys)
            if not ok:
                raise ValueError(
                    "action_confirmation failed signature verification — "
                    "a forged or unsigned confirmation cannot bridge class evidence"
                )
            confirmations.append(record)
        object.__setattr__(self, "action_confirmations", tuple(confirmations))
        object.__setattr__(
            self, "_confirmed_keys", {str(k) for k in self.confirmed_action_keys}
        )

    def _action_confirmed(
        self, causal_entry: dict, task_family: str | None, site: str | None
    ) -> bool:
        """True when a contextual canary of this action is on record.

        Two channels, one semantics — the store's own randomized record
        (``action_randomized_in_context``: the action *was* the candidate
        arm inside the answering scope) or an external attestation. The
        external path is cryptographically bound when verification keys
        are configured; without them the unsigned ``confirmed_action_keys``
        compatibility set applies. A signed record's declared scope must
        also cover this query — a confirmation for a different task family
        or site does not travel.
        """
        if causal_entry.get("action_randomized_in_context"):
            return True
        key = causal_entry.get("action_key")
        if key is None:
            return False
        if not self.confirmation_verify_keys:
            return key in self._confirmed_keys
        # Signed path: unsigned raw keys are closed the moment verification
        # keys exist — construction already refused to combine them.
        family = str(task_family or "").strip().lower()
        host = str(site or "").strip().lower()
        for record in self.action_confirmations:
            if record.get("action_key") != key:
                continue
            rf = record.get("task_family")
            rs = record.get("site")
            if rf and str(rf).strip().lower() != family:
                continue
            if rs and str(rs).strip().lower() != host:
                continue
            return True
        return False

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
            blocker = self._execution_blocker(
                causal_entry, refuted=refuted, task_family=task_family, site=site,
            )
            deployment = None
            if blocker is None and causal_entry is not None:
                deployment = self._deployment_estimate(causal_entry)
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
                # The estimate the executable path actually deploys on —
                # which estimator produced it and its deployed delta /
                # implied probability. ``None`` when the entry cannot
                # execute: a blocked candidate has no deployed estimate.
                "deployment": deployment,
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
        self,
        causal_entry: dict | None,
        *,
        refuted: bool = False,
        task_family: str | None = None,
        site: str | None = None,
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

        The exactness gate then decides what the class estimate may be
        *deployed* for — presence and evidence are different facts:

        - the action's own measured contrast (``action_exact``) is checked
          first and can only veto or outrank: established-harmful on *this*
          action beats a beneficial class, and a supported contrast that
          has not established benefit is exact evidence that we do not
          know — not an invitation for class data to override it;
        - ``exact`` generalization (the action identity itself randomized
          under the enforced signature) deploys on the action's own
          established effect, or — when its own data is thin — on the
          documented hierarchical shrinkage estimator;
        - ``same_context_class`` bridges to active authority only through
          a completed contextual canary of this action (``_action_confirmed``:
          the store's own in-context randomization record, or a signed /
          operator-attested confirmation), then through the same
          shrinkage rule — and when the store has no per-action counters
          at all, on the class estimate itself under the ``class_evidence``
          estimator label.

        ``None`` (an entry built before generalization annotation existed)
        or an unrecognized value fails closed.
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
        level = causal_entry.get("generalization_level")
        if level is not None and str(level) not in self.GENERALIZATION_LEVELS:
            return "unverified_generalization"
        action_exact = causal_entry.get("action_exact")
        if isinstance(action_exact, dict):
            # The action's own measured data answers first and can only
            # ever veto or outrank the class estimate: an established-
            # harmful verdict on *this* action is authoritative, and a
            # supported but undecided contrast means the action's own
            # evidence says "unknown" — class data cannot substitute.
            if action_exact.get("effect_status") == "harmful":
                return "harmful"
            if action_exact.get("support_sufficient"):
                if action_exact.get("effect_status") == "beneficial":
                    return None
                return "unresolved"
        confirmed = self._action_confirmed(causal_entry, task_family, site)
        if str(level or "") not in set(self.active_generalization_levels):
            if not (level == "same_context_class" and confirmed):
                return "unverified_generalization"
        # An active level was reached (``exact`` by default). ``exact``
        # presence is itself the in-context canary record, so a store that
        # predates per-action counters degrades to the confirmed path
        # rather than inventing support — the ``class_evidence`` estimator
        # reports that weaker basis honestly.
        return self._shrinkage_gate(causal_entry)

    def _shrinkage_gate(self, causal_entry: dict) -> str | None:
        """Hierarchical shrinkage deployment rule for thin per-action data.

        When the action's own contrast exists but cannot yet answer, the
        deployed estimate pools it toward the established class contrast:

            ``δ̂ = (w·δ_action + κ·δ_class) / (w + κ)``

        with ``w = min(candidate_ess, control_ess)`` of the action contrast
        and ``κ = CounterfactualTrials.MIN_ESS`` — the class estimate enters
        as a prior worth one MIN_ESS cell. Documented assumptions: the
        class effect is already established beneficial by the gates above;
        the action's own point estimate must not contradict it (``δ_action
        ≥ 0`` — a measured negative delta means the action's own data
        refutes the generalization and the bridge closes); and the shrunk
        estimate must still clear ``min_causal_delta``. Without per-action
        counters (``action_exact is None``) the rule degenerates to the
        class estimate under the confirmation that already fired — the
        ``class_evidence`` estimator.
        """
        action_exact = causal_entry.get("action_exact")
        delta_class = float(causal_entry.get("expected_delta") or 0.0)
        if not isinstance(action_exact, dict):
            return None if delta_class > self.min_causal_delta else "below_effect_threshold"
        delta_action = action_exact.get("delta")
        if delta_action is None or float(delta_action) < 0.0:
            # The action's own measured outcomes contradict the class
            # claim — generalization is not verified.
            return "unverified_generalization"
        w = min(
            float(action_exact["candidate"]["ess"]),
            float(action_exact["control"]["ess"]),
        )
        kappa = _MIN_ESS
        shrunk = (w * float(delta_action) + kappa * delta_class) / (w + kappa)
        return None if shrunk > self.min_causal_delta else "below_effect_threshold"

    def _deployment_estimate(self, causal_entry: dict) -> dict:
        """The estimate an executable entry actually deploys on.

        ``estimator`` names which evidence answered — ``exact_action``
        (the action's own established contrast), ``hierarchical_shrinkage``
        (thin per-action data pooled toward the class effect), or
        ``class_evidence`` (no per-action counters; the contextual-canary
        bridge attests the action under the class estimate). The deployed
        probability is the measured control rate plus *that* delta.
        """
        control_p = causal_entry.get("control_p")
        try:
            control_p = float(control_p)
            if not math.isfinite(control_p):
                control_p = None
        except (TypeError, ValueError):
            control_p = None

        def _p(delta):
            if control_p is None or delta is None:
                return None
            return min(max(control_p + float(delta), 0.0), 1.0)

        delta_class = float(causal_entry.get("expected_delta") or 0.0)
        action_exact = causal_entry.get("action_exact")
        if (
            isinstance(action_exact, dict)
            and action_exact.get("support_sufficient")
            and action_exact.get("effect_status") == "beneficial"
        ):
            delta = float(action_exact.get("delta") or 0.0)
            return {
                "estimator": "exact_action",
                "delta": delta,
                "p_progress": _p(delta),
            }
        if not isinstance(action_exact, dict):
            return {
                "estimator": "class_evidence",
                "delta": delta_class,
                "p_progress": _p(delta_class),
            }
        w = min(
            float(action_exact["candidate"]["ess"]),
            float(action_exact["control"]["ess"]),
        )
        shrunk = (
            w * float(action_exact.get("delta") or 0.0) + _MIN_ESS * delta_class
        ) / (w + _MIN_ESS)
        return {
            "estimator": "hierarchical_shrinkage",
            "delta": shrunk,
            "p_progress": _p(shrunk),
        }

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
            "deployment": top.get("deployment"),
            "deployment_score": top.get("deployment_score"),
            "experiment_priority_score": top.get("experiment_priority_score"),
            "source": top["source"],
            "mode": self.mode,
            "executable": top["executable"],
            "execution_blocker": top.get("execution_blocker"),
        }
