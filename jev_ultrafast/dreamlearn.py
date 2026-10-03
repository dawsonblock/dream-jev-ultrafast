"""Learned auxiliary models for DREAM-Jev hypothesis prioritization.

These are Level-1/Level-2 learners in the DREAM stack:

- Level 0 (authority): empirical replay of recorded trajectories plus bound
  live canaries. Only Level-0 evidence can stage or promote a policy.
- Level 1 (``CostModel``): estimates how token and latency cost scale with the
  offered candidate count, anchored to each recorded transition's real values.
  It reorders which replay-qualified candidate is selected for staging; it can
  never make a failing candidate pass a gate.
- Level 2 (``OutcomeModel``): a bucketed, Beta-smoothed estimate of coarse
  transition effects (did the page change?). It annotates candidates with prior
  expectations for experiment design; it is never consulted by a gate.
- Level 3 (``ChoiceModel``): an uncertainty-aware counterfactual choice prior
  trained on trajectory success — the run's final independently verified
  outcome labels every step of that run — over the action's *offered* rank.
  It predicts
  under the candidate policy's *own* recomputed offered rank and proposes
  which offered action it would prefer (posterior mean plus an exploration
  bonus on posterior stddev), abstaining via an explicit ``confident`` flag on
  sparse cells. A proposal is annotation metadata only — never an outcome,
  never evidence, never a gate input. A counterfactual "Candidate 17 would
  have succeeded" can only be tested by actually running Candidate 17; real
  executions qualify, priors merely propose.
- Level 4 (``CounterfactualTrials``): intention-to-treat effect estimates over
  real randomized arm assignments recorded before the authority plane. The
  unit of analysis is the assignment, not the executed step — denied,
  rejected, and stale trials count toward their arm — while runs that never
  produced a measured outcome are censored (with a structured termination
  reason) rather than counted as failures. Support and effect certainty are
  separate: ``support_sufficient`` is the weighted-sample floor,
  ``effect_status`` is the sign established by an anytime-valid confidence
  sequence (multiplicity-corrected across the tracked hypothesis family),
  and censoring is handled twice — rate-imbalance gates plus Manski
  worst/best-case bounds that must agree with the sequence.
- Level 5 (``TrialChoiceModel``): the causal decision prior. It proposes only
  divergences whose randomized evidence *established* a beneficial effect,
  treats only established-harmful divergences as refuted, and keeps its
  provenance (``source: "randomized"``) separate from the observational
  ``ChoiceModel`` forever. ``ExperimentScheduler`` ranks unresolved
  hypotheses by expected information gain per trial cost, and
  ``CausalChoicePolicy`` combines the channels under operator-gated modes
  (shadow → canary → active) — always as prioritization, never as authority.

No model may fabricate a model choice or a browser outcome, and none is part
of the trusted evidence path.
"""

import sys as _sys
import types as _types
from typing import Iterable  # noqa: F401  (facade: re-exported module attr)

from jev_ultrafast import _learning as _impl_pkg
from jev_ultrafast._learning import *  # noqa: F401,F403
from jev_ultrafast._learning import __all__ as __all__

# Compatibility: before the internal split, every implementation name and
# every imported module lived in this module's namespace, so callers could
# monkeypatch e.g. ``dream.time`` or ``dream.ReplaySimulator`` and have the
# implementation observe the patch. Re-expose every name bound in any
# implementation submodule here and forward attribute writes back to every
# submodule that binds the name, preserving that behavior.
_OWNERS = {}
for _mod in _impl_pkg._MODULES:
    for _n in vars(_mod):
        _OWNERS.setdefault(_n, []).append(_mod)

_self = _sys.modules[__name__]
for _n, _owners in _OWNERS.items():
    if _n not in _self.__dict__:
        _self.__dict__[_n] = getattr(_owners[0], _n)


class _FacadeModule(_types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for _owner in _OWNERS.get(name, ()):
            setattr(_owner, name, value)


_self.__class__ = _FacadeModule
_self.__dict__.pop("TYPE_CHECKING", None)
del _sys, _types, _impl_pkg, _n, _owners, _mod, _self, _FacadeModule
