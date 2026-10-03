"""Jev Ultrafast: finite browser execution plus bounded DREAM replay improvement."""

from .dream import (
    CanaryEvidence,
    CanaryGate,
    CanaryMetrics,
    DreamImprover,
    ExperienceStore,
    ExplorationPolicy,
    HealthGate,
    ObjectiveWeights,
    PolicyRegistry,
    PromotionGate,
    ReplaySimulator,
    ReplayWorld,
)
from .dreamlearn import (
    CausalChoicePolicy,
    ChoiceModel,
    CostModel,
    CounterfactualTrials,
    ExperimentScheduler,
    OutcomeModel,
    TrialChoiceModel,
    UtilityWeights,
)
from .model import DecisionBackend, SystemOneBackend
from .policy import (
    DefaultActionPolicy,
    Effect,
    PolicyDecision,
    assess_payload,
    classify_effect,
    classify_payload,
)
from .signing import EvidenceSigner

__version__ = "0.9.2"

# Browser-coupled exports are bound lazily (Phase 10): ``Agent`` and
# ``Browser`` pull in the CDP runtime, so importing the package — or just
# the learning/dream layers — must not require a browser stack. The public
# surface is unchanged; first attribute access performs the import.
_BROWSER_BOUND = {
    "Agent": ".agent",
    "Browser": ".browser",
}


def __getattr__(name: str):
    module_name = _BROWSER_BOUND.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)

__all__ = [
    "Agent",
    "Browser",
    "DecisionBackend",
    "SystemOneBackend",
    "DefaultActionPolicy",
    "PolicyDecision",
    "Effect",
    "classify_effect",
    "classify_payload",
    "assess_payload",
    "ExplorationPolicy",
    "ExperienceStore",
    "ReplayWorld",
    "ReplaySimulator",
    "PromotionGate",
    "ObjectiveWeights",
    "DreamImprover",
    "PolicyRegistry",
    "CanaryMetrics",
    "CanaryEvidence",
    "CanaryGate",
    "HealthGate",
    "CostModel",
    "OutcomeModel",
    "ChoiceModel",
    "CounterfactualTrials",
    "TrialChoiceModel",
    "ExperimentScheduler",
    "CausalChoicePolicy",
    "UtilityWeights",
    "EvidenceSigner",
]
