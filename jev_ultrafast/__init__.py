"""Jev Ultrafast: finite browser execution plus bounded DREAM replay improvement."""

from .agent import Agent
from .browser import Browser
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
from .dreamlearn import ChoiceModel, CostModel, CounterfactualTrials, OutcomeModel
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

__version__ = "0.8.2"

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
    "EvidenceSigner",
]
