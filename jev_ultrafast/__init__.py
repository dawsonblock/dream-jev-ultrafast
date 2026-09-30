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
from .dreamlearn import CostModel, OutcomeModel
from .model import DecisionBackend, SystemOneBackend
from .policy import DefaultActionPolicy, Effect, PolicyDecision, classify_effect
from .signing import EvidenceSigner

__version__ = "0.5.0"

__all__ = [
    "Agent",
    "Browser",
    "DecisionBackend",
    "SystemOneBackend",
    "DefaultActionPolicy",
    "PolicyDecision",
    "Effect",
    "classify_effect",
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
    "EvidenceSigner",
]
