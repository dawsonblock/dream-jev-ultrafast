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
from .model import DecisionBackend, SystemOneBackend
from .policy import DefaultActionPolicy, PolicyDecision

__version__ = "0.4.0"

__all__ = [
    "Agent",
    "Browser",
    "DecisionBackend",
    "SystemOneBackend",
    "DefaultActionPolicy",
    "PolicyDecision",
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
]
